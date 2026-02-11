import argparse
import logging
import os
import pprint
import time

import cv2
import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel # Thêm dòng này
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
from utils.optimizer_helper import get_optimizer

parser = argparse.ArgumentParser(description="UniAD Framework")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("--class_name", default="")
parser.add_argument("-v", "--visualization", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument("--single_gpu", action="store_true", help="Use single GPU mode") # Thêm dòng này


class_name_list = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def main():
    global args, config, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Logic nhận diện Single GPU mode
    single_gpu_mode = args.single_gpu or not torch.distributed.is_available() or not os.environ.get('WORLD_SIZE')

    config.dataset.train.meta_file = config.dataset.train.meta_file.replace(
        "{class_name}", args.class_name
    )
    config.port = config["port"] + class_name_list.index(args.class_name)
    
    if single_gpu_mode: # Thêm điều kiện này
        rank, world_size = 0, 1
    else:
        rank, world_size = setup_distributed(port=config.port)
        
    config = update_config(config)

    config.exp_path = os.path.join(os.path.dirname(args.config), args.class_name)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)
        
    # create model
    model = ModelHelper(config.net)
    model.cuda()
    
    # Bọc model thông minh (DDP cho đa GPU, DataParallel cho đơn GPU)
    if single_gpu_mode:
        model = DataParallel(model)
        use_ddp = False
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        use_ddp = True

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    
    # Parameters update (luôn dùng model.module để truy cập các lớp gốc)
    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    best_metric = float("inf")
    last_epoch = 0

    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        load_state(load_path, model)

    # Chỉnh lại distributed=False nếu là single_gpu
    train_loader, val_loader = build_dataloader(config.dataset, distributed=not single_gpu_mode)

    if args.visualization:
        # vis_rec(train_loader, model)
        vis_rec(val_loader, model)
        return

    criterion = build_criterion(config.criterion)

    for epoch in range(last_epoch, config.trainer.max_epoch):
        if not single_gpu_mode: # Chỉ gọi set_epoch nếu chạy distributed
            train_loader.sampler.set_epoch(epoch)
            
        last_iter = epoch * len(train_loader)
        train_loss = train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
            single_gpu_mode # Truyền thêm flag
        )
        lr_scheduler.step(epoch)

        if rank == 0:
            is_best = train_loss <= best_metric
            best_metric = min(train_loss, best_metric)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": config.net,
                    "state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                config,
            )

        if config.visualization:
            if (epoch + 1) % config.visualization.vis_freq_epoch == 0:
                vis_rec(train_loader, model)


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
    single_gpu_mode # Thêm tham số
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()
    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    world_size = 1 if single_gpu_mode else dist.get_world_size()
    rank = 0 if single_gpu_mode else dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()

    train_loss = 0
    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]
        data_time.update(time.time() - end)

        outputs = model(input)
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
            
        reduced_loss = loss.clone()
        if not single_gpu_mode: # Chỉ reduce nếu là distributed
            dist.all_reduce(reduced_loss)
            reduced_loss = reduced_loss / world_size
            
        losses.update(reduced_loss.item())
        train_loss += reduced_loss.item()

        optimizer.zero_grad()
        loss.backward()
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\tIter: [{2}/{3}]\tLoss {loss.val:.5f} ({loss.avg:.5f})\tLR {lr:.5f}".format(
                    epoch + 1, config.trainer.max_epoch, curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    loss=losses, lr=current_lr
                )
            )
        end = time.time()

    return train_loss / len(train_loader)


def vis_rec(loader, model):
    model.eval()
    
    # Sửa: Dùng view(3, 1, 1) thay vì (1, 3, 1, 1) để khớp với Tensor 3D trong loop
    pixel_mean = torch.tensor(config.dataset.pixel_mean).cuda().view(3, 1, 1)
    pixel_std = torch.tensor(config.dataset.pixel_std).cuda().view(3, 1, 1)

    with torch.no_grad():
        for i, input in enumerate(loader):
            # forward
            outputs = model(input)
            filenames = outputs["filename"]
            images = outputs["image"]           # [B, 3, H, W]
            image_recs = outputs["image_rec"]   # [B, 3, H, W]
            clsnames = outputs["clsname"]

            for filename, image, image_rec, clasname in zip(
                filenames, images, image_recs, clsnames
            ):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                
                vis_dir = os.path.join(config.visualization.vis_dir, clasname, defename)
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, filename_ + ".jpg")

                # De-normalize về 0-255
                image = (image * pixel_std + pixel_mean) * 255
                image_rec = (image_rec * pixel_std + pixel_mean) * 255
                
                # Ghép ảnh: image và image_rec đều là [3, 224, 224]
                # Ghép theo dim=2 (chiều rộng) để nhìn ảnh nằm ngang cho đẹp
                combined = torch.cat([image, image_rec], dim=2) # Kết quả: [3, 224, 448]
                
                # Chuyển về định dạng ảnh [H, W, C] để lưu
                combined = combined.permute(1, 2, 0).cpu().numpy().astype('uint8')
                
                # Chuyển RGB sang BGR cho OpenCV
                combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                cv2.imwrite(vis_path, combined)

if __name__ == "__main__":
    main()
