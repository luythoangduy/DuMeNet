import argparse
import logging
import os
import pprint
import shutil
import time
import math
import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
    init_wandb,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound, visualize_single
# import setproctitle
# setproctitle.setproctitle("Minh Tri is training...")
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

parser = argparse.ArgumentParser(description="UniAD Framework")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument("--single_gpu", action="store_true", help="Use single GPU mode")


def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Determine if running in single GPU mode
    single_gpu_mode = args.single_gpu or not torch.distributed.is_available() or not os.environ.get('WORLD_SIZE')
    
    if single_gpu_mode:
        rank = 0
        world_size = 1
        print("Running in single GPU mode")
    else:
        config.port = config.get("port", None)
        rank, world_size = setup_distributed(port=config.port)
    
    print("config: {}".format(pprint.pformat(config)))
    config = update_config(config)

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    
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
        
        # Initialize wandb if configured
        wandb_run = init_wandb(config, args)
    else:
        tb_logger = None
        logger = None
        wandb_run = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()
    
    # Use DDP only for multi-GPU, DataParallel for single GPU with multiple devices
    if single_gpu_mode:
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        use_ddp = False
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
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
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    # Handle both DDP and DataParallel/single GPU cases
    model_for_params = model.module if (use_ddp or isinstance(model, DataParallel)) else model
    parameters = [
        {"params": getattr(model_for_params, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    # load model: auto_resume > resume_model > load_path
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
        if rank == 0 and logger:
            logger.info(f"Resumed training from epoch {last_epoch} with best metric {best_metric}")
            logger.info(f"Resume model path: {resume_model}")
            
            # Log resume information to wandb
            if wandb_run:
                wandb_run.log({
                    "resume/start_epoch": last_epoch,
                    "resume/best_metric": best_metric,
                    "resume/model_path": resume_model
                }, step=0)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)
        if rank == 0 and logger:
            logger.info(f"Loaded model from: {load_path}")
            
            # Log load information to wandb
            if wandb_run:
                wandb_run.log({
                    "load/model_path": load_path
                }, step=0)

    # Build dataloader - use distributed=False for single GPU
    train_loader, val_loader = build_dataloader(config.dataset, distributed=not single_gpu_mode)

    if args.evaluate:
        validate(val_loader, model, single_gpu_mode)
        return

    criterion = build_criterion(config.criterion)

    for epoch in range(last_epoch, config.trainer.max_epoch):
        # Log current epoch info at start of training
        if rank == 0 and epoch == last_epoch and logger:
            logger.info(f"Starting training from epoch {epoch + 1}/{config.trainer.max_epoch}")
            if wandb_run:
                wandb_run.log({
                    "training/start_epoch": epoch + 1,
                    "training/total_epochs": config.trainer.max_epoch,
                    "training/remaining_epochs": config.trainer.max_epoch - epoch
                }, step=epoch * len(train_loader))
        
        if not single_gpu_mode:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
            single_gpu_mode,
            use_ddp,
            wandb_run,
        )
        lr_scheduler.step(epoch)

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, model, single_gpu_mode, wandb_run, epoch)
            # only ret_metrics on rank0 is not empty
            if rank == 0:
                ret_key_metric = ret_metrics[key_metric]
                is_best = ret_key_metric >= best_metric
                best_metric = max(ret_key_metric, best_metric)
                # save_checkpoint(
                #     {
                #         "epoch": epoch + 1,
                #         "arch": config.net,
                #         "state_dict": model.state_dict(),
                #         "best_metric": best_metric,
                #         "optimizer": optimizer.state_dict(),
                #     },
                #     is_best,
                #     config,
                # )
                
                # Log best metric to wandb
                if wandb_run:
                    wandb_run.log({
                        f"best_{key_metric}": best_metric,
                        "epoch": epoch + 1,
                    })

    if rank == 0 and wandb_run:
        # Log final training summary
        wandb_run.log({
            "training/completed": True,
            "training/final_epoch": config.trainer.max_epoch,
            "training/final_best_metric": best_metric
        })
        wandb_run.finish()


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
    single_gpu_mode,
    use_ddp,
    wandb_run=None,
):

    log_freq = config.trainer.print_freq_step
    batch_time = AverageMeter(log_freq)
    data_time = AverageMeter(log_freq)
    losses = AverageMeter(log_freq)

    # --- KHAI BÁO BỘ TÍCH LŨY MỚI (CHIA TÁCH MEAN/STD và MIN/MAX) ---
    stats_to_accumulate = {}
    min_max_trackers = {} # Sử dụng dict này cho logic cực trị
    
    stat_names = ["backbone_output", "encoded_feature", "decoder_output_raw", "decoder_output_sigmoid"]
    mean_std_metrics = ["mean", "std"]
    min_max_metrics = ["min", "max"]

    for name in stat_names:
        # Khai báo Mean/Std Accumulators (AverageMeter)
        for metric in mean_std_metrics:
            key = f"{name}_{metric}"
            stats_to_accumulate[key] = AverageMeter(log_freq)
        
        # Khai báo Min/Max Trackers (Giá trị cực trị ban đầu)
        min_max_trackers[f"{name}_min"] = float('inf')
        min_max_trackers[f"{name}_max"] = float('-inf')

    model.train()
    # freeze selected layers
    model_for_freeze = model.module if (use_ddp or isinstance(model, DataParallel)) else model
    for layer in frozen_layers:
        module = getattr(model_for_freeze, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    if single_gpu_mode:
        world_size = 1
        rank = 0
    else:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        outputs = model(input)
        loss = 0
        if "learned_k" in outputs:
            k_val = outputs["learned_k"].item() # .item() để chuyển Tensor về số Python float
        else:
            k_val = 1.0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
        
        if single_gpu_mode:
            reduced_loss = loss.clone()
        else:
            reduced_loss = loss.clone()
            dist.all_reduce(reduced_loss)
            reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        # --- CẬP NHẬT BỘ TÍCH LŨY (MIN/MAX CỰC TRỊ, MEAN/STD TRUNG BÌNH) ---
        for name in stat_names:
            stat_dict_name = f"{name}_stats"
            stat_dict = outputs.get(stat_dict_name, None)
            
            if stat_dict is not None:
                # Cập nhật MEAN và STD (AverageMeter)
                for metric in mean_std_metrics:
                    key = f"{name}_{metric}"
                    value_to_update = stat_dict.get(key, float('nan'))
                    
                    if not math.isnan(value_to_update):
                        stats_to_accumulate[key].update(value_to_update)

                # Cập nhật MIN và MAX (Cực trị tuyệt đối)
                current_min = stat_dict.get(f"{name}_min", float('nan'))
                current_max = stat_dict.get(f"{name}_max", float('nan'))

                min_key = f"{name}_min"
                max_key = f"{name}_max"

                if not math.isnan(current_min):
                    min_max_trackers[min_key] = min(min_max_trackers[min_key], current_min)
                
                if not math.isnan(current_max):
                    min_max_trackers[max_key] = max(min_max_trackers[max_key], current_max)
        # -----------------------------------------------------------------

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            if tb_logger:
                tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
                tb_logger.add_scalar("lr", current_lr, curr_step + 1)
                tb_logger.flush()
            
            # Log to wandb
            if wandb_run:
                stats_to_log = {}
                
                # 1. Log MEAN và STD (Lấy .avg và reset AverageMeter)
                for key, avg_meter in stats_to_accumulate.items():
                    stats_to_log[f"stats/{key}"] = avg_meter.avg
                    avg_meter.reset()
                
                # 2. Log MIN và MAX (Lấy giá trị cực trị và reset tracker)
                for key, value in min_max_trackers.items():
                    stats_to_log[f"stats/{key}"] = value
                    
                    # Reset Tracker về giá trị khởi tạo
                    if key.endswith('_min'):
                        min_max_trackers[key] = float('inf')
                    elif key.endswith('_max'):
                        min_max_trackers[key] = float('-inf')

                # 3. Log tổng hợp WandB
                wandb_run.log({
                    "train/loss": losses.avg,
                    "train/lr": current_lr,
                    "train/epoch": epoch + (i + 1) / len(train_loader),
                    "step": curr_step + 1,
                    "train/sigmoid_k": k_val,
                    **stats_to_log
                })
                
                # 4. Reset các metric chung (Sau khi đã log giá trị .avg của chúng)
                losses.reset()
                batch_time.reset()
                data_time.reset()

            if logger:
                logger.info(
                    "Epoch: [{0}/{1}]\t"
                    "Iter: [{2}/{3}]\t"
                    "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                    "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                    "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                    "LR {lr:.5f}\t".format(
                        epoch + 1,
                        config.trainer.max_epoch,
                        curr_step + 1,
                        len(train_loader) * config.trainer.max_epoch,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=current_lr,
                    )
                )

        end = time.time()


def validate(val_loader, model, single_gpu_mode, wandb_run=None, epoch=None):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    if single_gpu_mode:
        rank = 0
    else:
        rank = dist.get_rank()
    
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    if not single_gpu_mode:
        dist.barrier()

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input)
            dump(config.evaluator.eval_dir, outputs)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0 and logger:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    # gather final results
    if single_gpu_mode:
        final_loss = losses.avg
        total_num = losses.count
    else:
        dist.barrier()
        total_num = torch.Tensor([losses.count]).cuda()
        loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
        dist.all_reduce(total_num, async_op=True)
        dist.all_reduce(loss_sum, async_op=True)
        final_loss = loss_sum.item() / total_num.item()
        total_num = total_num.item()

    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    if rank == 0:
        if logger:
            logger.info("Gathering final results ...")
            # total loss
            logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num))
        fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
        shutil.rmtree(config.evaluator.eval_dir)
        # evaluate, log & vis
        ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)
        
        # Log validation metrics to wandb
        if wandb_run and epoch is not None:
            wandb_metrics = {"val/loss": final_loss, "epoch": epoch + 1}
            for metric_name, metric_value in ret_metrics.items():
                wandb_metrics[f"val/{metric_name}"] = metric_value
            wandb_run.log(wandb_metrics)
        
        if args.evaluate and config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
        if args.evaluate and config.evaluator.get("vis_single", None):
            visualize_single(
                fileinfos,
                preds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
            )
    model.train()
    return ret_metrics


if __name__ == "__main__":
    main()