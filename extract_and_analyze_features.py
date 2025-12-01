# python extract_and_analyze_features.py --config tools/config.yaml --single_gpu
import argparse
import logging
import os
import pprint
import json
import numpy as np
import torch
import torch.distributed as dist
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from utils.dist_helper import setup_distributed
from utils.misc_helper import (
    create_logger,
    get_current_time,
    load_state,
    set_random_seed,
    update_config,
)

parser = argparse.ArgumentParser(description="UniAD Feature Analysis")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument("--single_gpu", action="store_true", help="Use single GPU mode")
parser.add_argument("--save_dir", default="./analysis_results", help="Dir to save statistics")


def calculate_statistics(feature_tensor, logger, save_dir):
    """
    Tính toán Mean, Std, Range và các khoảng Percentile (CI).
    feature_tensor: Tensor chứa toàn bộ feature_align [N, C, H, W]
    """
    logger.info("Starting statistical analysis (converting to numpy)...")
    
    # Flatten toàn bộ dữ liệu thành 1 mảng 1 chiều để tính toán global
    # Việc này có thể tốn RAM nếu dataset quá lớn
    all_values = feature_tensor.flatten().cpu().numpy()
    
    # 1. Thống kê cơ bản
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    results = {
        "global_mean": float(mean_val),
        "global_std": float(std_val),
        "global_min": float(min_val),
        "global_max": float(max_val),
        "percentiles": {}
    }

    logger.info("=" * 50)
    logger.info(">>> FEATURE ALIGN STATISTICS <<<")
    logger.info("=" * 50)
    logger.info(f"Total parameters analyzed: {len(all_values)}")
    logger.info(f"Global Mean: {mean_val:.6f}")
    logger.info(f"Global Std:  {std_val:.6f}")
    logger.info(f"Global Range (Min - Max): [{min_val:.6f}, {max_val:.6f}]")
    logger.info("-" * 50)

    # 2. Tính toán các khoảng tin cậy (Confidence Intervals - CI)
    # Ví dụ: 98% CI nghĩa là bỏ 1% thấp nhất và 1% cao nhất (P1 - P99)
    cis = [98, 95, 90, 85, 80, 75, 70]
    
    logger.info(">>> PERCENTILE RANGES (Confidence Intervals) <<<")
    for ci in cis:
        # Tính phần đuôi cần loại bỏ
        tail = (100 - ci) / 2.0
        lower_p = tail
        upper_p = 100 - tail
        
        val_lower = np.percentile(all_values, lower_p)
        val_upper = np.percentile(all_values, upper_p)
        
        # --- CODE ĐÃ THAY ĐỔI / THÊM MỚI ---
        # Lọc các giá trị nằm trong khoảng CI
        filtered_values = all_values[(all_values >= val_lower) & (all_values <= val_upper)]
        
        # Tính Mean của các giá trị trong khoảng CI
        # Đây là Mean của "phần giữa" của dữ liệu, không tính các giá trị ở "đuôi"
        ci_mean = np.mean(filtered_values)
        # -----------------------------------
        
        key = f"{ci}%_CI"
        results["percentiles"][key] = {
            "lower_percentile": lower_p,
            "upper_percentile": upper_p,
            "min": float(val_lower),
            "max": float(val_upper),
            "mean": float(ci_mean) # Đã thêm Mean của CI
        }
        
        logger.info(f"{ci}% CI Range (P{lower_p:04.1f} - P{upper_p:04.1f}): Min = {val_lower:.6f}, Max = {val_upper:.6f}, Mean = {ci_mean:.6f}") # Cập nhật Log

    logger.info("=" * 50)

    # Lưu kết quả ra file JSON
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "feature_stats.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Statistics saved to: {json_path}")


def main():
    global args, config
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # --- 1. SETUP MÔI TRƯỜNG (Giữ nguyên) ---
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

    # Setup paths
    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        current_time = get_current_time()
        # Tạo logger riêng cho việc analysis
        logger = create_logger(
            "global_logger", os.path.join(args.save_dir, f"log_analysis_{current_time}.txt")
        )
        logger.info("args: {}".format(pprint.pformat(args)))
    else:
        logger = logging.getLogger("global_logger")

    random_seed = config.get("random_seed", None)
    if random_seed:
        set_random_seed(random_seed, config.get("reproduce", None))

    # --- 2. KHỞI TẠO MODEL (Giữ nguyên ModelHelper để tránh lỗi) ---
    model = ModelHelper(config.net)
    model.cuda()
    
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

    # Load checkpoint nếu có (để lấy feature từ model đã train)
    # Nếu không có load_path, nó sẽ dùng weights khởi tạo ngẫu nhiên/pre-trained của backbone
    load_path = config.saver.get("load_path", None)
    if load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)
        if rank == 0:
            logger.info(f"Loaded model weights from: {load_path}")

    # --- 3. BUILD DATALOADER ---
    # Sử dụng hàm build_dataloader gốc, đảm bảo distributed=False nếu chạy single GPU
    train_loader, test_loader = build_dataloader(config.dataset, distributed=not single_gpu_mode)
    data_use = test_loader
    # ==============================================================================
    # BẮT ĐẦU QUÁ TRÌNH TRÍCH XUẤT VÀ TÍNH TOÁN
    # ==============================================================================
    
    if rank == 0:
        logger.info(f"Starting feature extraction on {len(data_use)} batches...")

    # Đặt model về chế độ eval để cố định BatchNorm và tắt Dropout
    model.eval()
    
    all_features = []

    # Tắt Gradient hoàn toàn
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # Forward pass thông thường qua ModelHelper
            # ModelHelper sẽ tự động gọi backbone -> neck -> reconstruction
            # UniADMemory (reconstruction) sẽ trả về dict chứa "feature_align"
            outputs = model(input)
            
            # Lấy backbone output (feature_align) từ output dictionary
            # Lưu ý: UniADMemory trả về "feature_align" chính là output của Neck
            if "feature_align" in outputs:
                feature_align = outputs["feature_align"] # [B, C, H, W]
                
                # Đưa về CPU ngay để tiết kiệm VRAM
                all_features.append(feature_align.cpu())
            else:
                if rank == 0 and i == 0:
                    logger.warning("Key 'feature_align' not found in model outputs!")

            # Log tiến độ
            if rank == 0 and (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(data_use)} batches.")

    # ==============================================================================
    # TỔNG HỢP VÀ TÍNH STATS (Chỉ thực hiện trên Rank 0)
    # ==============================================================================
    
    if rank == 0:
        if len(all_features) > 0:
            logger.info("Concatenating features...")
            # Gộp tất cả các batch thành 1 tensor lớn
            full_feature_tensor = torch.cat(all_features, dim=0)
            logger.info(f"Full feature tensor shape: {full_feature_tensor.shape}")

            # Lưu tensor feature ra file (để dùng lại sau này nếu cần)
            tensor_path = os.path.join(args.save_dir, "backbone_feature_align.pth")
            torch.save(full_feature_tensor, tensor_path)
            logger.info(f"Saved feature tensor to: {tensor_path}")

            # >>> TÍNH TOÁN THỐNG KÊ <<<
            calculate_statistics(full_feature_tensor, logger, args.save_dir)
            
        else:
            logger.error("No features were extracted!")

    # Đồng bộ các process trước khi kết thúc
    if not single_gpu_mode:
        dist.barrier()

if __name__ == "__main__":
    main()