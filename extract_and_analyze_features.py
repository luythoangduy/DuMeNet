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
    Tính toán thống kê:
    1. Global: Mean, Std, Range, K-global trên toàn bộ tensor.
    2. Channel-wise: Mean, Std, Range, K-channel, và K-CI cho từng CI riêng biệt của mỗi channel.
    feature_tensor: Tensor chứa toàn bộ feature_align [N, C, H, W]
    """
    logger.info("Starting statistical analysis (converting to numpy)...")
    
    # Kích thước tensor
    N, C, H, W = feature_tensor.shape
    elements_per_channel = N * H * W
    cis = [98, 95, 90, 85, 80, 75, 70] 
    
    # --------------------------------------------------------------------------
    # I. GLOBAL STATISTICS (Tính trên toàn bộ N*C*H*W giá trị)
    # --------------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info(">>> FEATURE ALIGN GLOBAL STATISTICS <<<")
    
    # Flatten toàn bộ dữ liệu (Chỉ làm 1 lần để tính Global)
    all_values = feature_tensor.flatten().cpu().numpy()
    
    global_mean = np.mean(all_values)
    global_std = np.std(all_values)
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    global_range = global_max - global_min
    
    # Tính K-global
    if global_range > 1e-6:
        k_global = 8.0 / global_range
        k_global_rounded = round(k_global, 3)
    else:
        k_global_rounded = float('inf')

    global_results = {
        "feature_shape": [N, C, H, W],
        "global_mean": float(global_mean),
        "global_std": float(global_std),
        "global_min": float(global_min),
        "global_max": float(global_max),
        "global_range": float(global_range),
        "global_k_value": k_global_rounded,
    }
    
    logger.info(f"Total elements analyzed: {len(all_values)}")
    logger.info(f"Global Mean: {global_mean:.6f}, Global Std: {global_std:.6f}")
    logger.info(f"Global Range (Min - Max): [{global_min:.6f}, {global_max:.6f}]")
    logger.info(f"Calculated K-Global: {k_global_rounded}")
    logger.info("=" * 50)

    # --------------------------------------------------------------------------
    # II. CHANNEL-WISE STATISTICS (Tính trên từng channel)
    # --------------------------------------------------------------------------
    logger.info(f">>> FEATURE ALIGN CHANNEL-WISE STATISTICS ({C} Channels) <<<")
    
    # Reshape tensor về shape [C, N*H*W]
    feature_np = feature_tensor.permute(1, 0, 2, 3).reshape(C, -1).cpu().numpy()
    channel_stats_list = []
    
    # Lặp qua từng channel
    for channel_idx in range(C):
        channel_values = feature_np[channel_idx]
        
        # Thống kê cơ bản của channel
        mean_val = np.mean(channel_values)
        std_val = np.std(channel_values)
        min_val = np.min(channel_values)
        max_val = np.max(channel_values)
        channel_range = max_val - min_val

        # Tính K-channel (K dựa trên Min/Max toàn channel)
        if channel_range > 1e-6:
            k_channel = 8.0 / channel_range
            k_channel_rounded = round(k_channel, 3)
        else:
            k_channel_rounded = float('inf')

        channel_result = {
            "channel_idx": channel_idx,
            "mean": float(mean_val),
            "std": float(std_val),
            "min": float(min_val),
            "max": float(max_val),
            "range": float(channel_range),
            "k_channel_value": k_channel_rounded, # K-value dựa trên Min/Max của channel
            "percentiles": {}
        }

        # Log cơ bản
        if channel_idx < 5 or channel_idx == C - 1:
             logger.info(f"--- Channel {channel_idx} ---")
             logger.info(f"Mean: {mean_val:.6f}, Std: {std_val:.6f}")
             logger.info(f"Range: [{min_val:.6f}, {max_val:.6f}], K_channel: {k_channel_rounded}")
        
        # Tính toán các khoảng tin cậy (Confidence Intervals - CI) cho channel
        for ci in cis:
            tail = (100 - ci) / 2.0
            lower_p = tail
            upper_p = 100 - tail
            
            val_lower = np.percentile(channel_values, lower_p)
            val_upper = np.percentile(channel_values, upper_p)
            
            # Tính Mean của các giá trị trong khoảng CI
            filtered_values = channel_values[(channel_values >= val_lower) & (channel_values <= val_upper)]
            ci_mean = np.mean(filtered_values)
            ci_range = val_upper - val_lower
            
            # Tính K-CI (K riêng cho khoảng CI)
            if ci_range > 1e-6:
                k_ci = 8.0 / ci_range
                k_ci_rounded = round(k_ci, 3)
            else:
                k_ci_rounded = float('inf')

            key = f"{ci}%_CI"
            channel_result["percentiles"][key] = {
                "lower_percentile": lower_p,
                "upper_percentile": upper_p,
                "min": float(val_lower),
                "max": float(val_upper),
                "range": float(ci_range),
                "mean": float(ci_mean),
                "k_ci_value": k_ci_rounded # K-value riêng cho khoảng CI
            }
            
            if channel_idx < 5 or channel_idx == C - 1:
                logger.info(f"  {ci}% CI: Range = [{val_lower:.6f}, {val_upper:.6f}], K_CI = {k_ci_rounded}")

        channel_stats_list.append(channel_result)

    logger.info("=" * 50)
    logger.info(f"Successfully calculated statistics for {C} channels.")


    # Lưu kết quả ra file JSON
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "feature_stats_combined.json")
    
    # Kết hợp Global và Channel-wise stats
    final_results = {
        "global_stats": global_results,
        "channel_stats": channel_stats_list,
    }
    
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)
    logger.info(f"Combined statistics saved to: {json_path}")


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
    data_use = train_loader
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
        for i, input in enumerate(data_use):
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