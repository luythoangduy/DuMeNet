import glob
import logging
import os

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from skimage import measure
from multiprocessing import Pool
from functools import partial

def worker_cal_pro_one_threshold(inputs):
    """
    Hàm worker để tính PRO và FPR tại một ngưỡng cụ thể.
    Inputs: (preds, masks, threshold)
    """
    preds, masks, th = inputs
    
    # 1. Binarize predictions
    binary_amaps = np.zeros_like(preds, dtype=np.bool_)
    binary_amaps[preds > th] = 1  # Note: logic gốc là > th thì = 1
    
    # 2. Calculate PRO
    pro = []
    # masks là (N, H, W), iterate qua từng ảnh
    for binary_amap, mask in zip(binary_amaps, masks):
        # Chỉ tính nếu ảnh có ground truth defect
        if mask.sum() == 0:
            continue
            
        label_mask = measure.label(mask)
        props = measure.regionprops(label_mask)
        
        for region in props:
            # Đếm số pixel dự đoán đúng trong vùng defect này
            tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
            # Tính tỷ lệ bao phủ
            pro.append(tp_pixels / region.area)
    
    # Nếu không có vùng defect nào trong cả batch
    mean_pro = np.mean(pro) if pro else 0.0
    
    # 3. Calculate FPR (trên vùng background)
    inverse_masks = 1 - masks
    fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
    fpr = fp_pixels / inverse_masks.sum() if inverse_masks.sum() > 0 else 0.0
    
    return mean_pro, fpr

def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"])
    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    return fileinfos, preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int64)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc

class EvalProAUC:
    def __init__(self, data_meta, max_step=200, expect_fpr=0.3, **kwargs):
        # Flatten không cần thiết cho PRO vì cần cấu trúc không gian (H, W),
        # nên ta dùng raw data từ data_meta
        self.preds = data_meta.preds # N x H x W
        self.masks = data_meta.masks # N x H x W
        
        # Đảm bảo masks là nhị phân 0/1
        self.masks[self.masks > 0] = 1
        self.max_step = max_step
        self.expect_fpr = expect_fpr # Mặc định 0.3
        
    def eval_auc(self):
        # Lấy giá trị min và max TRỰC TIẾP từ preds (amaps trong code gốc)
        min_th, max_th = self.preds.min(), self.preds.max()
        
        # Tính delta và tạo thresholds
        # Note: Code gốc dùng np.arange, nhưng np.linspace đảm bảo số bước chính xác hơn
        delta = (max_th - min_th) / self.max_step
        
        # Tạo danh sách thresholds theo cách của code gốc: arange không bao gồm max_th
        # Sử dụng một epsilon nhỏ để đảm bảo không bị lỗi dấu phẩy động
        thresholds = np.arange(min_th, max_th, delta)
        
        # Nếu ngưỡng không được tạo (min == max), không cần tính toán
        if len(thresholds) == 0:
            return 0.0

        # 1. Chuẩn bị input cho worker (Sử dụng preds GỐC, không cần normalize)
        inputs = [(self.preds, self.masks, th) for th in thresholds]
        
        # 2. Tính toán song song (Multiprocessing)
        with Pool(processes=8) as pool:
            results = pool.map(worker_cal_pro_one_threshold, inputs)
            
        pros = [item[0] for item in results]
        fprs = [item[1] for item in results]
        
        # 3. Chuyển đổi sang numpy và sắp xếp theo FPR
        pros, fprs = np.array(pros), np.array(fprs)
        sorted_idxs = np.argsort(fprs)
        sorted_fprs = fprs[sorted_idxs]
        sorted_pros = pros[sorted_idxs]
        
        # 4. Lọc và Chuẩn hóa (Theo logic code gốc)
        
        # Lọc: Chỉ lấy các cặp (PRO, FPR) có FPR < expect_fpr (0.3)
        idxes = sorted_fprs < self.expect_fpr
        filtered_fprs = sorted_fprs[idxes]
        filtered_pros = sorted_pros[idxes]
        
        # Xử lý trường hợp không có điểm nào trong phạm vi
        if len(filtered_fprs) == 0:
            return 0.0

        # Chuẩn hóa: Chuẩn hóa trục FPR từ [min(FPR), max(FPR)] về [0, 1]
        fpr_min = filtered_fprs.min()
        fpr_max = filtered_fprs.max()
        
        # Thêm kiểm tra tránh chia cho 0
        if fpr_max == fpr_min:
            return filtered_pros.mean() 
        
        normalized_fprs = (filtered_fprs - fpr_min) / (fpr_max - fpr_min)
        
        # 5. Tính AU-PRO (AUC trên FPR đã chuẩn hóa)
        pro_auc = metrics.auc(normalized_fprs, filtered_pros)
        
        return pro_auc


eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
    "pro": EvalProAUC,
}


def performances(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls)

        # auc
        if config.get("auc", None):
            for metric in config.auc:
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc

    if config.get("auc", None):
        for metric in config.auc:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
