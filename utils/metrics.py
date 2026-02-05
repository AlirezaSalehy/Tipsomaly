import torch
import numpy as np
from skimage import measure
from torchmetrics import AUROC, AveragePrecision
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve

def calc_f1_max(gt, pr):
    precisions, recalls, _ = precision_recall_curve(gt, pr)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    return np.max(f1_scores[np.isfinite(f1_scores)])


def cal_pro_score_gpu(device, masks, amaps, max_step=200, expect_fpr=0.3):
    # GPU implementation using PyTorch
    if not torch.is_tensor(amaps):
        amaps = torch.tensor(amaps)
    amaps = amaps.to(device)
    masks = masks.to(device)
    
    binary_amaps = torch.zeros_like(amaps, dtype=torch.bool, device=device)
    min_th, max_th = amaps.min().item(), amaps.max().item()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    
    regionprops_list = [measure.regionprops(measure.label(mask.cpu().numpy())) for mask in masks]
    coords_list = [[(region.coords[:, 0], region.coords[:, 1], len(region.coords)) for region in regionprops] for regionprops in regionprops_list]
    inverse_masks = 1 - masks
    tn_pixel = inverse_masks.sum().item() # Pixels that truly has the label of 0
    for th in torch.arange(min_th, max_th, delta, device=device):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        
        for binary_amap, regions_coords in zip(binary_amaps, coords_list):
            for coords in regions_coords:
                tp_pixels = binary_amap[coords[0], coords[1]].sum().item()
                pro.append(tp_pixels / coords[2])
        
        fp_pixels = torch.logical_and(inverse_masks, binary_amaps).sum().item()
        fpr = fp_pixels / tn_pixel
        pros.append(np.mean(pro))
        fprs.append(fpr)
        ths.append(th.item())
    
    pros, fprs, ths = torch.tensor(pros, device=device), torch.tensor(fprs, device=device), torch.tensor(ths, device=device)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs.cpu().numpy(), pros[idxes].cpu().numpy())
    return pro_auc

def image_level_metrics(prd, lbl, metric):
    if len(np.unique(lbl)) < 2:
        print("only one class present, can not calculate image metrics")
        return 0
    
    if metric == 'auroc':
        performance = roc_auc_score(lbl, prd)
    elif metric == 'ap':
        performance = average_precision_score(lbl, prd)
    elif metric == 'f1-max':
        performance = calc_f1_max(lbl, prd)
    return performance

def pixel_level_metrics(device, prd, lbl, metric):
    if torch.unique(lbl).numel() < 2:
        print("only one class present, can not calculate pixel metrics")
        return 0
    
    if metric == 'auroc':
        performance = AUROC(task="binary")(prd, lbl.to(dtype=torch.long)).item()
        
    elif metric == 'aupro':
        if len(lbl.shape) == 4:
            lbl = lbl.squeeze(1)
        if len(prd.shape) == 4:
            prd = prd.squeeze(1)
        performance = cal_pro_score_gpu(device, lbl, prd)
        
    elif metric == 'ap':     
        performance = AveragePrecision(task="binary")(prd, lbl.to(dtype=torch.long)).item()
        
    elif metric == 'f1-max':
        performance = calc_f1_max(lbl.cpu().ravel(), prd.cpu().ravel())
    return performance
