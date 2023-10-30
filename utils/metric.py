# %%
import torch
import torch.nn.functional as F
import numpy as np
# import cv2

#计算dice系数，输入为两张图像，返回是对应dice系数，loss需要1 - dice_coeff
def dice_coeff(pred, target, smooth = 0.0001):
    intersection = torch.sum(pred * target)
    return (2. * intersection + smooth) / (torch.sum(target) + torch.sum(pred) + smooth)


def dice_coef_3d(pred, target):
    pred = torch.argmax(pred, dim=1, keepdim=True).float()
    target = torch.gt(target, 0.5).float()
    
    smooth = 0.0001
    target = target.view(-1)
    pred = pred.view(-1)
    intersect = torch.sum(target * pred)
    dice = (2 * intersect + smooth) / (torch.sum(target) + torch.sum(pred) + smooth)

    return dice

def dice_coef_2d(pred, target):
    pred = torch.argmax(pred, dim=1, keepdim=True).float()
    target = torch.gt(target, 0.5).float()
    
    n = target.size(0)
    smooth = 1e-4

    target = target.view(n, -1)
    pred = pred.view(n, -1)
    intersect = torch.sum(target * pred, dim=-1)
    dice = (2 * intersect + smooth) / (torch.sum(target, dim=-1) + torch.sum(pred, dim=-1) + smooth)
    dice = torch.mean(dice)

    return dice


def dice_coef(pred, target):
    smooth = 1e-4

    target = target.reshape(-1)
    pred = pred.reshape(-1)
    intersect = np.sum(target * pred)
    dice = (2 * intersect + smooth) / (np.sum(target) + np.sum(pred) + smooth)

    return dice

#计算dice系数，输入为两张图像，返回是对应dice系数，loss需要1 - dice_coeff
def numpy_dice_coeff(pred, target, smooth = 0.):
    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (np.sum(target) + np.sum(pred) + smooth)

def positive(y_true):
    return np.sum((y_true == 1))

def negative(y_true):
    return np.sum((y_true == 0))

def true_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 1))

def false_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 1))

def true_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 0))

def false_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 0))

def accuracy(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s
    return np.sum((y_true == y_pred)) / sample_count

#warning
def sensitive(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    p = positive(y_true) + 1e-9
    return tp / p

def specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    n = negative(y_true) + 1e-9
    return tn / n

#warning
def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp + 1e-9)

def save_log(file_path, data_str):
    # print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def recall_test(y_true, y_pred,file_path):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    save_log(file_path,"tp in rec:"+str(tp)+"  ;fn in rec:"+str(fn))
    return tp / (tp + fn + 1e-9)

#warning
def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn + 1e-9)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = (2 * prec * reca) / (prec + reca + 1e-9)
    return fs

def IntersectionOverUnion(y_true, y_pred):
    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    fp =false_positive(y_true, y_pred)
    IoU = tp / (tp + fp + fn + 1e-9)
    return IoU

# class calculate_metric_percase():
#     def __init__(self, y_true, y_pred, smooth):
#         self.positive = 