import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        total_loss = F.binary_cross_entropy(pred.squeeze(), target.squeeze().float())
        return total_loss

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
    def forward(self, *inputs):
        pred, mask = tuple(inputs)
        import torch.nn.functional as F
        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        pred = torch.sigmoid(pred)
        inter = pred * mask
        union = pred + mask
        iou = 1 - (2. * inter + 1) / (union + 1)
        return (bce + iou).mean()

#ssformer
def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()


def dice_bce_loss(pred, mask):
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()

def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return (1 - Tversky) ** gamma

def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return bce + (1 - Tversky) ** gamma

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, 1, -1) # (N, 1, *)

        predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()
    
class Bce_iou_loss(nn.Module):

    def __init__(self):
        super(Bce_iou_loss, self).__init__()
    
    def forward(self, pred, mask):
        weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        
        pred = torch.sigmoid(pred)
        inter = pred * mask
        union = pred + mask
        iou = 1 - (inter + 1) / (union - inter + 1)

        weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
        weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        return (weighted_bce + weighted_iou).mean()        

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




