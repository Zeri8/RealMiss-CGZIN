
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.softmax(1)
        ce = F.cross_entropy(pred, target)
        target_onehot = F.one_hot(target, 4).permute(0,4,1,2,3).float()
        intersection = (pred * target_onehot).sum(dim=(2,3,4))
        union = pred.sum(dim=(2,3,4)) + target_onehot.sum(dim=(2,3,4))
        dice = 1 - (2 * intersection + 1) / (union + 1)
        return ce + dice.mean()

class KLLoss(nn.Module):
    def forward(self, p, q):
        return F.kl_div(p, q, reduction='batchmean')