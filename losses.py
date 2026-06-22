import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'BoundaryBCEDiceLoss', 'BoundaryFocalTverskyLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class BoundaryBCEDiceLoss(nn.Module):
    """BCE + Dice with an extra soft boundary term for tooth contour quality."""
    def __init__(self, bce_weight=0.5, dice_weight=1.0, boundary_weight=0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    @staticmethod
    def _soft_boundary(x):
        dilate = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        erode = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        return (dilate - erode).clamp(0, 1)

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        prob = torch.sigmoid(input)
        smooth = 1e-5
        num = target.size(0)
        p = prob.view(num, -1)
        t = target.view(num, -1)
        dice = 1 - ((2. * (p * t).sum(1) + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()
        pred_edge = self._soft_boundary(prob)
        target_edge = self._soft_boundary(target)
        pe = pred_edge.view(num, -1)
        te = target_edge.view(num, -1)
        edge_dice = 1 - ((2. * (pe * te).sum(1) + smooth) / (pe.sum(1) + te.sum(1) + smooth)).mean()
        return self.bce_weight * bce + self.dice_weight * dice + self.boundary_weight * edge_dice



class BoundaryFocalTverskyLoss(nn.Module):
    """BCE + Dice + Focal-Tversky + boundary Dice for foreground semantic tooth masks."""
    def __init__(self, bce_weight=0.25, dice_weight=0.75, tversky_weight=0.50,
                 boundary_weight=0.15, alpha=0.45, beta=0.55, gamma=0.75):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _soft_boundary(x):
        dilate = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        erode = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        return (dilate - erode).clamp(0, 1)

    def forward(self, input, target):
        prob = torch.sigmoid(input)
        num = target.size(0)
        smooth = 1e-5
        bce = F.binary_cross_entropy_with_logits(input, target)
        p = prob.view(num, -1)
        t = target.view(num, -1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        dice = 1 - ((2 * tp + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        focal_tversky = torch.pow(1 - tversky, self.gamma).mean()
        pred_edge = self._soft_boundary(prob)
        target_edge = self._soft_boundary(target)
        pe = pred_edge.view(num, -1)
        te = target_edge.view(num, -1)
        edge_dice = 1 - ((2 * (pe * te).sum(1) + smooth) / (pe.sum(1) + te.sum(1) + smooth)).mean()
        return (self.bce_weight * bce + self.dice_weight * dice +
                self.tversky_weight * focal_tversky + self.boundary_weight * edge_dice)
