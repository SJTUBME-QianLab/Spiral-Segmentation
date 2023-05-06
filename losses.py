# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from tools.rebuild_torch import rebuild_3d
from config import spiral_config
spiral_Config = spiral_config()

class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):
        # print("output.size, mask.size:", output.size(), mask.size())
        probs = torch.squeeze(output,1)  #  output.size, mask.size: torch.Size([4, 1, 192, 256]) torch.Size([4, 1, 192, 256])
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class DICELoss_LV(nn.Module):

    def __init__(self):
        super(DICELoss_LV, self).__init__()

    def forward(self, output, mask):
        intersection = output * mask
        intersection = torch.sum(intersection)

        den1 = output * output
        den1 = torch.sum(den1)

        den2 = mask * mask
        den2 = torch.sum(den2)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        dice_eso = dice

        loss = 1 - dice_eso
        return loss


class DICELoss_LV_2D_3D(nn.Module):

    def __init__(self, a=1, b=1):
        super(DICELoss_LV_2D_3D, self).__init__()
        self.a = a
        self.b = b

    def forward(self, output_2D, output_3D, mask_2D, mask_3D):
        eps = 1e-8
        intersection1 = output_2D * mask_2D
        intersection1 = torch.sum(intersection1)
        den11 = output_2D * output_2D
        den11 = torch.sum(den11)
        den21 = mask_2D * mask_2D
        den21 = torch.sum(den21)
        dice1 = 2 * ((intersection1 + eps) / (den11 + den21 + eps))
        loss1 = 1 - dice1

        intersection2 = output_3D * mask_3D
        intersection2 = torch.sum(intersection2)
        den12 = output_3D * output_3D
        den12 = torch.sum(den12)
        den22 = mask_3D * mask_3D
        den22 = torch.sum(den22)
        dice2 = 2 * ((intersection2 + eps) / (den12 + den22 + eps))
        loss2 = 1 - dice2
        loss = self.a*loss1 + self.b*loss2
        # loss = loss2
        return loss1, loss2, loss


class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, output, mask):
        loss = self.criterion(output, mask)

        return loss


class BCE_DICELoss(nn.Module):

    def __init__(self):
        super(BCE_DICELoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, output, mask):
        intersection = output * mask
        intersection = torch.sum(intersection)

        den1 = output * output
        den1 = torch.sum(den1)

        den2 = mask * mask
        den2 = torch.sum(den2)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        dice_eso = dice

        loss = (1 - dice_eso) + self.criterion(output, mask)
        
        return loss


class IouLoss(nn.Module):

    def __init__(self):
        super(IouLoss, self).__init__()

    def forward(self, output, mask):
        inter = output * mask
        intersection = torch.sum(inter)

        union = output + mask - inter
        union = torch.sum(union)

        eps = 1e-8
        iou = (intersection + eps) / (union + eps)

        loss = 1 - iou
        return loss


class IouLoss_2D_3D(nn.Module):

    def __init__(self,a=1,b=1):
        super(IouLoss_2D_3D, self).__init__()
        self.a = a
        self.b = b

    def forward(self, output_2D, mask_2D, mask_3D, index, midxyz, dim):
        output_3D = rebuild_3d(output_2D, mask_3D, index, midxyz, dim=dim)
        eps = 1e-8
        inter1 = output_2D * mask_2D
        intersection1 = torch.sum(inter1)
        union1 = output_2D + mask_2D - inter1
        union1 = torch.sum(union1)
        iou1 = (intersection1 + eps) / (union1 + eps)
        loss1 = 1 - iou1

        inter2 = output_3D * mask_3D
        intersection2 = torch.sum(inter2)
        union2 = output_3D + mask_3D - inter2
        union2 = torch.sum(union2)
        iou2 = (intersection2 + eps) / (union2 + eps)
        loss2 = 1 - iou2

        loss = self.a*loss1 + self.b*loss2

        return loss1, loss2, loss


class IouLoss2D_DICELoss3D(nn.Module):  # 仅适用于batchsize=1

    def __init__(self, a=1, b=1):
        super(IouLoss2D_DICELoss3D, self).__init__()
        self.a = a
        self.b = b

    def forward(self, output_2D, mask_2D, index_weight,randomint=None):
        eps = 1e-8
        inter1 = output_2D * mask_2D
        intersection1 = inter1.sum()
        union1 = output_2D + mask_2D - inter1
        union1 = union1.sum()
        iou1 = (intersection1 + eps) / (union1 + eps)
        loss1 = 1 - iou1
        # output_2D_weight, mask_2D_weight = rebuild_3d(output_2D, mask_2D, index_weight)

        if randomint is None or randomint.item() > 10:
            output_2D_weight, mask_2D_weight = rebuild_3d(output_2D, mask_2D, index_weight)
        else:
            if randomint.item() == 0:  # 水平翻转
                index_weight = index_weight.view(spiral_Config.M * 2,spiral_Config.M * 2)
                index_weight = torch.flip(index_weight, dims=(1,))
                index_weight = index_weight.view(-1)
            elif randomint.item() == 1: # 垂直翻转
                index_weight = index_weight.view(spiral_Config.M * 2, spiral_Config.M * 2)
                index_weight = torch.flip(index_weight, dims=(0,))
                index_weight = index_weight.view(-1)
            elif randomint.item() == 2:
                index_weight = index_weight.view(spiral_Config.M * 2, spiral_Config.M * 2)
                index_weight = torch.flip(index_weight, dims=(0, 1))
                index_weight = index_weight.view(-1)

            output_2D_weight, mask_2D_weight = rebuild_3d(output_2D, mask_2D, index_weight)

        intersection2 = output_2D_weight * mask_2D.view(-1)
        intersection2 = intersection2.sum()
        den12 = output_2D_weight.sum()   # 没有平方，直接相加了
        den22 = mask_2D_weight.sum()
        dice2 = 2 * ((intersection2 + eps) / (den12 + den22 + eps))
        loss2 = 1 - dice2

        loss = self.a*loss1 + self.b*loss2
        return loss1, loss2, loss


class IouLoss2D_DICELoss3D_reg(nn.Module):

    def __init__(self, a=1, b=1, c=0.1):
        super(IouLoss2D_DICELoss3D_reg, self).__init__()
        self.criterion = nn.BCELoss(reduction='sum')  # reduction='sum'
        self.a = a
        self.b = b
        self.c = c

    def forward(self, output_2D, mask_2D, index_weight):
        eps = 1e-8
        inter1 = output_2D * mask_2D
        intersection1 = inter1.sum()
        union1 = output_2D + mask_2D - inter1
        union1 = union1.sum()
        iou1 = (intersection1 + eps) / (union1 + eps)
        loss1 = 1 - iou1
        output_2D_weight, mask_2D_weight = rebuild_3d(output_2D, mask_2D, index_weight)

        intersection2 = output_2D_weight * mask_2D.view(-1)
        intersection2 = intersection2.sum()
        den12 = output_2D_weight.sum()  # 没有平方，直接相加了
        den22 = mask_2D_weight.sum()
        dice2 = 2 * ((intersection2 + eps) / (den12 + den22 + eps))
        loss2 = 1 - dice2

        # output_2D_reg中不为0的值 和 mask_2D做交叉熵
        output_2D_reg = output_2D * mask_2D
        loss_reg = self.criterion(output_2D_reg, mask_2D)/mask_2D.sum()

        loss = self.a*loss1 + self.b*loss2 + self.c*loss_reg
        return loss1, loss2, loss


class IouLoss2D_DICELoss3D_regL2(nn.Module):

    def __init__(self, a=1, b=1, c=0.1):
        super(IouLoss2D_DICELoss3D_regL2, self).__init__()
        self.criterion = nn.BCELoss(reduction='sum')  # reduction='sum'
        self.a = a
        self.b = b
        self.c = c

    def forward(self, output_2D, mask_2D, index_weight):
        eps = 1e-8
        inter1 = output_2D * mask_2D
        intersection1 = inter1.sum()
        union1 = output_2D + mask_2D - inter1
        union1 = union1.sum()
        iou1 = (intersection1 + eps) / (union1 + eps)
        loss1 = 1 - iou1
        output_2D_weight, mask_2D_weight = rebuild_3d(output_2D, mask_2D, index_weight)

        intersection2 = output_2D_weight * mask_2D.view(-1)
        intersection2 = intersection2.sum()
        den12 = output_2D_weight.sum()  # 没有平方，直接相加了
        den22 = mask_2D_weight.sum()
        dice2 = 2 * ((intersection2 + eps) / (den12 + den22 + eps))
        loss2 = 1 - dice2

        # output_2D_weight和mask_2D_weight 不为0的项尽可能接近
        output_2D_reg = output_2D_weight * mask_2D.view(-1)
        loss_reg = torch.norm((output_2D_reg - mask_2D_weight), p=2)

        loss = self.a*loss1 + self.b*loss2 + self.c*loss_reg
        return loss1, loss2, loss


class IouLoss2D_DICELoss3D_regsmooth(nn.Module):

    def __init__(self, a=1, b=1, c=0.1):
        super(IouLoss2D_DICELoss3D_regsmooth, self).__init__()
        self.criterion = nn.BCELoss(reduction='sum')  # reduction='sum'
        self.a = a
        self.b = b
        self.c = c

    def forward(self, output_2D, mask_2D, index_weight):
        eps = 1e-8
        inter1 = output_2D * mask_2D
        intersection1 = inter1.sum()
        union1 = output_2D + mask_2D - inter1
        union1 = union1.sum()
        iou1 = (intersection1 + eps) / (union1 + eps)
        loss1 = 1 - iou1
        output_2D_weight, mask_2D_weight = rebuild_3d(output_2D, mask_2D, index_weight)

        intersection2 = output_2D_weight * mask_2D.view(-1)
        intersection2 = intersection2.sum()
        den12 = output_2D_weight.sum()  # 没有平方，直接相加了
        den22 = mask_2D_weight.sum()
        dice2 = 2 * ((intersection2 + eps) / (den12 + den22 + eps))
        loss2 = 1 - dice2
        loss_reg = smooth_loss(output_2D, mask_2D)
        loss = self.a*loss1 + self.b*loss2 + self.c*loss_reg
        return loss1, loss2, loss


class IouLoss2D_DICELoss3D_regweightsmooth(nn.Module):

    def __init__(self, a=1, b=1, c=0.1):
        super(IouLoss2D_DICELoss3D_regweightsmooth, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, output_2D, mask_2D, index_weight, randomint=None):
        eps = 1e-8
        inter1 = output_2D * mask_2D
        intersection1 = inter1.sum()
        union1 = output_2D + mask_2D - inter1
        union1 = union1.sum()
        iou1 = (intersection1 + eps) / (union1 + eps)
        loss1 = 1 - iou1

        if randomint is None  or randomint.item() > 10:
            output_2D_weight, mask_2D_weight = rebuild_3d(output_2D, mask_2D, index_weight)
        else:
            if randomint.item() == 0:  # 水平翻转
                index_weight = index_weight.view(spiral_Config.M * 2,spiral_Config.M * 2)
                index_weight = torch.flip(index_weight, dims=(1,))
                index_weight = index_weight.view(-1)
            elif randomint.item() == 1: # 垂直翻转
                index_weight = index_weight.view(spiral_Config.M * 2, spiral_Config.M * 2)
                index_weight = torch.flip(index_weight, dims=(0,))
                index_weight = index_weight.view(-1)
            elif randomint.item() == 2:
                index_weight = index_weight.view(spiral_Config.M * 2, spiral_Config.M * 2)
                index_weight = torch.flip(index_weight, dims=(0, 1))
                index_weight = index_weight.view(-1)

            output_2D_weight, mask_2D_weight = rebuild_3d(output_2D, mask_2D, index_weight)

        intersection2 = output_2D_weight * mask_2D.view(-1)
        intersection2 = intersection2.sum()
        den12 = output_2D_weight.sum()  # 没有平方，直接相加了
        den22 = mask_2D_weight.sum()
        dice2 = 2 * ((intersection2 + eps) / (den12 + den22 + eps))
        loss2 = 1 - dice2

        output_2D_weight = output_2D_weight.reshape(spiral_Config.M * 2, spiral_Config.M * 2)
        mask_2D_weight = mask_2D_weight.reshape(spiral_Config.M * 2, spiral_Config.M * 2)
        loss_reg = smooth_weightloss(output_2D_weight, mask_2D_weight, mask_2D)
        loss = self.a*loss1 + self.b*loss2 + self.c*loss_reg
        # print(loss1,loss2,loss_reg)
        return loss1, loss2, loss


def smooth_weightloss(output_2D_weight, mask_2D_weight, mask_2D):
    h, w = mask_2D_weight.shape
    valid1 = torch.eq(mask_2D_weight[1:, :], mask_2D_weight[:h-1, :])  # 水平方向比较是否相等
    valid2 = torch.eq(mask_2D_weight[:, 1:], mask_2D_weight[:, :w-1])  # 竖直方向比较是否相等
    aa = torch.where(valid1, mask_2D[:, :, 1:, :] * torch.abs(output_2D_weight[1:, :] - output_2D_weight[:h-1, :]),
                     torch.zeros_like(mask_2D_weight[1:, :]).cuda())
    bb = torch.where(valid2, mask_2D[:, :, :, 1:] * torch.abs(output_2D_weight[:, 1:] - output_2D_weight[:, :w-1]),
                     torch.zeros_like(mask_2D_weight[:, 1:]).cuda())
    loss = (aa.mean() + bb.mean()) / 2
    # loss = aa.sum() + bb.sum()
    return loss


def smooth_loss(output_2D, mask_2D):
    b, c, h, w = mask_2D.shape
    valid1 = torch.eq(mask_2D[:, :, 1:, :], mask_2D[:, :, :h-1, :])  # 水平方向比较是否相等
    valid2 = torch.eq(mask_2D[:, :, :, 1:], mask_2D[:, :, :, :w-1])  # 竖直方向比较是否相等
    aa = torch.where(valid1, mask_2D[:, :, 1:, :] * torch.abs(output_2D[:, :, 1:, :] - output_2D[:, :, :h-1, :]),
                     torch.zeros_like(mask_2D[:, :, 1:, :]).cuda())
    bb = torch.where(valid2, mask_2D[:, :, :, 1:] * torch.abs(output_2D[:, :, :, 1:] - output_2D[:, :, :, :w-1]),
                     torch.zeros_like(mask_2D[:, :, :, 1:]).cuda())
    loss = (aa.mean() + bb.mean()) / 2
    return loss