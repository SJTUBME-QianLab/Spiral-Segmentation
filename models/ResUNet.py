# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torchvision.models as models
# res = models.resnet34()  # [3, 4, 6, 3]


class ResUNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=1, layers=None, norm_layer=None, resize=False):
        super(ResUNet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]
        if layers is None:
            layers = [1, 1, 1, 1, 1]  # 第一个值没用的
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instancenorm':
            norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.resize = resize
        self.dilation = 1
        if self.resize is True:
            self.down0 = nn.Sequential(nn.Conv2d(num_channels, num_feat[0], kernel_size=7, stride=2, padding=3),
                                       norm_layer(num_channels),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2))
            self.down1 = nn.Sequential(Conv3x3(num_feat[0], num_feat[0], norm_layer))
        else:
            self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0], norm_layer))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   self._make_layer(BasicBlock, num_feat[0], num_feat[1], layers[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   self._make_layer(BasicBlock, num_feat[1], num_feat[2], layers[2]))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   self._make_layer(BasicBlock, num_feat[2], num_feat[3], layers[3]))

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    self._make_layer(BasicBlock, num_feat[3], num_feat[4], layers[4]))

        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3], norm_layer)
        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2], norm_layer)
        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1], norm_layer)
        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0], norm_layer)
        if self.resize is True:
            # self.final = nn.Conv2d(num_feat[0], num_classes, kernel_size=1)
            self.final = nn.Sequential(nn.ConvTranspose2d(num_feat[0], num_classes, kernel_size=2, stride=2),
                                       nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2),
                                       nn.Sigmoid())  # 改为sigmoid
        else:
            self.final = nn.Sequential(nn.Conv2d(num_feat[0], num_classes, kernel_size=1),
                                       nn.Sigmoid())  # 改为sigmoid

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer, previous_dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, stride, norm_layer=norm_layer, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, inputs, return_features=False):
        if self.resize is True:
            down0_feat = self.down0(inputs)   # torch.Size([1, 64, 256, 256])
            down1_feat = self.down1(down0_feat)  # torch.Size([64, 64, 256, 256])
        else:
            down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)  # torch.Size([1, 128, 128, 128])
        down3_feat = self.down3(down2_feat)  # torch.Size([1, 256, 64, 64])
        down4_feat = self.down4(down3_feat)  # torch.Size([1, 512, 32, 32])
        bottom_feat = self.bottom(down4_feat)  # torch.Size([1, 1024, 16, 16])

        up1_feat = self.up1(bottom_feat, down4_feat)  # torch.Size([1, 1024, 32, 32])
        up1_feat = self.upconv1(up1_feat)  # torch.Size([1, 512, 32, 32])
        up2_feat = self.up2(up1_feat, down3_feat)  # torch.Size([1, 512, 64, 64])
        up2_feat = self.upconv2(up2_feat)  # torch.Size([1, 256, 64, 64])
        up3_feat = self.up3(up2_feat, down2_feat)  # torch.Size([1, 256, 128, 128])
        up3_feat = self.upconv3(up3_feat)  # torch.Size([1, 128, 128, 128])
        up4_feat = self.up4(up3_feat, down1_feat)  # torch.Size([1, 128, 256, 256])
        up4_feat = self.upconv4(up4_feat)  # torch.Size([1, 64, 256, 256])

        if return_features:
            outputs = up4_feat
        else:
            outputs = self.final(up4_feat)  # torch.Size([1, 1, 256, 256])

        # if self.resize is True:
        #     outputs = self.upsample(outputs)  # torch.Size([1, 1, 1024, 1024])

        return outputs


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, dilation=1):
        super(BasicBlock, self).__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print('basicblock forward cbam is ', self.cbam)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat, norm_layer=None):
        super(Conv3x3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   norm_layer(out_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   norm_layer(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3Drop(nn.Module):
    def __init__(self, in_feat, out_feat, norm_layer):
        super(Conv3x3Drop, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout(p=0.2),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   norm_layer(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3Small(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3Small, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ELU(),
                                   nn.Dropout(p=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ELU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = UpsampleDeterministic()
        # self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
        #                                  kernel_size=3,
        #                                  stride=1,
        #                                  dilation=1)

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.deconv(inputs)

        out = torch.cat([down_outputs, outputs], 1)
        # out = down_outputs + outputs
        return out


class UpConcat2x4(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat2x4, self).__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
        #                                  kernel_size=3,
        #                                  stride=1,
        #                                  dilation=1)

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=(2, 4),
                                         output_padding=(0, 2))

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(inputs)
        # outputs = self.deconv(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out


def upsample_deterministic(x, upscale):
    '''
    x: 4-dim tensor. shape is (batch,channel,h,w)
    output: 4-dim tensor. shape is (batch,channel,self. upscale*h,self. upscale*w)
    '''
    return x[:, :, :, None, :, None].expand(-1, -1, -1, upscale, -1, upscale).reshape(x.size(0), x.size(1),
                                                                                      x.size(2) * upscale,
                                                                                      x.size(3) * upscale)


class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        '''
        Upsampling in pytorch is not deterministic (at least for the version of 1.0.1)
        see https://github.com/pytorch/pytorch/issues/12207
        '''
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        '''
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,upscale*h,upscale*w)
        '''
        return upsample_deterministic(x, self.upscale)

