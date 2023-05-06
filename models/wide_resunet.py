import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np


class WideResUNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=1, widen_factor=10, dropout_rate=0.3, layers=None, norm_layer=None, resize=False):
        super(WideResUNet, self).__init__()
        k = widen_factor
        num_feat = [16, 16*k, 32*k, 64*k, 128*k]
        if layers is None:
            layers = [1, 1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instancenorm':
            norm_layer = nn.InstanceNorm2d
        self.resize = resize
        if self.resize is True:
            self.down0 = nn.Sequential(nn.Conv2d(num_channels, num_feat[0], kernel_size=7, stride=2, padding=3),
                                       norm_layer(num_channels),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2))
            self.down1 = nn.Sequential(Conv3x3(num_feat[0], num_feat[0], norm_layer))
        else:
            self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0], norm_layer))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   self._wide_layer(wide_basic, num_feat[0], num_feat[1], layers[1], dropout_rate, stride=1))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   self._wide_layer(wide_basic, num_feat[1], num_feat[2], layers[2], dropout_rate, stride=1))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   self._wide_layer(wide_basic, num_feat[2], num_feat[3], layers[3], dropout_rate, stride=1))

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    self._wide_layer(wide_basic, num_feat[3], num_feat[4], layers[4], dropout_rate, stride=1))

        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3], norm_layer)
        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2], norm_layer)
        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1], norm_layer)
        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[0]*2, num_feat[0], norm_layer)
        if self.resize is True:
            # self.final = nn.Conv2d(num_feat[0], num_classes, kernel_size=1)
            self.final = nn.Sequential(nn.ConvTranspose2d(num_feat[0], num_classes, kernel_size=2, stride=2),
                                       nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2),
                                       nn.Sigmoid())  # 改为sigmoid
        else:
            self.final = nn.Sequential(nn.Conv2d(num_feat[0], num_classes, kernel_size=1),
                                       nn.Sigmoid())  # 改为sigmoid

    def _wide_layer(self, block, in_planes, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(in_planes, planes, dropout_rate, stride))
            in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, inputs, return_features=False):
        if self.resize is True:
            down0_feat = self.down0(inputs)   # torch.Size([1, 64, 256, 256])
            down1_feat = self.down1(down0_feat)  # torch.Size([64, 64, 256, 256])
        else:
            down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)  # torch.Size([1, 160, 256, 256])
        down3_feat = self.down3(down2_feat)  # torch.Size([1, 320, 128, 128])
        down4_feat = self.down4(down3_feat)  # torch.Size([1, 640, 64, 64])
        bottom_feat = self.bottom(down4_feat)  # torch.Size([1, 1280, 32, 32])

        up1_feat = self.up1(bottom_feat, down4_feat)  # torch.Size([1, 1280, 64, 64])
        up1_feat = self.upconv1(up1_feat)  # torch.Size([1, 640, 64, 64])
        up2_feat = self.up2(up1_feat, down3_feat)  # torch.Size([1, 640, 128, 128])
        up2_feat = self.upconv2(up2_feat)  # torch.Size([1, 320, 128, 128])
        up3_feat = self.up3(up2_feat, down2_feat)  # torch.Size([1, 320, 256, 256])
        up3_feat = self.upconv3(up3_feat)  # torch.Size([1, 160, 256, 256])
        up4_feat = self.up4(up3_feat, down1_feat)  # torch.Size([1, 32, 512, 512])
        up4_feat = self.upconv4(up4_feat)  # torch.Size([1, 16, 512, 512])

        if return_features:
            outputs = up4_feat
        else:
            outputs = self.final(up4_feat)  # torch.Size([1, 1, 512, 512])

            # print(outputs.size())
        # print("outputs: ", outputs)
        # print('resunet cbam is ', self.cbam)

        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

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


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    # net=Wide_ResNet(28, 10, 0.3, 10)
    # y = net(Variable(torch.randn(1,3,32,32)))
    net = ResUNet()
    print(net)
    # print(y.size())
