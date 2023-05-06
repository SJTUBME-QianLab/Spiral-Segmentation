import os.path as osp

import torch.nn as nn


class FCN8s(nn.Module):

    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

    def forward(self, x):
        h = x  # torch.Size([1, 1, 512, 512])
        h = self.relu1_1(self.conv1_1(h))  # torch.Size([1, 64, 512, 512])  [1, 64, 710, 710]
        h = self.relu1_2(self.conv1_2(h))  # torch.Size([1, 64, 512, 512])  [1, 64, 710, 710]
        h = self.pool1(h)  # torch.Size([1, 64, 256, 256])  1/2   [1, 64, 355, 355]

        h = self.relu2_1(self.conv2_1(h))  # torch.Size([1, 128, 256, 256])  [1, 128, 355, 355]
        h = self.relu2_2(self.conv2_2(h))  # torch.Size([1, 128, 256, 256])  [1, 128, 355, 355]
        h = self.pool2(h)  # torch.Size([1, 128, 128, 128]) 1/4              [1, 128, 178, 178]

        h = self.relu3_1(self.conv3_1(h))  # torch.Size([1, 256, 128, 128])  [1, 256, 178, 178]
        h = self.relu3_2(self.conv3_2(h))  # torch.Size([1, 256, 128, 128])  [1, 256, 178, 178]
        h = self.relu3_3(self.conv3_3(h))  # torch.Size([1, 256, 128, 128])  [1, 256, 178, 178]
        h = self.pool3(h)  # torch.Size([1, 256, 64, 64])                     [1, 256, 89, 89]
        pool3 = h  # 1/8  # torch.Size([1, 256, 64, 64])

        h = self.relu4_1(self.conv4_1(h))  # torch.Size([1, 512, 64, 64])    [1, 512, 89, 89]
        h = self.relu4_2(self.conv4_2(h))  # torch.Size([1, 512, 64, 64])    [1, 512, 89, 89]
        h = self.relu4_3(self.conv4_3(h))  # torch.Size([1, 512, 64, 64])    [1, 512, 89, 89]
        h = self.pool4(h)  # torch.Size([1, 512, 32, 32])     [1, 512, 45, 45]
        pool4 = h  # 1/16   # torch.Size([1, 512, 32, 32])

        h = self.relu5_1(self.conv5_1(h))  # torch.Size([1, 512, 32, 32])    [1, 512, 45, 45]
        h = self.relu5_2(self.conv5_2(h))  # torch.Size([1, 512, 32, 32])    [1, 512, 45, 45]
        h = self.relu5_3(self.conv5_3(h))  # torch.Size([1, 512, 32, 32])    [1, 512, 45, 45]
        h = self.pool5(h)  # torch.Size([1, 512, 16, 16])                    [1, 512, 23, 23]

        h = self.relu6(self.fc6(h))  # torch.Size([1, 4096, 1, 1])           [1, 4096, 17, 17]
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))  # torch.Size([1, 4096, 1, 1])           [1, 4096, 17, 17]
        h = self.drop7(h)

        h = self.score_fr(h)  # torch.Size([1, 1, 1, 1])                     [1, 1, 17, 17]
        h = self.upscore2(h)  # torch.Size([1, 1, 4, 4])                     [1, 1, 36, 36]
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)  # torch.Size([1, 1, 32, 32])            [1, 1, 45, 45]
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]  # torch.Size([1, 1, 4, 4])  [1, 1, 36, 36]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16  torch.Size([1, 1, 4, 4])        [1, 1, 36, 36]
        h = self.upscore_pool4(h)  # torch.Size([1, 1, 10, 10])              [1, 1, 74, 74]
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)  # [1, 1, 64, 64]                        [1, 1, 89, 89]
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8  ([1, 1, 10, 10])                            [1, 1, 74, 74]

        h = upscore_pool4 + score_pool3c  # 1/8  ([1, 1, 10, 10])            [1, 1, 74, 74]

        h = self.upscore8(h)  # ([1, 1, 88, 88])                             [1, 1, 600, 600]
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()  # torch.Size([1, 1, 57, 57])  [1, 1, 512, 512]
        h = nn.Sigmoid()(h)
        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)

