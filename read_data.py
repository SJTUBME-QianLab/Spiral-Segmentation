# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
import numpy as np
import nrrd
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import copy
import cv2
import time
import random
from config import spiral_config, two_d_config, three_d_config, three_dcube_config
spiral_Config = spiral_config()
two_d_Config = two_d_config()
three_d_Config = three_d_config()
three_cubed_Config = three_dcube_config()

Image_path = spiral_Config.Image_path
Label_path = spiral_Config.Label_path
Label_3D_path = spiral_Config.Label_3D_path
N = spiral_Config.N  # Number of cycles    32
K = spiral_Config.K  # Number of samples per cycle   128
R = spiral_Config.R  # The max radius of sampling region  60
M = spiral_Config.M  # Number of samples along the radius  256
# The image size is (2M,N*K/2)



def SetPara(axis='Z'):  # 跟据x,y,z方向的不同获取参数
    if axis == 'Z':
        Image_path = '/home/data/chenxiahan/data_pancreatic_cancer/original_T2_cube/original_T2_cube_Z/'
        Label_path = '/home/data/chenxiahan/data_pancreatic_cancer/original_T2_cube/original_T2_label_Z/'
        rows = 256  # 行
        cols = 256
    elif axis == 'X':
        Image_path = '/home/data/chenxiahan/data_pancreatic_cancer/original_T2_cube/original_T2_cube_X/'
        Label_path = '/home/data/chenxiahan/data_pancreatic_cancer/original_T2_cube/original_T2_label_X/'
        rows = 128  # 行
        cols = 256
    elif axis == 'Y':
        Image_path = '/home/data/chenxiahan/data_pancreatic_cancer/original_T2_cube/original_T2_cube_Y/'
        Label_path = '/home/data/chenxiahan/data_pancreatic_cancer/original_T2_cube/original_T2_label_Y/'
        rows = 128  # 行
        cols = 256
    else:
        raise Exception("input wrong datatype")
    return Image_path, Label_path, rows, cols


class DataSet_crop4_rebuild_aug(Dataset):
    def __init__(self, image_list_file, fold, fold_num=None, filepath=None, mode='val'):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            fold: five fold index list of dataset
            transform: optional transform to be applied on a sample.
            fold_num: the number of fold
            filepath: to save image_name.txt  (default: args)
        """
        self.crop_num = int((N * K / 2) / (2 * M))
        image_names = []
        mids = []
        self.datalen = len(fold)
        fileline = open(image_list_file, "r").readlines()
        for i in range(self.datalen):
            line = fileline[fold[i]]
            items = line.split()
            image_name = items[0]  # 没有后缀的文件名
            mid = []
            mid.append(float(items[1]))  # mid_x
            mid.append(float(items[2]))  # mid_y
            mid.append(float(items[3]))  # mid_z
            for j in range(self.crop_num):
                image_names.append(image_name)
                mids.append(mid)
            # print(i)
        # self.angle = {1: '_', 2: '_fh_', 3: '_fv_', 4: '_l15_', 5: '_l45_', 6: '_r15_',
        #               7: '_r45_', 8: '_l30_', 9: '_r30_', 10: '_l60_', 11: '_r60_', }
        self.angle = {1: '_', 2: '_r45_'}
        # self.angle = {1: '_', 2: '_l30_', 3: '_r30_', 4: '_l45_', 5: '_r45_'}
        self.image_names = image_names
        self.mids = mids
        normalize = spiral_Config.normalize  # transforms.Normalize([0.1018], [0.1182])  # R128[0.1018,0.1182]   R60[0.1696], [0.1267]
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])  # transforms.RandomChoice(transforms_list),
        self.mode = mode
        if fold_num is not None:
            filename = spiral_Config.MODEL_DIR + "{}/fold_{}_image_names_{}.txt".format(filepath.name, str(fold_num), mode)
            file = open(filename, 'w')  # 'a' 新建  'w+' 追加
            for w in range(len(image_names)):
                # print(image_names[w][0])
                out_write = image_names[w] + '\n'
                file.write(out_write)
            file.close()

    def __getitem__(self, index):  # 重载索引,对于实例的索引运算，会自动调用__getitem__
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        # print(index)
        midxyz = self.mids[index % len(self.image_names)]
        for g in range(1, len(self.angle) + 1):  # g:1,2,3
            flag = 0
            if index < len(self.image_names) * 3 * g:
                flag = 1
            if flag == 1:
                if (index - (g-1)*len(self.image_names) * 3) < len(self.image_names):  # origin
                    dim = 0
                    coordinate = 'xyz'
                    image_name = '{}-lab_two_d-{}-{}-{}-{}-2R{}{}.png'.format(self.image_names[index % len(self.image_names)], N, K, R, M, self.angle[g], coordinate)
                elif (index - (g-1)*len(self.image_names) * 3) < len(self.image_names) * 2:
                    dim = 1
                    coordinate = 'yzx'
                    image_name = '{}-lab_two_d-{}-{}-{}-{}-2R{}{}.png'.format(self.image_names[index % len(self.image_names)], N, K, R, M, self.angle[g], coordinate)
                else:
                    dim = 2
                    coordinate = 'zxy'
                    image_name = '{}-lab_two_d-{}-{}-{}-{}-2R{}{}.png'.format(self.image_names[index % len(self.image_names)], N, K, R, M, self.angle[g], coordinate)

                image = Image.open(Image_path + image_name).convert('L')
                label_2D = Image.open(Label_path + image_name).convert('L')
                mod = index % self.crop_num
                # label_3D_name = '{}-three_d_lab-{}-{}-{}-{}-2R_{}{}.npy'. \
                #     format(self.image_names[index % len(self.image_names)], N, K, R, M, coordinate, mod+1)
                # label_3D = np.load(Label_3D_path + label_3D_name)

                if self.mode == 'train':
                    # 生成0-3的随机数
                    randomint = random.randint(0,3)
                    if randomint == 0:  # Random horizontal flipping
                        image = TF.hflip(image)
                        label_2D = TF.hflip(label_2D)
                    elif randomint == 1:  # Random vertical flipping
                        image = TF.vflip(image)
                        label_2D = TF.vflip(label_2D)
                    elif randomint == 2:
                        image = TF.hflip(image)
                        image = TF.vflip(image)
                        label_2D = TF.hflip(label_2D)
                        label_2D = TF.vflip(label_2D)
                    # randomint == 3 不扩增

                    image_tensor = self.transform(image)[:, :, M * 2*mod:M * 2*(mod+1)]
                    label_2D_tensor = transforms.ToTensor()(label_2D)[:, :, M * 2*mod:M * 2*(mod+1)]
                    # label_3D_tensor = torch.Tensor(label_3D)
                    return image_tensor, label_2D_tensor, randomint, index, dim, midxyz, image_name
                else:
                    image_tensor = self.transform(image)[:, :, M * 2 * mod:M * 2 * (mod + 1)]
                    label_2D_tensor = transforms.ToTensor()(label_2D)[:, :, M * 2 * mod:M * 2 * (mod + 1)]
                    # label_3D_tensor = torch.Tensor(label_3D)
                    randomint = 1000
                    return image_tensor, label_2D_tensor, randomint, index, dim, midxyz, image_name

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_names) * 3 * len(self.angle)
        else:
            return len(self.image_names) * 3
