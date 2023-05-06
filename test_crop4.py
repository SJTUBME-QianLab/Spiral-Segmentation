import argparse
import os
import random
import shutil
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
from read_data import DataSet_crop4_rebuild
# from rebuild_torch import rotate_trans
from k_fold import k_fold_pre
from models.UNet import UNet
from models.ResUNet import ResUNet
from models.wide_resunet import WideResUNet
from models.fcn8s import FCN8s
from losses import IouLoss2D_DICELoss3D
import cv2
from tools.rebuild_np import rebuild_3d
import nrrd
import ast
from utils import *
from config import spiral_config
spiral_Config = spiral_config()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# CUDA_VISIBLE_DEVICES=1 python train.py
parser = argparse.ArgumentParser(description='PyTorch UNet Training')
parser.add_argument('--model', default='ResUNet', type=str,
                    help='baseline of the model')
parser.add_argument('--layers', default=[2,2,2,2,2], nargs='+', type=int,
                    help='numbers of Resbolck layers')
parser.add_argument('--norm_layer', default='instancenorm', type=str,
                    help='types of norm layers(None:batch norm,layernorm,instancenorm)')
parser.add_argument('--resize', default=True, type=ast.literal_eval,
                    help='resize layer of input and output（type=ast.literal_eval）')
parser.add_argument('--widen_factor', default=8, type=int,  # 5-fold
                    help='number of widen_factor')
parser.add_argument('--thre', '--threshold', default=0.5, type=float,
                    help='threshold')
parser.add_argument('--fold', default=5, type=int,  # 5-fold
                    help='number of k-fold')
parser.add_argument('--fold_index', default=0, type=int,
                    help='index of k-fold(0-4)')
parser.add_argument('--n_epoch', default=1, type=int,
                    help='number of epoch to change')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,  # 0.01
                    help='initial learning rate')
parser.add_argument('--optimizer', default='SGD', type=str,  # SGD
                    help='optimizer (SGD)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  # 正则化参数
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--growth', default=32, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--seed', default=0, type=int,  # 随机数种子
                    help='random seed(default: 1)')
parser.add_argument('--resume',
                    default='./runs/UNetcrop4_Nor_rebuild_IOU2DDICE3D_instanceN_SGD_rotate_1_1_10_0.001_seed1_datav3/checkpoint/',
                    type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name',
                    default='UNetcrop4_Nor_rebuild_IOU2DDICE3D_instanceN_SGD_rotate_1_1_10_0.001_seed1_datav3',
                    type=str,
                    help='name of experiment')  # args.batch_size, args.epochs, args.lr,seed))
parser.add_argument('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True,
                    help='whether to use_cuda(default: True)')
DATA_IMAGE_LIST = spiral_Config.DATA_IMAGE_LIST
MODEL_DIR = spiral_Config.MODEL_DIR
# 2D 坐标索引准备 共3种情况
# start = time.time()
# INDEX_WEIGHTS = {}
# for crop_num in range(4):
#     for dim in range(3):
#         INDEX_WEIGHTS[(crop_num, dim)] = rotate_trans(crop_num, dim=dim)
# print(time.time()-start)
# for dim in range(3):
#     INDEX_WEIGHTS[dim] = rotate_trans_weight(dim=dim)


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    global use_cuda, args, writer
    args = parser.parse_args()  # #####很重要
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.seed >= 0:
        seed_torch(args.seed)  # 固定随机数种子
    # create model
    if args.model == 'UNet':
        model = UNet(norm_layer=args.norm_layer, resize=args.resize)
    elif args.model == 'ResUNet':
        model = ResUNet(layers=args.layers, norm_layer=args.norm_layer, resize=args.resize)
    elif args.model == 'WideResUNet':
        model = WideResUNet(layers=args.layers, norm_layer=args.norm_layer, widen_factor=args.widen_factor, resize=args.resize)
    else:
        print('Please choose right network.')
        return 0
    if os.path.exists(MODEL_DIR+"%s/checkpoint_init.pth.tar" % args.name):
        print('os.path.exists')
        checkpoint = torch.load(MODEL_DIR+"%s/checkpoint_init.pth.tar" % args.name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('else')
        torch.save({'state_dict': model.state_dict()}, MODEL_DIR+"%s/checkpoint_init.pth.tar" % args.name)

    if use_cuda:
        model = model.cuda()
        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        # model = torch.nn.DataParallel(model).cuda()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # 5-fold 数据准备
    train_names, val_names = k_fold_pre(filename=MODEL_DIR+"%s/data_fold.txt" % args.name, image_list_file=DATA_IMAGE_LIST,
                                        fold=args.fold)

    for k in range(args.fold_index, args.fold_index + 1):  # args.fold
        # 读取第k个fold的数据
        val_dataset = DataSet_crop4_rebuild(image_list_file=DATA_IMAGE_LIST, fold=val_names[k], fold_num=k, filepath=args, mode='val')
        kwargs = {'num_workers': 0, 'pin_memory': True}

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume + 'checkpoint' + str(k) + '.pth.tar'):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume + 'checkpoint' + str(k) + '.pth.tar')
                model_dict = checkpoint['state_dict']
                model.load_state_dict(model_dict)
                args.start_epoch = checkpoint['epoch']
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                return 0
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return 0

        # Defining Loss Function
        criterion = IouLoss2D_DICELoss3D()
        # evaluate on validation set
        val_dices = validate(val_loader, model, criterion, args.start_epoch-1, k)

        file = open(MODEL_DIR+"{}/fold_{}_image_names_{}.txt".format(args.name, str(k), 'val'), 'r').readlines()  # 'a' 新建  'w+' 追加
        file_dice = open(MODEL_DIR+"{}/fold_{}_rebuild_dice_{}_{}.txt".format(args.name, str(k), 'val', str(args.thre)), 'w')  # 'a' 新建  'w+' 追加
        directory = MODEL_DIR+"{}/rebuild_result/".format(args.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image_num = 0
        for name in file:
            image_num = image_num + 1
            if image_num % 2 == 1:
                image_name = name.split("\n")[0]
                test_result_path = MODEL_DIR+"{}/test_result".format(args.name)
                pre_final, pre_final_lowfilter, pre_final_fill, dice = rebuild_3d(image_name, test_result_path)
                nrrd.write(directory + '{}-pre_final.nrrd'.format(image_name), pre_final.astype(np.int32))
                nrrd.write(directory + '{}-pre_final_lowfilter.nrrd'.format(image_name), pre_final_lowfilter.astype(np.int32))
                nrrd.write(directory + '{}-pre_final_fill.nrrd'.format(image_name), pre_final_fill.astype(np.int32))
                file_dice.write(image_name + ' ' + str(dice[0]) + ' ' + str(dice[1]) + ' ' + str(dice[2]) + '\n')
        file_dice.close()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # epoch = epoch - 150
    if epoch <= args.n_epoch:
        # lr = args.lr * epoch / args.n_epoch
        lr = args.lr
    else:
        lr = args.lr

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(val_loader, model, criterion, epoch, fold):  # 返回值为准确率
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    val_dices = AverageMeter()
    # switch to evaluate mode  切换到评估模式
    model.eval()  # 很重要

    end = time.time()
    log_txt = open(MODEL_DIR+"{}/test_fold{}_loss_dice.txt".format(args.name, args.fold_index), 'w')
    directory = MODEL_DIR+"{}/test_result/".format(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    num_crop = 0
    output_image = np.zeros(shape=(spiral_Config.M * 2, 0))
    for batch_idx, (image, mask_2D, _, index, dim, midxyz, image_name) in enumerate(val_loader):
        with torch.no_grad():
            if use_cuda:
                image, mask_2D = image.cuda(), mask_2D.cuda()
            image, mask_2D = image.float(), mask_2D.float()
            # print(image.size())
            output = model(image)
            output_crop = output.squeeze(0).squeeze(0).cpu().detach().numpy()
            output_crop = np.where(output_crop > args.thre, 1, 0)
            output_image = np.concatenate((output_image, output_crop), axis=1)
            print(image_name)
            if num_crop < 1:  # 0
                num_crop = num_crop + 1
            else:  # 1
                num_crop = 0
                # output_image = np.where(output_image > 0.5, 1, 0)
                cv2.imwrite(MODEL_DIR+"{}/test_result/{}".format(args.name, image_name[0]), output_image * 255)
                output_image = np.zeros(shape=(spiral_Config.M * 2, 0))

            dice = DSC_computation(output, mask_2D, image)
            log_txt.write(image_name[0] + ' ' + str(dice) + '\n')

            val_dices.update(dice, image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if batch_idx == 0 or (batch_idx + 1) % args.print_freq == 0 or batch_idx == len(val_loader) - 1:  # 按一定的打印频率输出
            print('val_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'dice2D {dice.val:.3f}({dice.avg:.3f})'
                  .format(epoch, batch_idx, len(val_loader), batch_time=batch_time, dice=val_dices))
    log_txt.close()
    return val_dices


def save_checkpoint(state, is_best, epoch, fold):
    """Saves checkpoint to disk"""
    # filename = 'checkpoint' + str(fold) + '_' + str(epoch) + '.pth.tar'
    filename = 'checkpoint' + str(fold) + '.pth.tar'
    directory = MODEL_DIR+"%s/checkpoint/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, MODEL_DIR+'%s/checkpoint/' % (args.name) + 'model_best' + str(fold) + '.pth.tar')


if __name__ == '__main__':

    main()

