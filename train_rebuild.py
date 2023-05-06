# 将3D重构计算出来的loss加入了训练过程
# 该代码只适用于batchsize=1的情况
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
from read_data import DataSet_crop4_rebuild_aug
from k_fold import k_fold_pre
from losses import DICELoss_LV_2D_3D, IouLoss_2D_3D, IouLoss2D_DICELoss3D, IouLoss2D_DICELoss3D_reg, \
    IouLoss2D_DICELoss3D_regL2, IouLoss2D_DICELoss3D_regsmooth, IouLoss2D_DICELoss3D_regweightsmooth
from models.UNet import UNet
from models.ResUNet import ResUNet
from models.wide_resunet import WideResUNet
from models.fcn8s import FCN8s
from tools.rebuild_torch import rotate_trans
import math
# used for logging to TensorBoard
from tensorboardX import SummaryWriter
import ast
from utils import AverageMeter, DSC_computation
from config import spiral_config
spiral_Config = spiral_config()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# CUDA_VISIBLE_DEVICES=1 python train.py
parser = argparse.ArgumentParser(description='PyTorch UNet Training')
parser.add_argument('--model', default='ResUNet', type=str,
                    help='baseline of the model')
parser.add_argument('--layers', default=[2, 2, 2, 2, 2], nargs='+', type=int,
                    help='numbers of Resbolck layers')
parser.add_argument('--norm_layer', default='instancenorm', type=str,
                    help='types of norm layers(None:batch norm,layernorm,instancenorm)')
parser.add_argument('--resize', default=True, type=ast.literal_eval,
                    help='resize layer of input and output（type=ast.literal_eval）')
parser.add_argument('--widen_factor', default=8, type=int,  # 5-fold
                    help='number of widen_factor')
parser.add_argument('--loss', default='IOU_2D_DICE_3D_regweightsmooth', type=str,
                    help='types of loss(DICE_2D_3D, DICE_3D, IOU_2D_3D, IOU_3D,IOU_2D_DICE_3D, '
                         'IOU_2D_DICE_3D_reg, IOU_2D_DICE_3D_regsmooth, IOU_2D_DICE_3D_regweightsmooth)')
parser.add_argument('--a', default=10, type=float,
                    help='coefficient of loss1')
parser.add_argument('--b', default=1, type=float,
                    help='coefficient of loss2')
parser.add_argument('--c', default=0.01, type=float,
                    help='coefficient of loss_reg')
parser.add_argument('--fold', default=5, type=int,  # 5-fold
                    help='number of k-fold')
parser.add_argument('--fold_index', default=0, type=int,
                    help='index of k-fold(0-4)')
parser.add_argument('--n_epoch', default=1, type=int,
                    help='number of epoch to change')
parser.add_argument('--epochs', default=15, type=int,
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
parser.add_argument('--seed', default=1, type=int,  # 随机数种子
                    help='random seed(default: 1 > 0)')
parser.add_argument('--resume',
                    default='./runs//checkpoint/',
                    type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name',
                    default='try_ResUNet',
                    type=str,
                    help='name of experiment')  # args.batch_size, args.n_epoch, args.epochs, args.lr,seed))
parser.add_argument('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True,
                    help='whether to use_cuda(default: True)')
# DATA_IMAGE_LIST = './data/TP53_3D_T2_v3.txt'
DATA_IMAGE_LIST = spiral_Config.DATA_IMAGE_LIST
MODEL_DIR = spiral_Config.MODEL_DIR
# 2D 坐标索引准备 共4*3=12种情况
start = time.time()
INDEX_WEIGHTS = {}
for crop_num in range(2):
    for dim in range(3):
        INDEX_WEIGHTS[(crop_num, dim)] = rotate_trans(crop_num, dim=dim)
print(time.time()-start)


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    global use_cuda, args, writer
    args = parser.parse_args()  # #####很重要
    if args.tensorboard:
        # configure("runs/%s" % args.name)
        writer = SummaryWriter(MODEL_DIR + "%s" % args.name)
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
    # input_random = torch.rand(32, 3, 100, 100)
    # if args.tensorboard:
    #     writer.add_graph(model, (input_random, input_random, input_random), True)
    if os.path.exists(MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name):
        checkpoint = torch.load(MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        torch.save({'state_dict': model.state_dict()}, MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name)

    if use_cuda:
        model = model.cuda()
        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        # model = torch.nn.DataParallel(model).cuda()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    else:
        print('Please choose true optimizer.')
        return 0

    # 5-fold 数据准备
    train_names, val_names = k_fold_pre(filename=MODEL_DIR+"%s/data_fold.txt" % args.name, image_list_file=DATA_IMAGE_LIST,
                                        fold=args.fold)

    filelossdice_name = MODEL_DIR+"{}/train_fold{}_loss_dice.txt".format(args.name, args.fold_index)
    filelossdice = open(filelossdice_name, 'a')
    for k in range(args.fold_index, args.fold_index+1):  # args.fold
        best_prec = 0  # 第k个fold的准确率
        # 读取第k个fold的数据
        train_dataset = DataSet_crop4_rebuild_aug(image_list_file=DATA_IMAGE_LIST, fold=train_names[k], mode='train')  # mode='train'
        val_dataset = DataSet_crop4_rebuild_aug(image_list_file=DATA_IMAGE_LIST, fold=val_names[k])
        kwargs = {'num_workers': 0, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_torch(args.seed), **kwargs)  # drop_last=True,
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_torch(args.seed), **kwargs)

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
                print("=> use initial checkpoint")
                checkpoint = torch.load(MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name)
                model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load("runs/%s/checkpoint_init.pth.tar" % args.name)
            model.load_state_dict(checkpoint['state_dict'])

        # Defining Loss Function
        if args.loss == 'DICE_2D_3D':
            criterion = DICELoss_LV_2D_3D(a=args.a, b=args.b)
        elif args.loss == 'IOU_2D_3D':
            criterion = IouLoss_2D_3D(a=args.a, b=args.b)
        elif args.loss == 'IOU_2D_DICE_3D':
            criterion = IouLoss2D_DICELoss3D(a=args.a, b=args.b)
        elif args.loss == 'IOU_2D_DICE_3D_reg':
            criterion = IouLoss2D_DICELoss3D_reg(a=args.a, b=args.b, c=args.c)
        elif args.loss == 'IOU_2D_DICE_3D_regL2':
            criterion = IouLoss2D_DICELoss3D_regL2(a=args.a, b=args.b, c=args.c)
        elif args.loss == 'IOU_2D_DICE_3D_regsmooth':
            criterion = IouLoss2D_DICELoss3D_regsmooth(a=args.a, b=args.b, c=args.c)
        elif args.loss == 'IOU_2D_DICE_3D_regweightsmooth':
            criterion = IouLoss2D_DICELoss3D_regweightsmooth(a=args.a, b=args.b, c=args.c)
        else:
            print('please input right loss function')
            return 0
        epoch_is_best = 0

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train_losses, train_dices = train(train_loader, model, criterion, optimizer, epoch, k)
            # for name, layer in model.named_parameters():
            #     writer.add_histogram('fold' + str(k) + '/' + name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            #     writer.add_histogram('fold' + str(k) + '/' + name + '_data', layer.cpu().data.numpy(), epoch)
            # evaluate on validation set
            val_losses, val_dices = validate(val_loader, model, criterion, epoch, k)
            if args.tensorboard:
                # x = model.conv1.weight.data
                # x = vutils.make_grid(x, normalize=True, scale_each=True)
                # writer.add_image('data' + str(k) + '/weight0', x, epoch)  # Tensor
                writer.add_scalars('data' + str(k) + '/loss',
                                   {'train_loss': train_losses.avg, 'val_loss': val_losses.avg}, epoch)
                writer.add_scalars('data' + str(k) + '/Accuracy', {'train_dice': train_dices.avg, 'val_dice': val_dices.avg},
                                   epoch)
            # remember best prec@1 and save checkpoint
            is_best = val_dices.avg > best_prec
            if is_best == 1:
                epoch_is_best = epoch
                best_prec = max(val_dices.avg, best_prec)  # 这个fold的最高准确率
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec,
            }, is_best, epoch, k)

            dice_loss_write = str(train_losses.avg) + ' ' + str(train_dices.avg) + '\n'
            filelossdice.write(dice_loss_write)
        writer.close()
        print('fold_num: [{}]\t Best accuracy {} \t epoch {}'.format(k, best_prec, epoch_is_best))
    filelossdice.close()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # epoch = epoch - 150
    if epoch <= args.n_epoch:
        lr = args.lr
    else:
        lr = args.lr * (1 + np.cos((epoch - args.n_epoch) * math.pi / args.epochs)) / 2

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, fold):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    train_losses = AverageMeter()
    train_dices = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, (image, mask_2D, randomint, index, dim, midxyz, _) in enumerate(train_loader):
        if use_cuda:
            image, mask_2D = image.cuda(), mask_2D.cuda()
        image, mask_2D = image.float(), mask_2D.float()

        optimizer.zero_grad()

        output_2D = model(image)

        loss1, loss2, train_loss = criterion(output_2D, mask_2D, INDEX_WEIGHTS[(index.item() % 2, dim.item())], randomint)

        train_losses.update(train_loss.item(), image.size(0))
        losses1.update(loss1.item(), image.size(0))
        losses2.update(loss2.item(), image.size(0))
        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()
        dice_2D = DSC_computation(output_2D, mask_2D, image)
        train_dices.update(dice_2D, image.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx == 0 or (batch_idx + 1) % args.print_freq == 0 or batch_idx == len(train_loader) - 1:  # 按一定的打印频率输出
            print('Train_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'L2D {loss1.val:4f} ({loss1.avg:.4f})\t'
                  'L3D {loss2.val:.4f} ({loss2.avg:.4f})\t'
                  'dice2D {dice.val:.3f}({dice.avg:.3f})'
                  .format(epoch, batch_idx, len(train_loader), batch_time=batch_time,
                          loss=train_losses, loss1=losses1, loss2=losses2, dice=train_dices))
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/train_loss', train_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/train_loss2D', losses1.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/train_loss3D', losses2.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/train_dice', train_dices.avg, epoch)
    return train_losses, train_dices


def validate(val_loader, model, criterion, epoch, fold):  # 返回值为准确率
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    val_losses = AverageMeter()
    val_dices = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    # switch to evaluate mode  切换到评估模式
    model.eval()  # 很重要

    end = time.time()
    with torch.no_grad():
        for batch_idx, (image, mask_2D, _, index, dim, midxyz,_) in enumerate(val_loader):
            if use_cuda:
                image, mask_2D = image.cuda(), mask_2D.cuda()
            image, mask_2D = image.float(), mask_2D.float()
            # print(image.size())
            output_2D = model(image)

            loss1, loss2, val_loss = criterion(output_2D, mask_2D, INDEX_WEIGHTS[(index.item() % 2, dim.item())])
            val_losses.update(val_loss.item(), image.size(0))
            losses1.update(loss1.item(), image.size(0))
            losses2.update(loss2.item(), image.size(0))

            dice_2D = DSC_computation(output_2D, mask_2D, image)
            val_dices.update(dice_2D, image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx == 0 or (batch_idx + 1) % args.print_freq == 0 or batch_idx == len(val_loader) - 1:  # 按一定的打印频率输出
                print('val_Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'L2D {loss1.val:.3f} ({loss1.avg:.3f})\t'
                      'L3D {loss2.val:.3f} ({loss2.avg:.3f})\t'
                      'dice2D {dice.val:.3f}({dice.avg:.3f})'
                      .format(epoch, batch_idx, len(val_loader), batch_time=batch_time,
                              loss=val_losses, loss1=losses1, loss2=losses2, dice=val_dices))
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/val_loss', val_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_loss2D', losses1.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_loss3D', losses2.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_acc', val_dices.avg, epoch)
    return val_losses, val_dices


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

