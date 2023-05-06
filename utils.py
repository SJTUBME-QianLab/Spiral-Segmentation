import numpy as np
import math
import cv2
import nrrd
MAXDISTANCE = 100000  # 默认的最大值


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def DSC_computation(pred, label, image):
    # 用于trainin 含batch
    pred = pred.squeeze(1).cpu().detach().numpy()  # (1, 1, 512, 2048) --> (1, 512, 2048)
    label = label.squeeze(1).cpu().detach().numpy()
    pred = pred > 0.5
    pred_sum = pred.sum(axis=(1, 2))
    label_sum = label.sum(axis=(1, 2))
    inter_sum = np.logical_and(pred, label).sum(axis=(1, 2))
    dice = 0
    for i in range(image.size()[0]):
        dice = dice + (2 * float(inter_sum[i]) + 1e-8) / (pred_sum[i] + label_sum[i] + 1e-8)

    return np.sum(dice) / image.size()[0]


def cal_DSC(pred, label):
    # pred = pred.squeeze(1).cpu().detach().numpy()  # (1, 1, 112, 112, 112) --> (1, 112, 112, 112)
    # label = label.squeeze(1).cpu().detach().numpy()
    pred = pred > 0.5
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.logical_and(pred, label).sum()
    dice = (2 * float(inter_sum) + 1e-8) / (pred_sum + label_sum + 1e-8)

    return dice


def cal_Jaccard(pred, label):
    # pred = pred.squeeze(1).cpu().detach().numpy()  # (1, 1, 112, 112, 112) --> (1, 112, 112, 112)
    # label = label.squeeze(1).cpu().detach().numpy()
    # pred = pred > 0.5
    inter = np.logical_and(pred, label).sum()
    union = np.logical_or(pred, label).sum()
    jaccard = (float(inter) + 1e-8) / (float(union) + 1e-8)
    return jaccard


def cal_RMSE(pred, label):
    pixel_space = 1  # 像素点之间的距离
    # pred = pred > 0.5
    pred = pred.transpose(2,0,1)
    label = label.transpose(2,0,1)
    start, end = find_layers(label)
    label = label[start:end + 1]
    pred = pred[start:end + 1]
    label = label.astype(np.uint8)
    pred = pred.astype(np.uint8)
    total_dis = 0
    num = 0
    for j in range(label.shape[0]):
        pos1 = np.nonzero(pred[j])  # 缺失的层数不计入计算
        pos2 = np.nonzero(label[j])  # 断层也不计算
        if len(pos1[0]) == 0 or len(pos2[0])==0:
            continue
        temp_dis = distance_cal_2d(pred[j], label[j], pixel_space)
        total_dis += temp_dis
        num += 1
        # print(num)
    dis = total_dis/num
    return dis


def cal_ASD_part(pred, label):

    start, end = find_layers(pred)
    label = label[start:end + 1]
    pred = pred[start:end + 1]
    height, wide, layer = pred.shape

    label = label.astype(np.uint8)
    pred = pred.astype(np.uint8)
    total_dis = 0
    num = 0
    surface_pred, surface_label = [], []

    # find surface
    for h in range(height):
        cont_pred, _ = cv2.findContours(pred[h], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 获取h层的轮廓
        cont_label, _ = cv2.findContours(label[h], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # list 包含多个轮廓
        for cp in range(len(cont_pred)):
            for cpp in range(len(cont_pred[cp])):
                surface_pred.append(np.append(cont_pred[cp][cpp], h))
        for cl in range(len(cont_label)):
            for cll in range(len(cont_label[cl])):
                surface_label.append(np.append(cont_label[cl][cll], h))
    # calculate distance
    for i in range(len(surface_pred)):  # 针对表面的每个点
        distance = MAXDISTANCE
        label_point = np.array(surface_label)   # (N, 3)去掉一个维度
        pred_point = np.zeros_like(label_point)  # （N，3）
        pred_point[:, 0] = surface_pred[i][0]  # 两个坐标
        pred_point[:, 1] = surface_pred[i][1]
        pred_point[:, 2] = surface_pred[i][2]
        result = pred_point - label_point  # x1 - x2 , y1 - y2, z1-z2
        result = result[:, 0] ** 2 + result[:, 1] ** 2 + result[:, 2] ** 2  # (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2
        temp_distance = math.sqrt(np.min(result))  # 获取最小距离
        if temp_distance < distance:
            distance = temp_distance
        total_dis += distance
        num += 1
    return total_dis, num


def cal_ASD(pred, label):
    pred = pred.transpose(2, 0, 1)
    label = label.transpose(2, 0, 1)
    dis1, num1 = cal_ASD_part(pred, label)
    dis2, num2 = cal_ASD_part(label, pred)
    return (dis1 + dis2)/(num1+num2)


def cal_recall(pred, label):
    return np.logical_and(pred, label).sum()/label.sum()


def cal_precision(pred, label):
    return np.logical_and(pred, label).sum()/pred.sum()


def find_layers(label):
    start_layer = 400
    end_layer = 0
    for layer in range(label.shape[0]):
        arr = np.nonzero(label[layer])
        if len(arr[0]) == 0:
            continue
        # print(layer)
        start_layer = layer if layer < start_layer else start_layer
        end_layer = layer if layer > end_layer else end_layer
    return start_layer, end_layer


def distance_cal_2d(pred, label, pixel_space):
    # label = label.astype(np.uint8)
    cont_pred, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 获取预测结果的轮廓
    cont_label, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # list 包含多个轮廓
    total_dis = 0
    num = 0
    for i in range(len(cont_pred)):  # 分割结果可能存在多个轮廓
        for j in range(cont_pred[i].shape[0]):  # 针对某个轮廓中的每个点
            distance = MAXDISTANCE
            for k in range(len(cont_label)):  # 标签中可能存在多个轮廓
                label_point = cont_label[k].reshape((cont_label[k].shape[0], cont_label[k].shape[2]))  # 去掉多余的一个维度
                pred_point = np.zeros_like(label_point)  # N行2列
                pred_point[:, 0] = cont_pred[i][j, 0, 0]  # 两个坐标
                pred_point[:, 1] = cont_pred[i][j, 0, 1]
                result = pred_point - label_point  # x1 - x2 , y1 - y2
                result = result[:, 0] ** 2 + result[:, 1] ** 2  # (x1 - x2)^2 + (y1 - y2)^2
                temp_distance = np.min(result)  # 获取最小距离
                if temp_distance < distance:
                    distance = temp_distance
            total_dis += distance
            num += 1
    # print("num:", num)
    if num == 0:
        return 0
    else:
        return math.sqrt(total_dis / num) * pixel_space


if __name__ == '__main__':
    pre_path='./runs/ResUNet/sftp_ResUNetcrop4_Nor_rebuild_IOU2DDICE3Dregweightsmooth_10_1_1_instanceN_SGD_rotate_1_1_10_0.001_seed1_datav3/rebuild_result/'
    label_path='D:/data/data_Pancreatic_cancer/cancer/original_T2/'
    name = '002305286-T2'
    pre_file = pre_path + name + '-pre_final_lowfilter.nrrd'
    label_file = label_path + name + '-s.nrrd'
    pre, _ = nrrd.read(pre_file)
    label, _ = nrrd.read(label_file)
    dice = cal_DSC(pre,label)
    print(dice)
    jac = cal_Jaccard(pre,label)
    print(jac)
    rmse = cal_RMSE(pre,label)
    print(rmse)
    asd = cal_ASD(pre,label)
    print(asd)
    recall = cal_recall(pre,label)
    print(recall)
    precision = cal_precision(pre, label)
    print(precision)
