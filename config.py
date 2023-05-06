import torchvision.transforms as transforms


class spiral_config(object):
    """stores all variable of spiral-Net"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.DATA_IMAGE_LIST = './data/TP53_3D_T2_v3.txt'
        self.MODEL_DIR = "./result/rotate_unet_datanew3/"
        self.N = 32  # Number of cycles
        self.K = 128  # Number of samples per cycle
        self.R = 128  # The max radius of sampling region
        self.M = 512  # Number of samples along the radius
        self.Image_path = './data/data_pancreatic_cancer/rotate_seg/rotate_seg_newT2/image/'
        self.Label_path = './data/data_pancreatic_cancer/rotate_seg/rotate_seg_newT2/label/'
        self.Label_3D_path = './data/data_pancreatic_cancer/rotate_seg/T2label3D_npy/'
        self.normalize = transforms.Normalize([0.0994], [0.1174])
        # R128M512 [0.0994], [0.1174]


class two_d_config(object):
    """stores all variable of 2D-Net"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.DATA_IMAGE_LIST = './data/TP53_2D_T2_v3.txt'   # TP53_2D_T2_35 TP53_2D_T2_v3  TP53_2D_T2_10
        self.DATA_IMAGE_INFORMATION_LIST = './data/TP53_2D_T2_infor.txt'
        self.MODEL_DIR = "./rotate_unet_datanew3/SOTA/"
        # self.MODEL_DIR = './runs/'
        self.N = 256  # 图像大小
        self.M = 128
        # self.IMAGE_PATH = 'D:/data/data_Pancreatic_cancer/cancer/original_T2_crop/original_T2_crop2d256z/'
        self.IMAGE_PATH = './data_pancreatic_cancer/original_T2_crop/original_T2_crop2d256z/'


class three_d_config(object):
    """stores all variable of 3D-Net"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.DATA_IMAGE_LIST = './data/TP53_3D_T2_v3.txt'
        self.MODEL_DIR = "./rotate_unet_datanew3/"
        self.IMAGE_PATH = './data_pancreatic_cancer/original_T2_crop/original_T2_crop3d112_np/'
        self.rows, self.cols, self.c = 112, 112, 112


class three_dcube_config(object):
    """stores all variable of stacked2.5D-Net"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.DATA_IMAGE_LIST = './data/TP53_3D_T2_v3.txt'
        self.MODEL_DIR = "./rotate_unet_datanew3/ReUNet3dcubexyz70/ReUNetcube/"

