import os

for seed_ in [29]:
    for fold_index_ in range(0,2):
        path = 'ResUNet34crop2_re_Nor_rebuild_IOU2DDICE3Dregweightsmooth_10_1_0.01_instanceN_SGD_rotate_1_1_15_0.001_seed{}_datav3'.format(str(seed_))
        print(path, fold_index_)
        os.system("CUDA_VISIBLE_DEVICES=0 python train_rebuild.py "
                  "--model ResUNet "
                  "--norm_layer instancenorm "
                  "--resize True "
                  "--layers 2 3 4 6 3 "
                  "--loss IOU_2D_DICE_3D_regweightsmooth "
                  "--optimizer SGD "
                  "--a 10 "
                  "--b 1 "
                  "--c 0.01 "
                  "--lr 0.001 "
                  "--n_epoch 1 "
                  "--epochs 15 "
                  "--batch-size 1 "
                  "--seed {} "
                  "--fold_index {} "
                  "--resume ./result/rotate_unet_datanew3/{path}/checkpoint/ "
                  "--name {path}".format(seed_, fold_index_, path=path))

        os.system("CUDA_VISIBLE_DEVICES=0 python test_crop4.py "
                  "--model ResUNet "
                  "--layers 2 3 4 6 3 "
                  "--norm_layer instancenorm "
                  "--resize True "
                  "--thre 0.5 "
                  "--seed {} "
                  "--fold_index {} "
                  "--resume ./result/rotate_unet_datanew3/{path}/checkpoint/ "
                  "--name {path}".format(seed_, fold_index_, path=path))
