"""
*Preliminary* pytorch implementation.
VoxelMorph testing
"""

# python imports
import os
import glob
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from model import cvpr2018_net, SpatialTransformer
import datageneratorsV1
import scipy.io as sio

from visual import addimage

# from medipy.metrics import dice
import metrics

import cv2
from voxelmorph import plt_flowV1


import SimpleITK as sitk

def test(gpu,
         ref_dir,
         mov_dir,
         model,
         init_model_file):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # Set up model
    # 读入fixed_img
    batch_size = 1

    f_img = sitk.ReadImage(ref_dir)  # 基于altas的配准 （ 指：选定一张图像作为fixed图像 ）。此处ref_dir为fixed.nii.gz
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed = np.repeat(input_fixed, batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))

    # set up
    mov_vol_names = glob.glob(os.path.join(mov_dir, '*gz'))
    nums = len(mov_vol_names)

    for k in range(0, nums):
        print(mov_vol_names[k])
        movs = sitk.ReadImage(mov_vol_names[k])
        movs = sitk.GetArrayFromImage(movs)[np.newaxis, np.newaxis, ...]
        input_moving = torch.from_numpy(movs).to(device).float()


        # Use this to warp segments
        # trf = SpatialTransformer(input_fixed.shape[2:], mode='nearest')
        # trf.to(device)
        warp, flow = model(input_moving, input_fixed)
        # flow_save = sitk.GetImageFromArray(flow.cpu().detach().numpy())
        # sitk.WriteImage(flow_save,'D:\peizhunsd\data\\flow_img\\' + str(k) + '.nii')


        # 位移向量场的可视化
        # addimage(input_fixed,input_moving,warp,k)   # 可视化结果
        dice_score = metrics.dice_score(warp,input_fixed)
        print('相似性度量dice:',dice_score)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--ref_dir",
                        type=str,
                        dest="ref_dir",
                        default='E:\peizhunsd\LPBA40\\fixed.nii.gz',
                        help="ref ")

    parser.add_argument("--mov_dir",
                        type=str,
                        dest="mov_dir",
                        default='E:\peizhunsd\LPBA40\\test\\',
                        help="ref ")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--init_model_file",
                        type=str,
                        dest="init_model_file",
                        default='E:\peizhunsd\\voxelmorph\LPBA\models2\\16000.ckpt',
                        help="model weight file")

    test(**vars(parser.parse_args()))

