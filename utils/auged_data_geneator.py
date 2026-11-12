import random
import sys
import torch
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split, dataset
from tqdm import tqdm
from pathlib import Path
import re
import cv2
from sklearn.model_selection import KFold
import numpy as np

import argparse

from Datasets.WMC_datastes import WMC_Datasets
from Datasets.WMC_datastes_torchio import get_Datasets
from model.modified_3D_UNet import Modified3DUNet
from utils.dice_score import dice_loss_3D

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from entry.predict import test_softmax, test_torchio
import os
import time
import nibabel as nib

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def crop_pred_label(label, pred, box_size):

    indices = torch.where(pred >= 1)

    center = []
    for index in indices:
        index = torch.mean(index.float())
        index = torch.round(index, decimals=0)
        center.append(index)


    _, _, H, W, S = center

    H = int(H - box_size / 2)
    W = int(W - box_size / 2)
    S = int(S - box_size / 2)

    if S < 0:
        S = 0


    sub_label = label[:, :, H:H + box_size, W:W + box_size, S:S + box_size]
    sub_pred = pred[:, :, H:H + box_size, W:W + box_size, S:S + box_size]

    return sub_label, sub_pred


def crop_tumor_area(cta, coarse_pred, label, box_size, threshold):
    coarse_pred[coarse_pred >= threshold] = 1
    coarse_pred[coarse_pred < threshold] = 0

    indices = torch.where(coarse_pred == 1)

    center = []
    for index in indices:
        index = torch.mean(index.float())
        index = torch.round(index, decimals=0)
        center.append(index)



    _, _, H, W, S = center

    H = int(H - box_size / 2)
    W = int(W - box_size / 2)
    S = int(S - box_size / 2)

    if S < 0:
        S = 0

    sub_cta = cta[:, :, H:H + box_size, W:W + box_size, S:S + box_size]
    sub_label = label[:, :, H:H + box_size, W:W + box_size, S:S + box_size]

    return sub_cta, sub_label


def cover_loss(pred, label, threshold, box_size):
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    if torch.sum(pred) == 0:
        v_sub_label = 0
    else:
        sub_label, _ = crop_pred_label(label, pred, box_size)
        v_sub_label = torch.sum(sub_label)

    v_label = torch.sum(label)

    return 1 - v_sub_label/v_label


if __name__ == '__main__':

    # 1.获得超参
    checkpoint_dir = r'/opt/data/private/Pycharm_project/intracranial aneurysm/log/ckp/'
    model_pth = r'/opt/data/private/Pycharm_project/intracranial aneurysm/log/ckp/epoch_55/best_model_55.pth'
    log_dir = r'/opt/data/private/tensorboard_logs/coarse_patch'
    val_freq = 5
    epochs = 1
    batch_size = 1
    lr = 1e-5
    continue_train = True
    in_channels = 1
    n_classes = 1
    scale_factor = 2
    threshold = 0.1
    patch_size = [192, 192, 64]
    max_length = 50
    samples_per_volume = 10


    # 4.创建数据加载器
    train_transforms = tio.Compose([
        tio.Clamp(out_min=-100, out_max=500),
        tio.RescaleIntensity((0, 1)),
        tio.ZNormalization(),
        tio.RandomNoise(p=0.5),
        tio.RandomFlip(),
        tio.OneOf({
            tio.RandomAffine(): 0.8,
            tio.RandomElasticDeformation(): 0.2,
        }),
    ])

    test_transforms = tio.Compose([
        tio.Clamp(out_min=-100, out_max=500),
        tio.RescaleIntensity((0, 1)),
        tio.ZNormalization(),
    ])

    train_dataset = get_Datasets(r'/opt/data/private/DataSets/WMC_div/train', train_transforms)

    test_dataset = get_Datasets(r'/opt/data/private/DataSets/WMC_div/test', test_transforms)

    patch_sampler = tio.data.LabelSampler(patch_size=patch_size, label_probabilities={0:0.66, 1:0.34})

    patches_queue = tio.Queue(
        train_dataset,
        max_length=max_length,      # 真正训练的时候记得改
        samples_per_volume=samples_per_volume,
        sampler=patch_sampler,
        num_workers=0,
    )

    patches_loader = DataLoader(patches_queue, batch_size=batch_size)

    idx = 0
    for epoch in range(0, epochs+1):

        with tqdm(total=len(train_dataset)*samples_per_volume, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in patches_loader:

                cta = batch['ct']['data'][0].numpy()
                label = batch['label']['data'][0].numpy()

                cta_nii = nib.Nifti1Image(cta, affine=np.eye(4))
                label_nii = nib.Nifti1Image(label, affine=np.eye(4))

                cta_path = "/opt/data/private/DataSets/auged_WMC/patched/train/ct/" + str(idx) + ".nii.gz"
                label_path = "/opt/data/private/DataSets/auged_WMC/patched/train/label/" + str(idx) + ".nii.gz"

                nib.save(cta_nii, cta_path)
                nib.save(label_nii, label_path)

                idx += 1

                pbar.update(cta.shape[0])



