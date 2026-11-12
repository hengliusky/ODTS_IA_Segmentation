import re
import torch
import torch.nn.functional as F
import numpy as np
import os
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

# 目的是生成下采样的聚焦阶段的输入

def get_downsample(np_data):
    tensor_data = torch.Tensor(np_data).unsqueeze(dim=0).unsqueeze(dim=0)
    tensor_data = F.interpolate(tensor_data, size=(256, 256, 160), mode='nearest')
    np_data = tensor_data.squeeze(dim=0).squeeze(dim=0).numpy()

    return np_data


if __name__ == '__main__':
    dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset401_LIASD_lowres'
    out_dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset402_LIASD_refine/img'

    ct_dir = os.path.join(dir, 'imagesTr')
    label_dir = os.path.join(dir, 'labelsTr')

    out_ct_dir = os.path.join(out_dir, 'imagesTr')
    out_label_dir = os.path.join(out_dir, 'labelsTr')

    # 确保输出目录存在
    os.makedirs(out_ct_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    patient_list = os.listdir(ct_dir)

    for patient_name in patient_list:
        ct_path = os.path.join(ct_dir, patient_name)

        # 根据图像文件名推导出相应的标签文件名
        label_base_name = '_'.join(patient_name.split('_')[:2]) + '.nii.gz'
        label_path = os.path.join(label_dir, label_base_name)

        if not os.path.exists(label_path):
            print(f"Label file not found for {patient_name}, expected at {label_path}")
            continue

        try:
            ct_data = nib.load(ct_path).get_fdata()
        except EOFError:
            print(f"Error: The file {ct_path} appears to be corrupted. Skipping this file.")
            continue
        try:
            label_data = nib.load(label_path).get_fdata()
        except EOFError:
            print(f"Error: The file {label_path} appears to be corrupted. Skipping this file.")
            continue
        print(patient_name)
        print(ct_data.shape)
        print(label_data.shape)

        # ======= 下采样插值 ===========
        ct_data = get_downsample(ct_data)
        label_data = get_downsample(label_data)

        print(ct_data.shape)
        print(label_data.shape)
        print(np.unique(label_data))

        cta_nii = nib.Nifti1Image(ct_data, affine=np.eye(4))
        label_nii = nib.Nifti1Image(label_data, affine=np.eye(4))

        # 构建输出路径
        out_label_path = os.path.join(out_label_dir, label_base_name)
        out_ct_path = os.path.join(out_ct_dir, patient_name)

        print(out_ct_path)
        print(out_label_path)

        nib.save(cta_nii, out_ct_path)
        nib.save(label_nii, out_label_path)

        print('----------------------------')