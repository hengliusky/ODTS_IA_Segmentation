import re
import torch
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

if __name__ == '__main__':

    dir = r'/opt/data/private/DataSets/WMC_mydiv/train'
    out_dir = r'/opt/data/private/DataSets/nnunet_base/nnUNet_raw/Dataset511_WMC_lowres_input'

    ct_dir = os.path.join(dir, 'ct')
    label_dir = os.path.join(dir, 'label')

    out_ct_dir = os.path.join(out_dir, 'imagesTr')
    out_label_dir = os.path.join(out_dir, 'labelsTr')

    patient_list = os.listdir(ct_dir)

    # 1.ct的名字后面要加上0000
    # 2.怎么完成插值
    # 3.插值之后对掩码怎么处理，在下采样过程中出现的失真
    for patient_name in patient_list:
        ct_path = os.path.join(ct_dir, patient_name)
        label_path = os.path.join(label_dir, patient_name)

        ct_data = nib.load(ct_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()

        print(patient_name)
        print(ct_data.shape)
        print(label_data.shape)
        print(type(ct_data))

        cta_nii = nib.Nifti1Image(ct_data, affine=np.eye(4))
        label_nii = nib.Nifti1Image(label_data, affine=np.eye(4))

        out_label_path = os.path.join(out_label_dir, patient_name)
        ret = re.split('\.', patient_name)
        ct_patient_name = ret[0] + '_0000.nii.gz'
        out_ct_path = os.path.join(out_ct_dir, ct_patient_name)


        print(out_ct_path)
        print(out_label_path)

        break

        nib.save(cta_nii, out_ct_path)
        nib.save(label_nii, out_label_path)