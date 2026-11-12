import os
import re

import nibabel as nib
import numpy as np

def match_pred_label(label, pred, box_size):

    indices = np.where(pred >= 1)

    center = []
    for index in indices:
        index = np.mean(index)
        index = np.round(index, decimals=0)
        center.append(index)


    H, W, S = center

    H = int(H - box_size / 2)
    W = int(W - box_size / 2)
    S = int(S - box_size / 2)

    if S < 0:
        S = 0

    sub_label = label[H:H + box_size, W:W + box_size, S:S + box_size]
    sub_pred = pred[H:H + box_size, W:W + box_size, S:S + box_size]

    print('---------变换后--------')
    print(sub_label.shape)
    print(sub_pred.shape)

    print(np.sum(sub_label))
    print(np.sum(sub_pred))

    return sub_label, sub_pred

def get_fb(label, pred, ct, box_size):

    indices = np.where(pred >= 1)

    center = []
    for index in indices:
        index = np.mean(index)
        index = np.round(index, decimals=0)
        center.append(index)


    H, W, S = center

    H = int(H - box_size / 2)
    W = int(W - box_size / 2)
    S = int(S - box_size / 2)

    if S < 0:
        S = 0

    sub_label = label[H:H + box_size, W:W + box_size, S:S + box_size]
    sub_pred = pred[H:H + box_size, W:W + box_size, S:S + box_size]
    sub_ct = ct[H:H + box_size, W:W + box_size, S:S + box_size]

    print('---------变换后--------')
    print(sub_label.shape)
    print(sub_pred.shape)

    print(np.sum(label))
    print(np.sum(sub_label))
    print(np.sum(sub_pred))

    return sub_label, sub_pred, sub_ct

# 从粗分割结果作为指标参考，结合原始分辨率的 CTA，得到 128×128×128 的聚焦框
if __name__ == '__main__':
    coarse_pred_dir = r'/opt/data/private/DataSets/nnunet_base/nnUNet_raw/Dataset508_WMC_focus_as_refine_train/test_pred/pred'
    patient_list = os.listdir(coarse_pred_dir)

    label_dir = r'/opt/data/private/DataSets/nnunet_base/nnUNet_old_raw/Dataset501_WMC/labelsTs'
    HR_cta_dir = r'/opt/data/private/DataSets/nnunet_base/nnUNet_old_raw/Dataset501_WMC/imagesTs'

    ct_out_dir = r'/opt/data/private/DataSets/nnunet_base/nnUNet_raw/Dataset508_WMC_focus_as_refine_train/imagesTs'
    label_out_dir = r'/opt/data/private/DataSets/nnunet_base/nnUNet_raw/Dataset508_WMC_focus_as_refine_train/labelsTs'

    for patient_name in patient_list:
        print(patient_name)
        pred_path = os.path.join(coarse_pred_dir, patient_name)
        pred_data = nib.load(pred_path).get_fdata()

        label_path = os.path.join(label_dir, patient_name)
        label_data = nib.load(label_path).get_fdata()

        ret = re.split('\.', patient_name)
        ct_patient_name = ret[0] + '_0000.nii.gz'
        ct_path = os.path.join(HR_cta_dir, ct_patient_name)
        ct_data = nib.load(ct_path).get_fdata()

        print(pred_data.shape)
        print(label_data.shape)
        print(ct_data.shape)

        # 预测的太过分散导致散落在各个地方，然后求平均坐标的地方什么都没
        sub_label, sub_pred, sub_ct = get_fb(label_data, pred_data, ct_data, 128)

        cta_nii = nib.Nifti1Image(sub_ct, affine=np.eye(4))
        label_nii = nib.Nifti1Image(sub_label, affine=np.eye(4))

        ret = re.split('\.', patient_name)
        cta_path = os.path.join(ct_out_dir , ret[0]+'_0000.nii.gz')
        label_path = os.path.join(label_out_dir, patient_name)

        print(cta_path)
        print(label_path)

        nib.save(cta_nii, cta_path)
        nib.save(label_nii, label_path)

        print('=======================================')
