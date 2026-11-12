import os
import re

import nibabel as nib
import numpy as np

from scipy import ndimage

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

    is_notcover = False

    # ============= 这是求掩码区域中心的算法 =================

    # 找到最大连通区域
    labeled_array, num_features = ndimage.label(pred)

    max_region_sum = 0
    max_region_idx = 0
    for idx in range(num_features):
        region_sum = np.sum(labeled_array == (idx+1))
        if region_sum > max_region_sum:
            max_region_sum = region_sum
            max_region_idx = idx

    # 计算每个连通区域的几何中心坐标
    centers = ndimage.center_of_mass(pred, labeled_array, range(1, num_features + 1))

    H, W, S = np.round(centers[max_region_idx], decimals=0)
    print(centers)
    print(np.unique(labeled_array))
    print(num_features)
    print(max_region_idx)
    print(H, W, S)

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

    print(np.sum(label))
    print(np.sum(sub_label))
    print(np.sum(sub_pred))

    if np.sum(sub_label) != np.sum(label):
        is_notcover = True

    return sub_label, sub_pred, sub_ct, is_notcover

# 从粗分割结果作为指标参考，结合原始分辨率的 CTA，得到 128×128×128 的聚焦框
if __name__ == '__main__':
    coarse_pred_dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset301_LIASD_lowres/labelsTr'
    patient_list = os.listdir(coarse_pred_dir)

    label_dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset301_LIASD_lowres/labelsTr'
    HR_cta_dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset301_LIASD_lowres/imagesTr'

    ct_out_dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset302_LIASD_fullres/imagesTr'
    label_out_dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset302_LIASD_fullres/labelsTr'

    n_nopred = 0
    label_nopred = 0

    n_notcover = 0

    for patient_name in patient_list:
        if '.nii.gz' in patient_name:
            pass
        else:
            continue
        # 获取预测、标签和CTA数据
        print(patient_name)
        pred_path = os.path.join(coarse_pred_dir, patient_name)
        pred_data = nib.load(pred_path).get_fdata()

        label_path = os.path.join(label_dir, patient_name)
        try:
            label_data = nib.load(label_path).get_fdata()
        except EOFError:
            print(f"Error: The file {label_path} appears to be corrupted. Skipping this file.")
            continue
        ret = re.split('\.', patient_name)
        ct_patient_name = ret[0] + '_0000.nii.gz'
        ct_path = os.path.join(HR_cta_dir, ct_patient_name)
        try:
            ct_data = nib.load(ct_path).get_fdata()
        except EOFError:
            print(f"Error: The file {ct_path} appears to be corrupted. Skipping this file.")
            continue

        print(pred_data.shape)

        if np.sum(pred_data) == 0:
            n_nopred += 1
            label_nopred += np.sum(label_data)
            print(np.sum(label_data))
            print('==================有一个没有做出预测的=====================')
            continue

        # 预测的太过分散导致散落在各个地方，然后求平均坐标的地方什么都没，考虑使用最大联通分量
        sub_label, sub_pred, sub_ct, is_notcover = get_fb(label_data, pred_data, ct_data, 128)

        if is_notcover == True:
            n_notcover += 1
            print('==================有一个没有覆盖完整的=====================')

        cta_nii = nib.Nifti1Image(sub_ct, affine=np.eye(4))
        label_nii = nib.Nifti1Image(sub_label, affine=np.eye(4))

        ret = re.split('\.', patient_name)
        cta_path = os.path.join(ct_out_dir , ret[0]+'_0000.nii.gz')
        label_path = os.path.join(label_out_dir, patient_name)

        print(cta_path)
        print(label_path)

        nib.save(cta_nii, cta_path)
        nib.save(label_nii, label_path)

        print('==================正常保存=====================')

    print(n_notcover)
    print(n_nopred)
    if n_nopred != 0:
        print(label_nopred/n_nopred)

