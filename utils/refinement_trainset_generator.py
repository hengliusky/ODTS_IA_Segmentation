import re
import torch
import numpy as np
import os
import nibabel as nib
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def trainset_generator(ct, label):
    box_size = 128
    indices = np.where(label >= 1)

    center = []
    for index in indices:
        index = np.mean(index)
        index = np.round(index, decimals=0)
        center.append(index)

    H, W, S = center
    H_shift = random.randint(-32, 32)
    W_shift = random.randint(-32, 32)
    S_shift = random.randint(-32, 32)

    H = int(H + H_shift - 64)
    W = int(W + W_shift - 64)
    S = int(S + S_shift - 64)

    if S < 0 :
        S = 0

    print(H, W, S)
    print(H_shift, W_shift, S_shift)

    sub_ct = ct[H:H + box_size, W:W + box_size, S:S + box_size]
    sub_label = label[H:H + box_size, W:W + box_size, S:S + box_size]

    print('---------变换后--------')
    print(sub_label.shape)

    print(np.sum(label))
    print(np.sum(sub_label))

    return sub_ct, sub_label


if __name__ == '__main__':

    # 粗分割网络raw数据集路径
    dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset604_MAS2_lowres_coarse_finetune'
    # 细分割网络输出路径
    out_dir = '/opt/data/private/Datasets/nnunet_base/nnUNet_raw/test'

    # 聚焦影像尺寸
    box_size = [128, 128, 128]
    # 细分割训练集数量缩放系数
    scale_factor = 2

    ct_dir = os.path.join(dir, 'imagesTr')
    label_dir = os.path.join(dir, 'labelsTr')

    out_ct_dir = os.path.join(out_dir, 'imagesTr')
    out_label_dir = os.path.join(out_dir, 'labelsTr')

    patient_list = os.listdir(ct_dir)

    tr_keys = ['0000056863', '0000535326', '0000710433', '0000732171', '0000208578','0000118857',  '0000862853', '0000925566', '0000561064', '0000539967', '0001016478', '0000099489', '0000742142', '0000111371', '0001017295', '0000218994', '0001019586', '0001017612', '0000706836', '0000872440', '0000000692', '0000098621', '0000181566', '0000231737', '0000410329', '0000501060', '0000510480', '0000611721', '0000712063', '0000951772', '0000956102', '0000916594', '0000900932', '0000638155', '0000986040', '0000963377', '0000899179', '0000858081', '0000697937', '0000447651', '0000996543', '0000971138', '0000954259', '0000667324', '0000231397', '0000924122', '0000736374', '0000591801', '0000391043', '0000126611']

    for patient_name in patient_list:

        ret = re.split('[_\.]', patient_name)

        print(patient_name)
        print(ret)

        if ret[0] in tr_keys:
            pass
        else:
            continue

        ct_path = os.path.join(ct_dir, patient_name)
        label_path = os.path.join(label_dir, ret[0] + '.nii.gz')

        ct_data = nib.load(ct_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()

        print(ct_data.shape)
        # ======= 随机位移以获取训练过程中的聚焦框 ===========
        for k in range(scale_factor):

            sub_ct_data, sub_label_data = trainset_generator(ct_data, label_data)

            sub_cta_nii = nib.Nifti1Image(sub_ct_data, affine=np.eye(4))
            sub_label_nii = nib.Nifti1Image(sub_label_data, affine=np.eye(4))

            ct_patient_name = ret[0]+'_' + str(k) + '_0000.nii.gz'
            out_ct_path = os.path.join(out_ct_dir, ct_patient_name)
            label_patient_name = ret[0]+'_' + str(k) + '.nii.gz'
            out_label_path = os.path.join(out_label_dir, label_patient_name)

            print(out_ct_path)
            print(out_label_path)

            nib.save(sub_cta_nii, out_ct_path)
            nib.save(sub_label_nii, out_label_path)

            print('=======================================')