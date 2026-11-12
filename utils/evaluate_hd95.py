import os
import numpy as np
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
from hausdorff import hausdorff_distance
from medpy.metric import binary

def hd95_score(pred, label):
    hd95_total = 0
    conut = 0
    H, W, S = pred.shape
    for s in range(S):
        pred_single = pred[:, :, s]
        label_single = label[:, :, s]

        hd95_single = hausdorff_distance(pred_single, label_single)

        if hd95_single != 0.0:
            hd95_total = hd95_total + hd95_single
            conut = conut + 1
    hd95 = hd95_total / conut
    return hd95


def compute_hd95(label1, label2, voxel_spacing):
    """
    计算两个三维标签的HD95指标值

    参数：
        label1：第一个三维标签，numpy数组
        label2：第二个三维标签，numpy数组
        voxel_spacing：体素间距，三维元组（或列表）

    返回值：
        HD95：Hausdorff距离的95百分位数
    """
    # 计算标签1到标签2的Hausdorff距离
    hd1 = directed_hausdorff(label1, label2)[0]

    # 计算标签2到标签1的Hausdorff距离
    hd2 = directed_hausdorff(label2, label1)[0]

    # 计算HD95（Hausdorff距离的95百分位数）
    HD95 = np.percentile([hd1, hd2], 95)

    return HD95


if __name__ == '__main__':
    ref_dir = r'/opt/data/private/DataSets/nnunet_base/nnUNet_old_raw/Dataset501_WMC/labelsTs'
    pred_dir = r'/opt/data/private/Comparative_work/Prediction/nnUNet/fullres'

    patient_list = os.listdir(ref_dir)
    print(len(patient_list))
    count = len(patient_list)
    hd_total = 0

    for patient_name in patient_list:
        print(patient_name)
        ref_path = os.path.join(ref_dir, patient_name)
        pred_path = os.path.join(pred_dir, patient_name)

        ref_data = nib.load(ref_path).get_fdata()
        pred_data = nib.load(pred_path).get_fdata()

        if np.max(pred_data) == 0. :
            count = count - 1
            print('啥都没检测出来')
            print('------------------')
            continue

        print(ref_data.shape)
        print(pred_data.shape)
        print(np.max(pred_data))

        hd95 = binary.hd95(pred_data, ref_data, voxelspacing=None)

        print(hd95)
        hd_total += hd95
        print('--------------------')

    print(hd_total/count)

