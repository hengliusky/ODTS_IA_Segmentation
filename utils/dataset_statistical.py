import nibabel as nib
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

if __name__ == '__main__':
    dir = r'/opt/data/private/Datasets/MAS/NII-100'
    ct_dir = os.path.join(dir, 'ct_with_chest')
    label_dir = os.path.join(dir, 'label_with_chest')

    patient_list = os.listdir(ct_dir)

    sum_radius = 0
    n_patient = len(patient_list)

    for idx, patien_name in enumerate(patient_list):
        # 求动脉瘤的最大直径
        label_path = os.path.join(label_dir, patien_name)
        label_data = nib.load(label_path).get_fdata()

        # 找到掩码中所有非零元素的坐标
        nonzero_indices = np.transpose(np.nonzero(label_data))

        # 计算坐标之间的欧氏距离
        distances = pdist(nonzero_indices)

        # 将距离转换为方阵形式
        distance_matrix = squareform(distances)

        # 找到距离矩阵中的最大值，即为最大直径
        max_diameter = np.max(distance_matrix)

        print("最大直径：", max_diameter)

        sum_radius += max_diameter

    print("平均直径：", sum_radius/(n_patient))
