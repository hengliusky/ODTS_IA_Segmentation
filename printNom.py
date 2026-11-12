import os
import nibabel as nib
import numpy as np

def print_pixel_range(input_dir):
    """
    打印指定目录中每个 .nii.gz 文件的像素值范围。
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, filename)

            # 加载 .nii.gz 文件
            img = nib.load(file_path)
            data = img.get_fdata()

            # 计算最小值和最大值
            min_val = np.min(data)
            max_val = np.max(data)

            # 打印文件名和像素值范围
            print(f"File: {filename} - Min: {min_val}, Max: {max_val}")

# 指定输入目录
input_dir = r"E:\my_nnunet\data\nnUNet_raw\Dataset505_WMC_only_foucs\imagesTs_Nom"
print_pixel_range(input_dir)
