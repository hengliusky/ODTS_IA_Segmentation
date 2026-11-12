import os
import nibabel as nib
import numpy as np

def normalize_to_255(data):
    """
    归一化图像数据到 0~255 范围
    """
    data_min = np.min(data)
    data_max = np.max(data)
    # 避免除零错误
    if data_max > data_min:
        norm_data = (data - data_min) / (data_max - data_min+1e-10) * 255.0
    else:
        norm_data = np.zeros(data.shape)
    return norm_data.astype(np.uint8)

def process_and_save(input_dir, output_dir):
    """
    对指定路径下的 .nii.gz 文件归一化并保存到输出路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 加载 .nii.gz 文件
            img = nib.load(input_path)
            data = img.get_fdata()

            # 归一化
            normalized_data = normalize_to_255(data)

            # 使用原始图像头信息保存归一化图像
            normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
            nib.save(normalized_img, output_path)
            print(f"Processed and saved: {output_path}")

# 指定输入和输出目录
input_dir = r"E:\my_nnunet\data\nnUNet_raw\Dataset505_WMC_only_foucs\imagesTs"
output_dir = r"E:\my_nnunet\data\nnUNet_raw\Dataset505_WMC_only_foucs\imagesTs_Nom"
process_and_save(input_dir, output_dir)
