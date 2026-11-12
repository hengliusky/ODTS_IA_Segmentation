import os
import nibabel as nib
import numpy as np


# 函数用于检查单个nii.gz文件的每个切片是否包含非零像素
def check_nonzero_slices(nifti_file):
    try:
        # 读取nii.gz文件
        img = nib.load(nifti_file)
        data = img.get_fdata()

        nonzero_slices = []

        # 遍历所有切片并检查是否包含非零像素
        for i in range(data.shape[2]):  # 假设切片在Z轴方向 (第三个维度)
            slice_data = data[:, :, i]
            if np.any(slice_data != 0):  # 检查该切片是否包含非零像素
                nonzero_slices.append(i)

        return nonzero_slices
    except Exception as e:
        print(f"Error reading {nifti_file}: {e}")
        return None


# 获取文件的三位数字排序的key
def get_sort_key(filename):
    # 提取文件名中 .nii.gz 之前的三位数字
    parts = filename.split('_')
    for part in parts:
        if part.isdigit() and len(part) == 3:
            return int(part)
    return 0  # 如果找不到三位数字，返回0作为默认排序值


# 遍历指定文件夹下的所有nii.gz文件，并将结果写入txt文件
def check_all_nifti_files(directory, nonzero_file, zero_file):
    nonzero_files = []
    zero_files = []

    # 遍历目录
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                print(f"Checking file: {file_path}")
                nonzero_slices = check_nonzero_slices(file_path)

                if nonzero_slices is None:
                    print(f"Could not read file {file_path}")
                elif nonzero_slices:
                    print(f"File {file} contains non-zero pixels in slices: {nonzero_slices}")
                    nonzero_files.append(file)  # 保存包含非零像素的文件
                else:
                    print(f"File {file} does not contain any non-zero pixels in any slices.")
                    zero_files.append(file)  # 保存不包含非零像素的文件

    # 按照文件名中的三位数字排序
    nonzero_files.sort(key=get_sort_key)
    zero_files.sort(key=get_sort_key)

    # 将结果写入txt文件
    with open(nonzero_file, 'w') as f:
        f.write("Files containing non-zero pixels:\n")
        for file in nonzero_files:
            f.write(f"{file}\n")

    with open(zero_file, 'w') as f:
        f.write("Files without non-zero pixels:\n")
        for file in zero_files:
            f.write(f"{file}\n")


# 执行检查
if __name__ == "__main__":
    # 指定要检查的文件夹路径
    input_dir = "/opt/data/private/Datasets/nnunet_base/nnUNet_pred/Dataset401_LIASD_lowres/pred"  # 替换为实际路径
    nonzero_file = os.path.join(input_dir, "nonzero_files.txt")
    zero_file = os.path.join(input_dir, "zero_files.txt")
    check_all_nifti_files(input_dir, nonzero_file, zero_file)
