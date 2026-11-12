import os

# 指定要重命名的文件目录
directory = "/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset401_LIASD_lowres/imagesTs_2/"

# 列出目录中的所有文件
files = os.listdir(directory)

# 过滤出 .nii.gz 文件
nii_files = [f for f in files if f.endswith('.nii.gz')]

# 按自然排序对文件进行排序
nii_files.sort()

# 重命名文件
for i, filename in enumerate(nii_files):
    # 生成新的文件名，使用四位数格式（Tr_XXXX.nii.gz）
    new_filename = f"Ts_{i+1:03d}_0000.nii.gz"

    # 获取旧文件的完整路径
    old_filepath = os.path.join(directory, filename)

    # 获取新文件的完整路径
    new_filepath = os.path.join(directory, new_filename)

    # 重命名文件
    os.rename(old_filepath, new_filepath)

    print(f"Renamed {filename} to {new_filename}")

print("Renaming completed.")
