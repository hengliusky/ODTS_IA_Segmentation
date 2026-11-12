import os
import shutil


def rename_dicom_files(directory, insert_position=None):
    """
    重命名 DICOM 文件，在文件名的指定位置插入 _dce2。

    Args:
    - directory (str): DICOM 文件的目标目录。
    - insert_position (int, optional): 插入 _dce2 的位置。如果为 None，则添加到文件名的末尾。
    """
    # 遍历目标目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):  # 仅处理 .dcm 文件
                file_path = os.path.join(root, file)

                # 分离文件名和扩展名
                filename, ext = os.path.splitext(file)

                # 如果指定了插入位置
                if insert_position is not None and 0 <= insert_position <= len(filename):
                    # 在指定位置插入 _dce2
                    new_filename = f"{filename[:insert_position]}_dce2_1{filename[insert_position:]}"
                else:
                    # 默认在文件名的末尾添加 _dce2
                    new_filename = f"{filename}_dce2"

                # 构建新的文件路径
                new_file_path = os.path.join(root, new_filename + ext)

                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"Renamed {file_path} to {new_file_path}")


def remove_part_of_filename(directory, start_position, end_position):
    """
    删除文件名中特定位置的字符，并重命名文件。

    Args:
    - directory (str): 目标目录，包含需要重命名的文件。
    - start_position (int): 删除部分的开始位置（包含）。
    - end_position (int): 删除部分的结束位置（不包含）。
    """
    # 遍历目标目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):  # 仅处理 .dcm 文件
                file_path = os.path.join(root, file)

                # 分离文件名和扩展名
                filename, ext = os.path.splitext(file)

                # 如果指定的删除范围有效
                if 0 <= start_position < len(filename) and 0 < end_position <= len(
                        filename) and start_position < end_position:
                    # 删除指定位置的字符
                    new_filename = filename[:start_position] + filename[end_position:]
                else:
                    # 如果范围不合法，跳过
                    print(f"Skipping {file} due to invalid positions.")
                    continue

                # 构建新的文件路径
                new_file_path = os.path.join(root, new_filename + ext)

                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"Renamed {file_path} to {new_file_path}")


def move_part_of_filename(directory, start_position, end_position, insert_position):
    """
    将文件名中某个部分移动到另一个位置，并重命名文件。

    Args:
    - directory (str): 目标目录，包含需要重命名的文件。
    - start_position (int): 要移动的部分的开始位置（包含）。
    - end_position (int): 要移动的部分的结束位置（不包含）。
    - insert_position (int): 要将文件名部分移动到的位置。
    """
    # 遍历目标目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):  # 仅处理 .dcm 文件
                file_path = os.path.join(root, file)

                # 分离文件名和扩展名
                filename, ext = os.path.splitext(file)

                # 确保指定的范围和插入位置有效
                if (0 <= start_position < len(filename) and
                        0 < end_position <= len(filename) and
                        start_position < end_position and
                        0 <= insert_position <= len(filename)):

                    # 提取要移动的部分
                    part_to_move = filename[start_position:end_position]

                    # 将文件名分为移动部分前后的两部分
                    new_filename = filename[:start_position] + filename[end_position:]

                    # 插入提取的部分到指定位置
                    new_filename = new_filename[:insert_position] + part_to_move + new_filename[insert_position:]

                else:
                    print(f"Skipping {file} due to invalid positions.")
                    continue

                # 构建新的文件路径
                new_file_path = os.path.join(root, new_filename + ext)

                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"Renamed {file_path} to {new_file_path}")


def match_and_rename(roi_dir, label_dir):
    """
    将 ROI 目录中的文件与 label 目录中的文件进行匹配，并将匹配的 ROI 文件重命名为与标签文件一致。

    Args:
    - roi_dir (str): 包含切片（ROI 文件）的目录。
    - label_dir (str): 包含标签文件的目录。
    """
    # 遍历标签目录中的所有文件
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.dcm'):  # 只处理 DICOM 标签文件
            label_path = os.path.join(label_dir, label_file)

            # 获取文件名前 10 位作为匹配依据（病人ID）
            label_prefix = label_file[:10]

            # 在 ROI 目录中寻找匹配的文件
            for roi_file in os.listdir(roi_dir):
                if roi_file.startswith(label_prefix) and roi_file.endswith('.dcm'):
                    roi_path = os.path.join(roi_dir, roi_file)

                    # 构建新的 ROI 文件名，使其与标签文件名一致
                    new_roi_path = os.path.join(roi_dir, label_file)

                    # 重命名 ROI 文件
                    shutil.move(roi_path, new_roi_path)
                    print(f"Renamed {roi_path} to {new_roi_path}")
                    break  # 匹配成功后跳出循环，避免多次重命名


def compare_directories(dir1, dir2):
    # 从第一个目录获取文件名集合
    files_in_dir1 = set(os.listdir(dir1))
    # 从第二个目录获取文件名集合
    files_in_dir2 = set(os.listdir(dir2))

    # 找出只存在于第一个目录中的文件
    only_in_dir1 = files_in_dir1.difference(files_in_dir2)
    # 找出只存在于第二个目录中的文件
    only_in_dir2 = files_in_dir2.difference(files_in_dir1)

    # 打印结果
    print("Files only in", dir1, ":")
    for file in only_in_dir1:
        print(file)

    print("\nFiles only in", dir2, ":")
    for file in only_in_dir2:
        print(file)



# 函数用于在nii.gz文件名中添加字符
# 函数用于在指定位置插入字符串
def add_str_to_filenames(directory, add_str, insert_position):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii.gz"):
                # 提取文件名（去掉 .nii.gz 扩展名）
                base_name = file[:-7]

                # 检查插入位置是否超出文件名长度
                if insert_position > len(base_name):
                    print(
                        f"Insert position {insert_position} is greater than the file name length for {file}. Skipping.")
                    continue

                # 在指定位置插入字符串
                new_name = base_name[:insert_position] + add_str + base_name[insert_position:] + ".nii.gz"

                old_path = os.path.join(root, file)  # 原文件路径
                new_path = os.path.join(root, new_name)  # 新文件路径

                # 重命名文件
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {file} -> {new_name}")
                except Exception as e:
                    print(f"Error renaming {file}: {e}")


# 如果不传 insert_position，则 _dce2 添加到文件名末尾
# rename_dicom_files(directory)
if __name__ == '__main__':
    # 示例使用方式
    directory = '/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset401_LIASD_lowres/labelsTs_100_152'
    #
    # # 在文件名的第10个字符位置插入 "_dce2"（例如 "0000001545" 变成 "0000001545_dce2"）
    # rename_dicom_files(directory, insert_position=2)

    # 删除文件名中从第10个字符到第14个字符的位置（索引从0开始）
    # remove_part_of_filename(directory, start_position=2, end_position=3)

    # 将文件名中从第10个字符到第14个字符移动到第5个字符的位置
    # move_part_of_filename(directory, start_position=10, end_position=12, insert_position=16)
    # 示例使用
    # roi_directory = '/opt/data/private/sh_dataset/RegGan_CEMRI_diagnosis/train/ROI'
    # label_directory = '/opt/data/private/sh_dataset/RegGan_CEMRI_diagnosis/train/label'

    # 批量匹配和重命名文件
    # match_and_rename(roi_directory, label_directory)
    #
    # 指定两个目录路径
    # directory1 = '/opt/data/private/sh_dataset/RegGan_CEMRI_diagnosis/test/ROI'
    # directory2 = '/opt/data/private/sh_dataset/RegGan_CEMRI_diagnosis/test/label'

    # 调用函数比较这两个目录
    # compare_directories(directory1, directory2)
    # 执行文件重命名操作
    # 指定要处理的文件夹路径
    # 指定要添加的字符
    add_str = "1"  # 替换为你要添加的字符
    add_str_to_filenames(directory, add_str,3)
