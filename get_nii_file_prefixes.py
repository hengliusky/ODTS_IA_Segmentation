import os


def get_file_prefixes(directory):
    # 检查指定目录是否存在
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []

    # 列出目录中的所有文件
    files = os.listdir(directory)

    # 过滤出 .nii.gz 文件
    nii_files = [f for f in files if f.endswith('.nii.gz')]

    # 获取文件名前缀（去除 '.nii.gz' 后缀）
    prefixes = [f[:-7] for f in nii_files]

    return prefixes


def write_prefixes_to_file(prefixes, output_file):
    # 构建字符串
    prefixes_string = "val_keys = [" + ", ".join([f"'{prefix}'" for prefix in prefixes]) + "]"

    # 将字符串写入文件
    with open(output_file, 'w') as f:
        f.write(prefixes_string)

def write_prefixes_to_file2(prefixes, output_file):
    # 将文件名前缀列表写入文件，每个前缀一行
    with open(output_file, 'w') as f:
        for prefix in prefixes:
            f.write(f"{prefix}\n")

def classify_nii_files(prefixes):
    # 初始化分类字典
    categories = {'_0_': [], '_1_': [], '_00': []}

    for filename in prefixes:
        # 确保文件名长度足够长
        if len(filename) >= 13:
            # 提取第 11 到第 13 位的字符（注意索引从 0 开始）
            key = filename[10:13]

            # 根据提取的子串分类
            if key in categories:
                categories[key].append(filename)
            else:
                print(f"文件 '{filename}' 的第 11-13 位字符为 '{key}'，不在指定分类中。")
        else:
            print(f"文件名 '{filename}' 长度不足 13 个字符，无法分类。")

    return categories

if __name__ == '__main__':

    # 指定目录路径
    directory = "/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset607_MAS2_refine/imagesTr"

    # 获取前六位文件名
    file_prefixes = get_file_prefixes(directory)

    # 输出文件路径
    output_file = "/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset607_MAS2_refine/imagesTr/file_prefixes.txt"

    # 检查是否有文件前缀
    if file_prefixes:
        # 对文件名前缀进行分类
        categories = classify_nii_files(file_prefixes)

        # 为每个分类创建输出文件并写入文件名前缀
        for key, prefixes in categories.items():
            # 定义输出文件路径，例如：/path/to/output/_0_.txt
            output_file = os.path.join(directory, f"{key}.txt")
            write_prefixes_to_file(prefixes, output_file)
            print(f"Prefixes for category '{key}' written to {output_file}")

    else:
        print("No .nii.gz files found in the specified directory.")
