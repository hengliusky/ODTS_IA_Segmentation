import os
import shutil

if __name__ == '__main__':
    dir = r'/opt/data/private/DataSets/WMC_mydiv/test/ct/'
    target_dir = r'/opt/data/private/DataSets/nnunet_base/nnUNet_raw/Dataset501_WMC/imagesTs/'

    file_list = os.listdir(dir)

    for file_name in file_list:
        print(file_name)

        new_filename = file_name.replace(".nii.gz", "_0000.nii.gz")

        print(new_filename)

        source_path = dir + file_name
        target_path = target_dir + new_filename

        print(source_path)
        print(target_path)

        shutil.copy2(source_path, target_path)

    print(len(file_list))