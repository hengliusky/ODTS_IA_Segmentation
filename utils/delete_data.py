import os
import re

if __name__ == '__main__':
    dir = r'/opt/data/private/Datasets/nnunet_base/nnUNet_raw/Dataset602_MAS_lowres_coarse_finetune/labelsTs'
    patient_list = os.listdir(dir)

    tr_keys = ['0000420571', '0000966323', '0000861405', '0000965795', '0000972912', '0000163718', '0000924122',
               '0000979454', '0000987587', '0000977646', '0000925737', '0000990926', '0000306571', '0000959870',
               '0000927108', '0000454967', '0000963538', '0000231737', '0000098621', '0000447651', '0000410329',
               '0000231397', '0000859304', '0000724089', '0000954259', '0000996543', '0000951772', '0001015136',
               '0000986040', '0000065253']

    for name in patient_list:
        ret = re.split('[_\.]', name)

        if ret[0] in tr_keys:

            path = os.path.join(dir, name)

            print(path)

            # os.remove(path)

            # break