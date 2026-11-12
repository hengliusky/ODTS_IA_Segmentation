import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import nibabel as nib
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def get_Datasets(dir, transforms):

    modal_list = ['ct', 'label']
    patient_list = os.listdir(os.path.join(dir, 'ct'))

    data_path = []
    for patient in patient_list:
        patient_path = []
        for modal in modal_list:
            path = dir + '//' + modal + '//' + patient
            patient_path.append(path)
        data_path.append(patient_path)

    subject_list = []
    for path in data_path:
        subject = tio.Subject(
            ct=tio.ScalarImage(path[0]),
            label=tio.LabelMap(path[1]),
        )
        subject_list.append(subject)

    subject_dataset = tio.SubjectsDataset(subject_list, transform=transforms)

    return subject_dataset

if __name__ == '__main__':
    train_transforms = tio.Compose([
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.RescaleIntensity((0.5, 99.5)),
        tio.CropOrPad((512, 512, 149), padding_mode=0),
        tio.RandomNoise(p=0.5),
        tio.RandomFlip(),
        tio.OneOf({
            tio.RandomAffine(): 0.8,
            tio.RandomElasticDeformation(): 0.2,
        }),
    ])

    test_transforms = tio.Compose([
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.RescaleIntensity((0.5, 99.5)),
        tio.CropOrPad((512, 512, 256)),
    ])

    patch_sampler = tio.data.WeightedSampler(patch_size=(64, 64, 64), probability_map='label')

    mydatasets = get_Datasets(r'/opt/data/private/DataSets/WMC_div/train', train_transforms)

    patches_queue = tio.Queue(
        mydatasets,
        max_length=2,      # 真正训练的时候记得改
        samples_per_volume=1,
        sampler=patch_sampler,
        num_workers=0,
        shuffle_patches=False,
        shuffle_subjects=False
    )

    datalaoder = DataLoader(patches_queue, batch_size=2, shuffle=False)


    for batch in datalaoder:

        ct = batch['ct']['data']
        label = batch['label']['data']


        print(label.shape)
        print(ct.shape)

        idx_slice = 32
        plt.imshow(ct[0, 0, :, :, idx_slice])
        plt.show()

        plt.imshow(label[0][0][:, :, idx_slice].cpu().numpy())
        plt.show()
        print('----------')