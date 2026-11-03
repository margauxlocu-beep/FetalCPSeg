import os
import time

import nibabel as nib
import numpy as np

import torch

from Network import MixAttNet
from Utils import AvgMeter, check_dir
from DataOp import get_list

from glob import glob

torch.cuda.set_device(0)

output_path = os.path.join('output/')

test_list = [
    {"image_path": path}
    for path in sorted(glob("Data_test/case*/image.nii.gz"))
    ]

net = MixAttNet().cuda()
net.load_state_dict(torch.load(output_path+'/ckpt/best_val.pth.gz'))

patch_size = (64, 64, 16)
spacing = 4

save_path = os.path.join(output_path, 'save_data')
check_dir(save_path)

net.eval()

test_meter = AvgMeter()

for idx, data_dict in enumerate(test_list):
    image_path = data_dict['image_path']

    image = nib.load(image_path).get_fdata()

    #Normalisation de l'IRM
    image = image.astype(np.float32)
    zone = image[image > 0]

    if zone.size > 0:
        mean = zone.mean()
        std = zone.std()
        image[image > 0] = (zone - mean) / (std + 1e-8)

    w, h, d = image.shape

    pre_count = np.zeros_like(image, dtype=np.float32)
    predict = np.zeros_like(image, dtype=np.float32)

    # Dimensions du patch
    patch_w, patch_h, patch_d = patch_size

    def positions(dimension, taille_patch):
        if dimension < taille_patch:
            raise ValueError("Patch plus grand que le volume")

        pas = max(1, taille_patch // spacing)

        return sorted(set(
            list(range(0, dimension - taille_patch + 1, pas))
            + [dimension - taille_patch]
        ))

    x_list = positions(w, patch_w)
    y_list = positions(h, patch_h)
    z_list = positions(d, patch_d)
    
    start_time = time.time()

    with torch.no_grad():
        for x in x_list:
            for y in y_list:
                for z in z_list:

                    # Extraction du patch 64x64x16
                    image_patch = image[
                        x:x+patch_w,
                        y:y+patch_h,
                        z:z+patch_d
                    ].astype(np.float32)

                    # Conversion en tenseur PyTorch
                    patch_tensor = torch.from_numpy(
                        image_patch[np.newaxis, np.newaxis, ...]
                    ).cuda()

                    # Prédiction du réseau
                    pre_patch = net(patch_tensor)[0, 0].cpu().numpy()

                    # Accumuler les prédictions
                    predict[
                        x:x+patch_w,
                        y:y+patch_h,
                        z:z+patch_d
                    ] += pre_patch

                    # Compter les prédictions
                    pre_count[
                        x:x+patch_w,
                        y:y+patch_h,
                        z:z+patch_d
                    ] += 1

    predict /= pre_count

    predict = np.squeeze(predict)
    image = np.squeeze(image)

    predict[predict > 0.5] = 1
    predict[predict < 0.5] = 0

    image_nii = nib.Nifti1Image(image, affine=None)
    predict_nii = nib.Nifti1Image(predict, affine=None)

    check_dir(os.path.join(save_path, '{}'.format(idx)))
    nib.save(image_nii, os.path.join(save_path, '{}/image.nii.gz'.format(idx)))
    nib.save(predict_nii, os.path.join(save_path, '{}/predict.nii.gz'.format(idx)))

    print("[{}] Testing Finished, Cost {:.2f}s".format(idx, time.time()-start_time))
