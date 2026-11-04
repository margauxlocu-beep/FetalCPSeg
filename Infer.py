import os
import time

import nibabel as nib
import numpy as np

import torch

from Network import MixAttNet
from Utils import AvgMeter, check_dir
from DataOp import get_list

from glob import glob



# Utilisation du GPU si disponible, sinon du CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_path = os.path.join('output/')

test_list = [
    {"image_path": path}
    for path in sorted(glob("Data_test/case*/image.nii"))
    ]

# Chargement du réseau
net = MixAttNet().to(device)
net.load_state_dict(
    torch.load(
        output_path + '/ckpt/best_val.pth.gz',
        map_location=device,
        weights_only=True
    )
)

patch_size = (64, 64, 16)
spacing = 4

save_path = os.path.join(output_path, 'save_data')
check_dir(save_path)

net.eval()

test_meter = AvgMeter()

for idx, data_dict in enumerate(test_list):
    image_path = data_dict['image_path']

    #conserver l'affine de l'IRM
    image_nii_source = nib.load(image_path)
    image = image_nii_source.get_fdata()

    #Normalisation de l'IRM
    image = image.astype(np.float32)
    zone = image[image > 0]

    if zone.size > 0:
        mean = zone.mean()
        std = zone.std()
        image[image > 0] = (zone - mean) / (std + 1e-8)

    # Dimensions originales
    w, h, d = image.shape

    # Dimensions du patch
    patch_w, patch_h, patch_d = patch_size

    # ajouter du padding si nécessaire
    padding = [
        (0, max(0, patch_w - w)),
        (0, max(0, patch_h - h)),
        (0, max(0, patch_d - d))
    ]

    image_pad = np.pad(
        image,
        padding,
        mode='constant',
        constant_values=0
    )
    # Dimensions après padding
    W, H, D = image_pad.shape

    # initialiser 5 cartes de probabilités
    predict = np.zeros(
        (5, W, H, D),
        dtype=np.float32
    )

    pre_count = np.zeros(
        (W, H, D),
        dtype=np.float32
    )

    def positions(dimension, taille_patch):
        if dimension < taille_patch:
            raise ValueError("Patch plus grand que le volume")

        pas = max(1, taille_patch // spacing)

        return sorted(set(
            list(range(0, dimension - taille_patch + 1, pas))
            + [dimension - taille_patch]
        ))

    x_list = positions(W, patch_w)
    y_list = positions(H, patch_h)
    z_list = positions(D, patch_d)
    
    start_time = time.time()

    with torch.no_grad():
        for x in x_list:
            for y in y_list:
                for z in z_list:

                    # Extraction du patch 64x64x16
                    image_patch = image_pad[
                        x:x+patch_w,
                        y:y+patch_h,
                        z:z+patch_d
                    ].astype(np.float32)

                    # Conversion en tenseur PyTorch
                    patch_tensor = torch.from_numpy(
                        image_patch[np.newaxis, np.newaxis, ...]
                    ).to(device)

                    # MODIFICATION 4 : récupérer les 5 classes
                    outputs = net(patch_tensor)

                    # Si le réseau renvoie plusieurs sorties,
                    # on utilise la sortie principale
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]

                    if outputs.ndim != 5 or outputs.shape[1] != 5:
                        raise ValueError(
                            "Le réseau doit produire 5 canaux. "
                            f"Forme obtenue : {outputs.shape}"
                        )

                    # Transformer les logits en probabilités
                    proba_patch = torch.softmax(
                        outputs, dim=1
                    )[0].cpu().numpy()

                    #accumuler les 5 probabilités
                    predict[
                        :,
                        x:x+patch_w,
                        y:y+patch_h,
                        z:z+patch_d
                    ] += proba_patch

                    # Compter les prédictions par voxel
                    pre_count[
                        x:x+patch_w,
                        y:y+patch_h,
                        z:z+patch_d
                    ] += 1


    #moyenne des probabilités
    if np.any(pre_count == 0):
        raise RuntimeError("Certains voxels n'ont pas été couverts.")

    predict /= pre_count[np.newaxis, ...]

    # choisir la classe la plus probable
    predict = np.argmax(
        predict, axis=0
    ).astype(np.uint8)

    #retirer le padding
    predict = predict[:w, :h, :d]

    #conserver l'affine originale
    image_nii = nib.Nifti1Image(
        image,
        affine=image_nii_source.affine,
        header=image_nii_source.header.copy()
    )
    image_nii.set_data_dtype(np.float32)

    predict_nii = nib.Nifti1Image(
        predict,
        affine=image_nii_source.affine,
        header=image_nii_source.header.copy()
    )
    predict_nii.set_data_dtype(np.uint8)

    # Sauvegarde des fichiers
    case_path = os.path.join(save_path, str(idx))
    check_dir(case_path)

    nib.save(
        image_nii,
        os.path.join(case_path, 'image.nii.gz')
    )

    nib.save(
        predict_nii,
        os.path.join(case_path, 'predict.nii.gz')
    )

    print("Dimensions IRM :", image.shape)
    print("Dimensions segmentation :", predict.shape)
    print("Classes prédites :", np.unique(predict))

    print(
        "[{}] Testing Finished, Cost {:.2f}s".format(
            idx, time.time() - start_time
        )
    )
