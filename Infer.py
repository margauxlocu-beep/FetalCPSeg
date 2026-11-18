
import os
import time

import nibabel as nib
import numpy as np
import torch

from Network import MixAttNet
from tqdm import tqdm


# ============================================================
# 1. PARAMÈTRES
# ============================================================

# CPU ou GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Appareil utilisé :", device)

# Dossier de l'entraînement de 100 itérations
output_path = "output_100/"

# Meilleur modèle sauvegardé pendant l'entraînement
checkpoint_path = os.path.join(
    output_path, "ckpt", "best_val.pth.gz"
)

# IRM à segmenter
test_list = [
    {"image_path": "Data/case13/image.nii"},
    {"image_path": "Data/case19/image.nii"}
]

# Taille des patches
patch_size = (64, 64, 16)

# Dossier de sauvegarde des segmentations
save_path = os.path.join(output_path, "save_data")
os.makedirs(save_path, exist_ok=True)


# ============================================================
# 2. CHARGEMENT DU MODÈLE ENTRAÎNÉ
# ============================================================

net = MixAttNet().to(device)

net.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

net.eval()

print("Modèle chargé :", checkpoint_path)


# ============================================================
# 3. FONCTION POUR POSITIONNER LES PATCHES
# ============================================================

def positions(dimension, taille_patch):

    if dimension < taille_patch:
        raise ValueError("Patch plus grand que le volume")

    # Même méthode que dans Train.py pendant la validation
    pas = max(1, taille_patch)

    return sorted(set(
        list(range(
            0,
            dimension - taille_patch + 1,
            pas
        ))
        + [dimension - taille_patch]
    ))


# ============================================================
# 4. INFÉRENCE SUR CHAQUE IRM
# ============================================================

for idx, data_dict in enumerate(test_list):

    start_time = time.time()

    image_path = data_dict["image_path"]

    case_name = os.path.basename(
        os.path.dirname(image_path)
    )

    print("\n======================================")
    print("Segmentation de :", case_name)
    print("======================================")

    # --------------------------------------------------------
    # A. CHARGEMENT DE L'IRM
    # --------------------------------------------------------

    image_nii_source = nib.load(image_path)

    image = image_nii_source.get_fdata().astype(np.float32)

    # Normalisation de l'IRM
    zone = image[image > 0]

    if zone.size > 0:
        mean = zone.mean()
        std = zone.std()

        image[image > 0] = (
            zone - mean
        ) / (std + 1e-8)

    # Dimensions originales
    w, h, d = image.shape

    print("Dimensions originales :", image.shape)

    # --------------------------------------------------------
    # B. PADDING
    # --------------------------------------------------------

    patch_w, patch_h, patch_d = patch_size

    padding = [
        (0, max(0, patch_w - w)),
        (0, max(0, patch_h - h)),
        (0, max(0, patch_d - d))
    ]

    image_pad = np.pad(
        image,
        padding,
        mode="constant",
        constant_values=0
    )

    W, H, D = image_pad.shape

    # --------------------------------------------------------
    # C. INITIALISATION DES PRÉDICTIONS
    # --------------------------------------------------------

    # Une carte par classe : 0, 1, 2, 3, 4
    predict = np.zeros(
        (5, W, H, D),
        dtype=np.float32
    )

    # Nombre de prédictions par voxel
    pre_count = np.zeros(
        (W, H, D),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # D. POSITIONS DES PATCHES
    # --------------------------------------------------------

    x_list = positions(W, patch_w)
    y_list = positions(H, patch_h)
    z_list = positions(D, patch_d)

    total_patches = (
        len(x_list) * len(y_list) * len(z_list)
    )

    print("Nombre total de patches :", total_patches)

    # --------------------------------------------------------
    # E. PRÉDICTION AVEC MIXATTNET
    # --------------------------------------------------------

    with torch.no_grad():

        with tqdm(
            total=total_patches,
            desc="Inférence",
            unit="patch"
        ) as pbar:

            for x in x_list:
                for y in y_list:
                    for z in z_list:

                        # Extraire un patch
                        image_patch = image_pad[
                            x:x+patch_w,
                            y:y+patch_h,
                            z:z+patch_d
                        ].astype(np.float32)

                        # Convertir en tenseur PyTorch
                        patch_tensor = torch.from_numpy(
                            image_patch[
                                np.newaxis,
                                np.newaxis,
                                ...
                            ]
                        ).to(device)

                        # Prédiction
                        outputs = net(patch_tensor)

                        # Garder la sortie principale
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]

                        # Vérifier les 5 classes
                        if outputs.ndim != 5 or outputs.shape[1] != 5:
                            raise ValueError(
                                "Le réseau doit produire 5 canaux. "
                                f"Forme obtenue : {outputs.shape}"
                            )

                        # Récupérer les scores des 5 classes
                        # Même méthode que dans Train.py (validation)
                        proba_patch = outputs[0].cpu().numpy()

                        # Accumuler les scores
                        predict[
                            :,
                            x:x+patch_w,
                            y:y+patch_h,
                            z:z+patch_d
                        ] += proba_patch

                        # Compter les passages
                        pre_count[
                            x:x+patch_w,
                            y:y+patch_h,
                            z:z+patch_d
                        ] += 1

                        pbar.update(1)

    # --------------------------------------------------------
    # F. RECONSTRUCTION DE LA SEGMENTATION 3D
    # --------------------------------------------------------

    if np.any(pre_count == 0):
        raise RuntimeError(
            "Certains voxels n'ont pas été couverts."
        )

    # Moyenne des scores dans les zones de chevauchement
    predict /= pre_count[np.newaxis, ...]

    # Choisir la classe ayant le score le plus élevé
    predict = np.argmax(
        predict,
        axis=0
    ).astype(np.uint8)

    # Retirer le padding
    predict = predict[:w, :h, :d]

    # --------------------------------------------------------
    # G. SAUVEGARDE AU FORMAT NIFTI
    # --------------------------------------------------------

    # IRM normalisée
    image_header = image_nii_source.header.copy()
    image_header.set_data_dtype(np.float32)

    image_nii = nib.Nifti1Image(
        image,
        affine=image_nii_source.affine,
        header=image_header
    )

    # Segmentation prédite
    predict_header = image_nii_source.header.copy()
    predict_header.set_data_dtype(np.uint8)

    predict_nii = nib.Nifti1Image(
        predict,
        affine=image_nii_source.affine,
        header=predict_header
    )

    # Dossier propre à chaque cas
    case_path = os.path.join(
        save_path,
        case_name
    )

    os.makedirs(case_path, exist_ok=True)

    # Sauvegarder l'IRM
    nib.save(
        image_nii,
        os.path.join(case_path, "image.nii.gz")
    )

    # Sauvegarder la segmentation
    nib.save(
        predict_nii,
        os.path.join(case_path, "predict.nii.gz")
    )

    # --------------------------------------------------------
    # H. AFFICHAGE DES RÉSULTATS
    # --------------------------------------------------------

    print("\nRésultats pour :", case_name)

    print("Dimensions IRM :", image.shape)
    print("Dimensions segmentation :", predict.shape)

    print("Classes prédites :", np.unique(predict))

    print("\nRépartition des voxels :")

    for classe in range(5):

        nb_voxels = np.sum(predict == classe)

        pourcentage = (
            100 * nb_voxels / predict.size
        )

        print(
            f"Classe {classe} : "
            f"{nb_voxels} voxels "
            f"({pourcentage:.2f} %)"
        )

    print("\nFichiers sauvegardés dans :", case_path)

    print(
        "Temps d'inférence : {:.2f} secondes".format(
            time.time() - start_time
        )
    )

print("\n======================================")
print("INFÉRENCE TERMINÉE")
print("======================================")
