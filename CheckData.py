
import numpy as np
import nibabel as nib

from DataOp import get_list, TrainGenerator


# ==========================================================
# 1. PARAMÈTRES
# ==========================================================

DATA_PATH = "Data/"
PATCH_SIZE = (64, 64, 16)
BATCH_SIZE = 1
N_PATCHES = 60

# ==========================================================
# 2. RÉCUPÉRATION DES DONNÉES
# ==========================================================

train_list, val_list, _ = get_list(DATA_PATH)

print("\n========================================")
print("ANALYSE DES MASQUES COMPLETS")
print("========================================")

# ==========================================================
# 3. VÉRIFICATION DES IMAGES ET DES MASQUES
# ==========================================================

for data in train_list + val_list:

    image_path = data["image_path"]
    label_path = data["label_path"]

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image = np.asarray(image_nii.dataobj)
    label = np.asarray(label_nii.dataobj)

    case_name = image_path

    print(f"\nIRM : {case_name}")

    # Vérifier que les dimensions correspondent
    if image.shape != label.shape:
        raise ValueError(
            f"Dimensions différentes : "
            f"IRM {image.shape}, masque {label.shape}"
        )

    # Vérifier l'alignement spatial
    if not np.allclose(image_nii.affine, label_nii.affine):
        raise ValueError(
            f"Affines différentes pour {case_name}"
        )

    # Vérifier les classes
    classes, counts = np.unique(
        label,
        return_counts=True
    )

    if not np.all(np.isin(classes, [0, 1, 2, 3, 4])):
        raise ValueError(
            f"Classes inattendues : {classes}"
        )

    print("Dimensions :", image.shape)
    print("Classes présentes :", classes)

    for c, n in zip(classes, counts):

        percentage = 100 * n / label.size

        print(
            f"Classe {int(c)} : "
            f"{n} voxels "
            f"({percentage:.3f} %)"
        )


# ==========================================================
# 4. ANALYSE DES PATCHES D'ENTRAÎNEMENT
# ==========================================================

print("\n========================================")
print("ANALYSE DES PATCHES")
print("========================================")

generator = TrainGenerator(
    train_list,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE
)

# Nombre de patches contenant chaque classe
presence = np.zeros(5, dtype=int)

# Nombre total de voxels de chaque classe
total_voxels = np.zeros(5, dtype=np.int64)

for i in range(N_PATCHES):

    # Utilise exactement la fonction de DataOp.py
    image_patch, label_patch = generator.get_item()

    # Le masque a normalement la forme :
    # (batch, 1, 64, 64, 16)
    label = label_patch[0, 0]

    classes, counts = np.unique(
        label,
        return_counts=True
    )

    for c, n in zip(classes, counts):

        c = int(c)

        presence[c] += 1
        total_voxels[c] += n

    if (i + 1) % 10 == 0:
        print(
            f"{i + 1}/{N_PATCHES} patches analysés"
        )


# ==========================================================
# 5. AFFICHAGE DES RÉSULTATS
# ==========================================================

print("\n========================================")
print("RÉSULTATS SUR LES PATCHES")
print("========================================")

print(f"Nombre de patches : {N_PATCHES}")
print(f"Taille des patches : {PATCH_SIZE}")

for c in range(5):

    percentage = (
        100 * total_voxels[c] / total_voxels.sum()
    )

    print(
        f"\nClasse {c} :"
        f"\n  Présente dans {presence[c]}/{N_PATCHES} patches"
        f"\n  Proportion totale : {percentage:.3f} %"
    )


print("\n========================================")
print("ANALYSE TERMINÉE")
print("========================================")
