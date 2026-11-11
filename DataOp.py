import os
import glob

import nibabel as nib
import numpy as np
import random

import torch
from torch.utils.data import Dataset

from volumentations import *


def get_list(dir_path): #lit et range les fichiers
    """
    This function is to read data from data dir.
    The data dir should be set as follow:
    -- Data
        -- case1
            -- image.nii.gz
            -- label.nii.gz
        -- case2
        ...
    """
    print("Reading Data...")
    dict_list = [] #remplie avec les chemins des images et de leurs masques.
    #réccupère dossiers du chemin dir_path
    path_list = glob.glob(os.path.join(dir_path, '*')) #dir_path : représente le chemin vers le dossier contenant les cas. 

    path_list.sort()

    #Défini les noms des fichiers
    image_name = 'image.nii'
    label_name = 'label.nii'

    #associe images & masques. Création dictionnaire contenant deux associations clé-valeur. On enregistre les chemins aux images
    for path in path_list:
        dict_list.append(
            {
                'image_path': os.path.join(path, image_name),
                'label_path': os.path.join(path, label_name),
            }
        )

    # we split the data set to train set(0.75), val set(0.05), test set(0.2)
    #train_ratio = 0.75 75% des cas pour l'entraînement
    #val_ratio = 0.8 Attention : 80% == limite set test + set validation donc 5% de val et donc bien 20% de test 
    
    # 75% des patients pour l'entraînement
    # 25% des patients pour la validation
    # Les patients de test sont dans Data_test/
    train_ratio = 0.75 #
    val_ratio = 1.0 # 

    train_num = round(len(dict_list)*train_ratio)
    val_num = round(len(dict_list)*val_ratio)
    #slicing du data set
    train_list = dict_list[:train_num]
    val_list = dict_list[train_num:val_num]
    test_list = dict_list[val_num:]
    print("Finished! Train:{} Val:{} Test:{}".format(len(train_list), len(val_list), len(test_list)))

    return train_list, val_list, test_list


def get_augmentation():
    """
    here is the data augmentation compose function by packages volumentations:
    https://github.com/ashawkey/volumentations
    It provide a various augmentation strategy in 3D data
    """
    #retirer les # qui sont les mouvements pour la data augmentation si on veut l'utiliser / si besoin pour nb de cas
    return Compose([
        # Flip(0),
        # Flip(1),
        # Flip(2),
        # RandomRotate90((0, 1)),
        # RandomRotate90((0, 2)),
        # RandomRotate90((1, 2))
    ], p=0.5)


class TrainGenerator(object):
    """
    This is the class to generate the patches
    """
    def __init__(self, data_list, batch_size, patch_size):
        self.data_list = data_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        #self.aug = get_augmentation()

    def get_item(self): #pour obtenir un nouveau batch (de patient) d'entraînement

        dict_list = random.sample(self.data_list, self.batch_size) #choisit batch_size éléments dans la liste
        #extraction des fichiers (réccupère les chemins des fichiers sélectionnés)
        image_list = [dict_item['image_path'] for dict_item in dict_list]
        label_list = [dict_item['label_path'] for dict_item in dict_list]

        image_patch, label_patch = self._sample_patch(image_list, label_list) #génère les patches

        return image_patch, label_patch # retourne deux tableaux

    def _sample_patch(self, image_list, clean_list): #extractions des patches 3D
        # Dimensions du patch : largeur, hauteur, profondeur
        patch_w, patch_h, patch_d = self.patch_size

        # Moitié de chaque dimension
        half_w = patch_w // 2
        half_h = patch_h // 2
        half_d = patch_d // 2

        image_patch_list = []
        label_patch_list = []
    

        for image_path, clean_path in zip(image_list, clean_list): #parcours lites et extrait les chemins
            #1. Chargement de l'IRM et du masque
            image = nib.load(image_path).get_fdata() #Récupère les valeurs du volume sous forme de tableau NumPy
            label = nib.load(clean_path).get_fdata() #Les deux volumes doivent avoir des dimensions compatibles et être spatialement alignés (label et image)

            #2. Normalisation de l'IRM
            image = image.astype(np.float32)
            zone = image[image > 0]
            if zone.size > 0:
                mean = zone.mean()
                std = zone.std()
                # Normalisation des voxels non nuls
                image[image > 0] = (zone - mean) / (std + 1e-8)

            # here we augment the corresponding data and label
            #data = {'image': image, 'label': label} #dico avec image et masque de segmentation associé
            #aug_data = self.aug(**data) #application des mêmes transformations que image et label
            #image, label = aug_data['image'], aug_data['label']

            w, h, d = image.shape

            # Choisir un centre de patch en équilibrant les classes

            p = random.random()

            # Classes anatomiques présentes dans le masque
            classes_presentes = [
                c for c in range(1, 5)
                if np.any(label == c)
            ]

            if p < 0.875 and classes_presentes:

                # Choisir une classe anatomique au hasard
                classe = random.choice(classes_presentes)

                # Repérer les voxels de cette classe
                label_index = np.where(label == classe)

                # Choisir un voxel de cette classe comme centre
                sample_id = random.randrange(len(label_index[0]))

                x = int(label_index[0][sample_id])
                y = int(label_index[1][sample_id])
                z = int(label_index[2][sample_id])

            else:
                # Choisir aléatoirement un point dans le volume
                x = random.randrange(w)
                y = random.randrange(h)
                z = random.randrange(d)


            # here we prevent the sampling patch overflow volume
            # Vérifier que le volume est assez grand
            if w < patch_w or h < patch_h or d < patch_d:
                raise ValueError("Volume plus petit que le patch demandé")

            # Empêcher le patch de dépasser les limites du volume
            x = max(half_w, min(int(x), w - half_w))
            y = max(half_h, min(int(y), h - half_h))
            z = max(half_d, min(int(z), d - half_d))

            # Extraction du patch de l'IRM
            image_patch = image[
                x-half_w:x+half_w,
                y-half_h:y+half_h,
                z-half_d:z+half_d
            ].astype(np.float32)

            # Extraction du patch correspondant dans le masque
            label_patch = label[
                x-half_w:x+half_w,
                y-half_h:y+half_h,
                z-half_d:z+half_d
            ].astype(np.int64)
            


            #ajout des dimensions aux tableaux (pour réseau convolutif sous PyTorch (taille du batch, nb de cannaux, X,Y, Z) où X,Y,Z sont les dimensions spatiales)
            image_patch_list.append(image_patch[np.newaxis, np.newaxis, ...])
            label_patch_list.append(label_patch[np.newaxis, np.newaxis, ...])

        #Regrouper patches en batches
        image_out = np.concatenate(image_patch_list, axis=0)
        label_out = np.concatenate(label_patch_list, axis=0)

        #renvoie deux tableaux NumPy, patches d'images médicales et patches des masques associés
        return image_out, label_out

