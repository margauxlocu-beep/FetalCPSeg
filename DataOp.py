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
    image_name = 'image.nii.gz'
    label_name = 'label.nii.gz'

    #associe images & masques. Création dictionnaire contenant deux associations clé-valeur. On enregistre les chemins aux images
    for path in path_list:
        dict_list.append(
            {
                'image_path': os.path.join(path, image_name),
                'label_path': os.path.join(path, label_name),
            }
        )

    # we split the data set to train set(0.75), val set(0.05), test set(0.2)
    train_ratio = 0.75 #75% des cas pour l'entraînement
    val_ratio = 0.8 # Attention : 80% == limite set test + set validation donc 5% de val et donc bien 20% de test 
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
        self.aug = get_augmentation()

    def get_item(self): #pour obtenir un nouveau batch (de patient) d'entraînement

        dict_list = random.sample(self.data_list, self.batch_size) #choisit batch_size éléments dans la liste
        #extraction des fichiers (réccupère les chemins des fichiers sélectionnés)
        image_list = [dict_item['image_path'] for dict_item in dict_list]
        label_list = [dict_item['label_path'] for dict_item in dict_list]

        image_patch, label_patch = self._sample_patch(image_list, label_list) #génère les patches

        return image_patch, label_patch # retourne deux tableaux

    def _sample_patch(self, image_list, clean_list): #extractions des patches 3D
        half_size = self.patch_size // 2
        image_patch_list = []
        label_patch_list = []

        for image_path, clean_path in zip(image_list, clean_list): #parcours lites et extrait les chemins
            image = nib.load(image_path).get_fdata() #Récupère les valeurs du volume sous forme de tableau NumPy
            label = nib.load(clean_path).get_fdata() #Les deux volumes doivent avoir des dimensions compatibles et être spatialement alignés (label et image)

            # here we augment the corresponding data and label
            data = {'image': image, 'label': label} #dico avec image et masque de segmentation associé
            aug_data = self.aug(**data) #application des mêmes transformations que image et label
            image, label = aug_data['image'], aug_data['label']

            w, h, d = image.shape

            #Localiser les voxels de la structure à segmenter
            label_index = np.where(label == 1) #récupère les coordonnées de tous les voxels dont la valeur du masque est égale à 1 (ie voxel dans structure d'intérêt)=> 3 tableaux = coordonnées sur les axes x, y et z
            length_label = label_index[0].shape[0] #compte le nombre de voxels appartenant à la classe 1

            #choisir le centre du patch
            p = random.random()
            # we set a probability(p) to make most of the center of sampling patches
            # locate to the regions with label not background
            if p < 0.875: #Le centre du patch est choisi parmi les voxels dont le masque vaut 1. L'objectif est de privilégier les régions contenant la structure anatomique à apprendre.
                sample_id = random.randint(1, length_label-1) #choisit un indice aléatoire parmi les voxels appartenant à la classe 1
                x, y, z = label_index[0][sample_id], label_index[1][sample_id], label_index[2][sample_id] #coordonnées du voxel séléctionné. Il devient le centre théorique du patch.
            else: #Le centre du patch est choisi aléatoirement dans l'ensemble du volume. Cela permet aussi de sélectionner des régions d'arrière-plan et d'apprendre au réseau à les reconnaître.
                x, y, z = random.randint(0, w), random.randint(0, h), random.randint(0, d) #sinon centre aléatoire

            # here we prevent the sampling patch overflow volume
            if x < half_size:
                x = half_size
            elif x > w-half_size:
                x = w-half_size-1

            if y < half_size:
                y = half_size
            elif y > h-half_size:
                y = h-half_size-1

            if z < half_size:
                z = half_size
            elif z > d-half_size:
                z = d-half_size-1

            image_patch = image[x-half_size:x+half_size, y-half_size:y+half_size, z-half_size:z+half_size].astype(np.float32)
            label_patch = label[x-half_size:x+half_size, y-half_size:y+half_size, z-half_size:z+half_size].astype(np.float32)

            image_patch_list.append(image_patch[np.newaxis, np.newaxis, ...])
            label_patch_list.append(label_patch[np.newaxis, np.newaxis, ...])

        image_out = np.concatenate(image_patch_list, axis=0)
        label_out = np.concatenate(label_patch_list, axis=0)

        return image_out, label_out

