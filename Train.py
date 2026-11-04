import os
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.autograd import Variable

from DataOp import TrainGenerator, get_list
from Network import MixAttNet
from Utils import check_dir, AvgMeter, dice_score

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Appareil utilisé :", device)

def adjust_lr(optimizer, iteration, num_iteration):
    """
    we decay the learning rate by a factor of 0.1 in 1/2 and 3/4 of whole training process
    """
    if iteration == num_iteration // 2:
        lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif iteration == num_iteration // 4 * 3:
        lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass


def loss_func(predict, label):
    """
    here we define the loss function, which you can upload additional loss in here
    """
    bce_loss = F.cross_entropy(predict, label)
    return bce_loss


def train_batch(net, optimizer, loader, patch_size, batch_size):
    net.train()

    image, label = loader.get_item()

    image = torch.from_numpy(image).float().to(device)
    label = torch.from_numpy(label).long().to(device)

    if label.ndim == 5 and label.shape[1] == 1:
        label = label.squeeze(1)

    predict = net(image)

    optimizer.zero_grad()

    weights = [1.0, 0.8, 0.7, 0.6, 0.5,0.8, 0.7, 0.6, 0.5]

    loss = sum(w * loss_func(pred, label) for w, pred in zip(weights, predict))
    loss.backward()
    optimizer.step()
    return loss.item()


def val(net, val_list, patch_size):
    net.eval()
    metric_meter = AvgMeter()
    device = next(net.parameters()).device

    for data_dict in val_list:

        image_path = data_dict['image_path']
        label_path = data_dict['label_path']

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # 1. Normalisation de l'IRM
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

        # 2. Padding si le volume est trop petit
        padding = [
            (0, max(0, patch_w - w)),
            (0, max(0, patch_h - h)),
            (0, max(0, patch_d - d))
        ]

        image_pad = np.pad(
            image, padding, mode='constant'
        )

        W, H, D = image_pad.shape

        # 3. Initialiser les 5 cartes de probabilités
        predict = np.zeros(
            (5, W, H, D), dtype=np.float32
        )

        # Nombre de prédictions reçues par voxel
        pre_count = np.zeros(
            (W, H, D), dtype=np.float32
        )

        # 4. Positions des patches
        def positions(dimension, taille_patch):
            pas = max(1, taille_patch // 4)

            return sorted(set(
                list(range(
                    0,
                    dimension - taille_patch + 1,
                    pas
                ))
                + [dimension - taille_patch]
            ))

        x_list = positions(W, patch_w)
        y_list = positions(H, patch_h)
        z_list = positions(D, patch_d)

        # 5. Parcourir l'IRM par patches
        with torch.no_grad():
            for x in x_list:
                for y in y_list:
                    for z in z_list:

                        image_patch = image_pad[
                            x:x+patch_w,
                            y:y+patch_h,
                            z:z+patch_d
                        ].astype(np.float32)

                        patch_tensor = torch.from_numpy(
                            image_patch[np.newaxis, np.newaxis, ...]
                        ).to(device)

                        # Prédiction du réseau
                        outputs = net(patch_tensor)

                        # Utiliser la sortie principale
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]

                        # Vérifier les 5 classes
                        if outputs.shape[1] != 5:
                            raise ValueError(
                                "Le réseau doit produire 5 canaux"
                            )

                        # Conversion des logits en probabilités
                        proba_patch = torch.softmax(
                            outputs, dim=1
                        )[0].cpu().numpy()

                        # 6. Accumuler les 5 cartes
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

        # 7. Moyenne des prédictions
        predict /= np.maximum(
            pre_count[np.newaxis, ...], 1
        )

        # 8. Choisir la classe la plus probable
        predict = np.argmax(
            predict, axis=0
        ).astype(np.uint8)

        # 9. Retirer le padding
        predict = predict[:w, :h, :d]

        # 10. Calculer le Dice multi-classes
        label = label.astype(np.uint8)

        dices = []

        for classe in range(1, 5):
            pred_classe = (predict == classe)
            label_classe = (label == classe)

            somme = pred_classe.sum() + label_classe.sum()

            if somme > 0:
                intersection = np.logical_and(
                    pred_classe, label_classe
                ).sum()

                dice = 2 * intersection / somme
                dices.append(dice)

        # Moyenne sur les classes présentes
        mean_dice = np.mean(dices) if dices else 1.0
        metric_meter.update(mean_dice)

    return metric_meter.avg


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    check_dir(args.output_path)
    ckpt_path = os.path.join(args.output_path, "ckpt")
    check_dir(ckpt_path)

    train_list, val_list, test_list = get_list(dir_path=args.data_path)

    train_generator = TrainGenerator(train_list,
                                     batch_size=args.batch_size,
                                     patch_size=args.patch_size)
    net = MixAttNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    open(os.path.join(args.output_path, "train_record.txt"), 'w+')

    loss_meter = AvgMeter()
    start_time = time.time()
    best_metric = 0.

    for iteration in range(1, args.num_iteration+1):
        adjust_lr(optimizer, iteration, args.num_iteration)
        train_loss = train_batch(net=net, optimizer=optimizer, loader=train_generator, patch_size=args.patch_size, batch_size=args.batch_size)
        loss_meter.update(train_loss)

        if iteration % args.pre_fre == 0:
            iteration_time = time.time() - start_time
            info = [iteration, loss_meter.avg, iteration_time]
            print("Iter[{}] | Loss: {:.3f} | Time: {:.2f}".format(*info))
            start_time = time.time()
            loss_meter.reset()

        if iteration % args.val_fre == 0:
            val_dice = val(net, val_list, args.patch_size)
            if val_dice > best_metric:
                torch.save(net.state_dict(), os.path.join(ckpt_path, "best_val.pth.gz"))
                best_metric = val_dice
            open(os.path.join(args.output_path, "train_record.txt"), 'a+').write("{:.3f} | {:.3f}\n".format(train_loss, val_dice))
            print("Val in Iter[{}] Dice: {:.3f}".format(iteration, val_dice))
        if iteration % 100 == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_path, "train_{}.pth.gz".format(iteration)))


if __name__ == '__main__':

    class Parser(object):
        def __init__(self):
            self.gpu_id = 0

            self.lr = 1e-3
            self.weight_decay = 1e-4
            self.batch_size = 4

            self.num_iteration = 4000
            self.val_fre = 200
            self.pre_fre = 20

            self.patch_size = (64, 64, 16)

            self.data_path = 'Data/'
            self.output_path = 'output/'

    parser = Parser()
    main(parser)
