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


def loss_func(predict, label, pos_weight):
    """
    here we define the loss function, which you can upload additional loss in here
    """
    bce_loss = F.cross_entropy(predict, label)
    return bce_loss


def train_batch(net, optimizer, loader, patch_size, batch_size):
    net.train()

    image, label = loader.get_item()

    image = torch.from_numpy(image).float().cuda()
    label = torch.from_numpy(label).long().cuda()

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

    for data_dict in val_list:

        image_path = data_dict['image_path']
        label_path = data_dict['label_path']

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        #Normalisation de l'IRM
        image = image.astype(np.float32)
        zone = image[image > 0]

        if zone.size > 0:
            mean = zone.mean()
            std = zone.std()
            image[image > 0] = (zone - mean) / (std + 1e-8)

        pre_count = np.zeros_like(image, dtype=np.float32)
        predict = np.zeros_like(image, dtype=np.float32)

        w, h, d = image.shape
        # Récupérer les dimensions du patch
        patch_w, patch_h, patch_d = patch_size

        # Fonction pour déterminer les positions des patches
        def positions(dimension, taille_patch):
            if dimension < taille_patch:
                raise ValueError("Patch plus grand que le volume")

            pas = max(1, taille_patch // 4)

            return sorted(set(
                list(range(0, dimension - taille_patch + 1, pas))
                + [dimension - taille_patch]
            ))

        # Positions de départ sur chaque axe
        x_list = positions(w, patch_w)
        y_list = positions(h, patch_h)
        z_list = positions(d, patch_d)

        with torch.no_grad():
            for x in x_list:
                for y in y_list:
                    for z in z_list:

                        # Extraction du patch
                        image_patch = image[
                            x:x+patch_w,
                            y:y+patch_h,
                            z:z+patch_d
                        ].astype(np.float32)

                        # Conversion en tenseur PyTorch
                        image_patch_tensor = torch.from_numpy(
                            image_patch[np.newaxis, np.newaxis, ...]
                        ).cuda()

                        # Prédiction du réseau (version binaire provisoire)
                        pre_patch = net(image_patch_tensor)[0, 0].cpu().numpy()

                        # Ajouter la prédiction au volume
                        predict[
                            x:x+patch_w,
                            y:y+patch_h,
                            z:z+patch_d
                        ] += pre_patch

                        # Compter combien de fois chaque voxel est prédit
                        pre_count[
                            x:x+patch_w,
                            y:y+patch_h,
                            z:z+patch_d
                        ] += 1
        predict /= pre_count
        metric_meter.update(dice_score(predict, label))

    return metric_meter.avg


def main(args):
    torch.cuda.set_device(args.gpu_id)

    check_dir(args.output_path)
    ckpt_path = os.path.join(args.output_path, "ckpt")
    check_dir(ckpt_path)

    train_list, val_list, test_list = get_list(dir_path=args.data_path)

    train_generator = TrainGenerator(train_list,
                                     batch_size=args.batch_size,
                                     patch_size=args.patch_size)
    net = MixAttNet().cuda()
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
