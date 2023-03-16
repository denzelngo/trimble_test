import os
from os.path import join
import argparse
import glob
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset/train', type=str, help='path to dataset')
    parser.add_argument('--val-size', default=0.2, type=float, help='percentage of validation set')
    arg = parser.parse_args()

    if 'train' not in arg.data:
        raise UserWarning('Please copy your data in a folder named train')
    val_path = arg.data.replace('train', 'val')
    if os.path.isdir(val_path):
        raise UserWarning(f'A created valid folder detected. Please verify if the dataset is splitted. '
                          f'If not, please delete the folder {val_path}')
    else:
        name_cls = os.listdir(arg.data)
        os.mkdir(val_path)
        for n in name_cls:
            os.mkdir(join(val_path, n))

    img_pths = glob.glob(join(arg.data, '*/*'))
    val_pths = random.sample(img_pths, int(arg.val_size * len(img_pths)))
    for f in val_pths:
        move_to = f.replace('train', 'val')
        os.rename(f, move_to)
    print(f'The validation set is created in {val_path}')
