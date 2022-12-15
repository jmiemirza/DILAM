import os
from os.path import join

import numpy as np
import torch
# from colorama import Fore


def get_grad(params):
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(filter(lambda p: p.grad is not None, params))
    grad = [p.grad.data.cpu().view(-1) for p in params]
    return torch.cat(grad)


def write_to_txt(name, content):
    with open(name, 'w') as text_file:
        text_file.write(content)


def make_dirs(path):
    os.makedirs(path, exist_ok=True)


def print_args(opt):
    for arg in vars(opt):
        print('%s %s' % (arg, getattr(opt, arg)))


def mean(ls):
    return sum(ls) / len(ls)


def normalize(v):
    return (v - v.mean()) / v.std()


def flat_grad(grad_tuple):
    return torch.cat([p.view(-1) for p in grad_tuple])


def print_nparams(model):
    nparams = sum([param.nelement() for param in model.parameters()])
    print('number of parameters: %d' % (nparams))


# def print_color(color, string):
#     print(getattr(Fore, color) + string + Fore.RESET)


def plot_adaptation_err(all_err_cls, corr, args):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    fig, _ = plt.subplots()

    plt.plot(all_err_cls, color='r', label=corr)
    plt.xlabel('Number of Samples for Adaptation')
    plt.ylabel('Test Error (%)')
    plt.legend()
    plt.savefig(os.path.join(args.outf, corr), format="png")
    plt.close(fig)


def timedelta_to_str(timedelta, explicit_days=False):
    s = ''
    if explicit_days:
        s = f'{timedelta.days} Days, {timedelta.seconds // 3600:02}:'
    else:
        total_hrs = timedelta.days * 24 + timedelta.seconds // 3600
        s = f'{str(total_hrs).zfill(2 if total_hrs < 100 else 3)}:'
    s += f'{(timedelta.seconds % 3600) // 60:02}:{timedelta.seconds % 60:02}'
    return s


def setup_tiny_imagenet_val_dir(val_dir_path, val_num_imgs=10000, rm_initial=False):
    """
        Tiny ImageNet validation set comes with 10k images from all 200 classes
        placed in the same folder (images) and a val_annotations.txt pointing
        out which image belongs to which class.
        This method moves all of the images into an image folder inside a folder
        named after the class they belong to.
    """
    import glob
    from os.path import exists, join, split
    from shutil import copy, move

    from tqdm import tqdm

    val_dict = {}
    with open(f'{val_dir_path}/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.iglob(join(val_dir_path, 'images', '*'))
    for path in tqdm(paths, total=val_num_imgs):
        file = split(path)[1]
        folder = val_dict[file]
        if not exists(val_dir_path + str(folder)):
            make_dirs(join(val_dir_path, str(folder), 'images'))
        # copy(path, join(val_dir_path, str(folder), 'images', str(file)))
        move(path, join(val_dir_path, str(folder), 'images', str(file)))

    if rm_initial:
        os.rmdir(join(val_dir_path, 'images'))
        os.remove(join(val_dir_path, 'val_annotations.txt'))


def kitti_labels_to_yolo(dataroot):
    from cv2 import imread

    print('Converting KITTI labels to YOLO label format.')

    imgs_dir = join(dataroot, 'raw', 'training', 'image_2')
    labels_dir = join(dataroot, 'raw', 'training', 'label_2')
    save_at_dir = join(dataroot, 'raw', 'yolo_style_labels')
    make_dirs(save_at_dir)

    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                   'Cyclist', 'Tram', 'Misc']
    img_file_names = sorted(os.listdir(imgs_dir))
    label_file_names = sorted(os.listdir(labels_dir))
    label_dict = dict(zip(class_names, range(len(class_names))))

    for img_file_name, label_file_name in zip(img_file_names, label_file_names):
        img_path = join(imgs_dir, img_file_name)
        label_path = join(labels_dir, label_file_name)

        img = imread(img_path)
        img_height, img_width, = img.shape[:2]

        with open(label_path, 'r') as f:
            label_lines = f.readlines()

        yolo_label_file = open(join(save_at_dir, label_file_name), 'w')

        for line in label_lines:
            label_entry = line.split(' ')
            if len(label_entry) != 15:
                raise Exception(f'Faulty original label in: {label_file_name}')

            class_name = label_entry[0]
            if class_name == 'DontCare':
                continue

            x1 = float(label_entry[4]) # left
            y1 = float(label_entry[5]) # top
            x2 = float(label_entry[6]) # right
            y2 = float(label_entry[7]) # bottom

            bbox_center_x = (x1 + x2) / 2.0 / img_width
            bbox_center_y = (y1 + y2) / 2.0 / img_height
            bbox_width = float((x2 - x1) / img_width)
            bbox_height = float((y2 - y1) / img_height)

            yolo_label_line = f'{label_dict[class_name]} {bbox_center_x} ' \
                              f'{bbox_center_y} {bbox_width} {bbox_height}\n'
            yolo_label_file.write(yolo_label_line)
        yolo_label_file.close()

