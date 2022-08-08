import os
import torch
from colorama import Fore
import numpy as np


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


def print_color(color, string):
    print(getattr(Fore, color) + string + Fore.RESET)


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


def setup_tiny_imagenet_val_dir(val_dir_path, val_num_imgs=10000, rm_initial=False):
    """
        Tiny ImageNet validation set comes with 10k images from all 200 classes
        placed in the same folder (images) and a val_annotations.txt pointing
        out which image belongs to which class.
        This method moves all of the images into an image folder inside a folder
        named after the class they belong to.
    """
    import glob
    from shutil import move, copy
    from os.path import join, split, exists
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
