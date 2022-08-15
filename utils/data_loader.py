import copy
import logging
import numpy as np
from os.path import exists
from torch import manual_seed, randperm
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from utils.dataset_wrappers import CIFAR, KITTI, ImgNet
from globals import *

log = logging.getLogger('MAIN.DATA')

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)
                                    ])

tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)
                                    ])

tr_transforms_tiny = transforms.Compose([transforms.RandomCrop(64, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(*NORM)
                                         ])

NORM_IMGNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

tr_transforms_imgnet = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*NORM_IMGNET)
                                           ])

te_transforms_imgnet = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*NORM_IMGNET)
                                           ])

# TODO
tr_transforms_kitti = tr_transforms_imgnet
te_transforms_kitti = te_transforms_imgnet


def get_test_loader(args):
    teset = fetch_dataset(args, split='test')
    return DataLoader(teset, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.workers)


def get_train_loader(args):
    trset = fetch_dataset(args, split='train')
    return DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.workers)


def get_val_loader(args):
    valset = fetch_dataset(args, split='val')
    return DataLoader(valset, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.workers)


def fetch_dataset(args, *, split: str = None):
    if not hasattr(args, 'task'):
        args.task = 'initial'
    if args.task not in ['initial'] + TASKS:
        raise Exception(f'Invalid task: {args.task}')

    if args.dataset == 'cifar10':
        transform = tr_transforms if split == 'train' else te_transforms
        ds = CIFAR(args.dataroot, args.task, split=split, transform=transform,
                   severity=args.level)
        if split != 'test':
            # train and val split are being created from the train set
            ds = get_split_subset(args, ds, split)

    elif args.dataset in ['imagenet', 'imagenet-mini']:
        transform = tr_transforms_imgnet if split == 'train' else te_transforms_imgnet
        ds = ImgNet(args.dataroot, split, args.task, args.level, transform)
        if split != 'val':
            # train and test split are being created from the train set
            ds = get_split_subset(args, ds, split)

    # TODO
    # elif args.dataset == 'kitti':
    #     trfs = tr_transforms_kitti if train else te_transforms_kitti
    #     if not hasattr(args, 'task') or args.task == 'initial':
    #         ds = KITTI(args.dataroot, split, 'initial', args.level, transforms=...)
    #     elif args.task in TASKS:
    #         ds = KITTI(args.dataroot, split, args.task, args.level, transforms=...)

    return ds


def get_split_subset(args, ds, split):
    manual_seed(args.split_seed)
    indices = randperm(len(ds))
    valid_size = round(len(ds) * args.split_ratio)

    if args.dataset == 'cifar10':
        if split == 'train':
            ds = Subset(ds, indices[:-valid_size])
        elif split == 'val':
            ds = Subset(ds, indices[-valid_size:])

    elif args.dataset in ['imagenet', 'imagenet-mini']:
        if split == 'train':
            ds = Subset(ds, indices[:-valid_size])
        elif split == 'test':
            ds = Subset(ds, indices[-valid_size:])

    return ds


def get_pil_image_from_idx(self, idx: int = 0):
    return self.dataset.get_pil_image_from_idx(idx)
Subset.get_pil_image_from_idx = get_pil_image_from_idx


def get_joint_loader(args, tasks):
    datasets = []
    for args.task in tasks:
        datasets.append(fetch_dataset(args, split='train'))
        datasets.append(fetch_dataset(args, split='test'))
        datasets.append(fetch_dataset(args, split='val'))

    joint_loader = DataLoader(ConcatDataset(datasets),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)
    return joint_loader


from os.path import join, normpath
def dataset_checks(args):
    if not args.dataset in VALID_DATASETS:
        raise Exception(f'Invalid dataset argument: {args.dataset}')

    error = False
    if args.dataset == 'cifar10':
        error = check_cifar10_c(args)
    elif args.dataset in ['imagenet', 'imagenet-mini']:
        error = check_imgnet_c(args)

    if error:
        log.critical('Dataset checks unsuccessful!')
        raise Exception('Dataset checks unsuccessful!')
    else:
        log.info('Dataset checks successful!')


def check_cifar10_c(args):
    CIFAR10(root=args.dataroot, download=True)
    error = False
    test_set_path = join(args.dataroot, 'CIFAR-10-C', 'test')
    train_set_path = join(args.dataroot, 'CIFAR-10-C', 'train')
    if not exists(test_set_path):
        error = True
        log.error(f'CIFAR-10-C test set not found. Expected at {test_set_path}')
    if not exists(train_set_path):
        error = True
        log.error(f'CIFAR-10-C training set not found. Expected at {train_set_path}')
    missing_files = []
    for task in TASKS:
        test_samples = join(test_set_path, task + '.npy')
        train_samples = join(train_set_path, task + '.npy')
        if not exists(test_samples):
            missing_files.append(test_samples)
        if not exists(train_samples):
            missing_files[:0] = [train_samples]
    if len(missing_files):
        error = True
        log.error('Missing the following CIFAR-10-C samples:')
        for f_path in missing_files:
            log.error(normpath(f_path))
    return error


def check_imgnet_c(args):
    error = False
    val_set_path = join(args.dataroot, args.dataset + '-c', 'val')
    train_set_path = join(args.dataroot, args.dataset + '-c', 'train')

    if not exists(val_set_path):
        error = True
        log.error(f'{args.dataset.capitalize()} validation set not found. '
                  f'Expected at {val_set_path}')
    if not exists(train_set_path):
        error = True
        log.error(f'{args.dataset.capitalize()} training set not found. '
                  f'Expected at {train_set_path}')
    missing_dirs = []
    for task in TASKS:
        for severity in SEVERTITIES:
            val_samples_dir = join(val_set_path, task, str(severity))
            train_samples_dir = join(train_set_path, task, str(severity))
            if not exists(val_samples_dir):
                missing_dirs.append(val_samples_dir)
            if not exists(train_samples_dir):
                missing_dirs[:0] = [train_samples_dir]
    if len(missing_dirs):
        error = True
        log.error(f'Missing the following {args.dataset.capitalize()} directories:')
        for f_path in missing_dirs:
            log.error(normpath(f_path))
    return error



# def check_kitti(args):
#     datasets.Kitti(root=args.dataroot, download=True)
#     ...

