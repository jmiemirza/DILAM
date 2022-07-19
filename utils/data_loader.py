import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import copy
from os.path import exists

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)
                                    ])

tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)
                                    ])

common_corruptions = ['cifar_new', 'gaussian_noise', 'original', 'shot_noise', 'impulse_noise', 'defocus_blur',
                      'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

valid_datasets = ['cifar10']


def prepare_test_data(args):
    if args.dataset == 'cifar10':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'initial':
            # print('Test on the original test set')
            teset = datasets.CIFAR10(root=args.dataroot, train=False,
                                     transform=te_transforms)
        elif args.corruption in common_corruptions:
            # print('Test on %s level %d' % (args.corruption, args.level))
            teset_raw = np.load(args.dataroot + f'/CIFAR-10-C/test/{args.corruption}.npy')
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = datasets.CIFAR10(root=args.dataroot, train=False,
                                     transform=te_transforms)
            teset.data = teset_raw
        else:
            raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1
    teloader = DataLoader(teset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers)
    return teset, teloader


def prepare_train_data(args):
    print('Preparing data...')
    if args.dataset == 'cifar10':
        trset = datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                 train=True)

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers)
    return trset, trloader


def prepare_train_valid_loaders(args):
    if not hasattr(args, 'workers'):
        args.workers = 1
    assert args.train_val_split > 0 and args.train_val_split < 1

    train_set_raw = np.load(args.dataroot + f'/CIFAR-10-C/train/{args.corruption}.npy')
    train_set = datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                 train=True)
    valid_set = datasets.CIFAR10(root=args.dataroot, transform=te_transforms,
                                 train=True)
    train_set.data = train_set_raw
    valid_set.data = copy.deepcopy(train_set_raw)

    if args.train_val_split_seed != 0:
        torch.manual_seed(args.train_val_split_seed)
    indices = torch.randperm(len(train_set))
    valid_size = round(len(train_set) * args.train_val_split)
    train_set = Subset(train_set, indices[:-valid_size])
    valid_set = Subset(valid_set, indices[-valid_size:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    return train_loader, valid_loader


def dataset_checks(args, corruptions):
    if not args.dataset in valid_datasets:
        raise Exception(f'Invalid dataset argument: {args.dataset}')

    error = False
    if args.dataset == 'cifar10':
        error = check_cifar_ten_c(args, corruptions)

    if error:
        raise Exception('Dataset checks unsuccessful!')
    else:
        print('Dataset checks successful!')


def check_cifar_ten_c(args, corruptions):
    datasets.CIFAR10(root=args.dataroot, download=True)
    error = False
    test_set_path = args.dataroot + '/CIFAR-10-C/test/'
    train_set_path = args.dataroot + '/CIFAR-10-C/train/'
    if not exists(test_set_path):
        error = True
        print(f'CIFAR-10-C test set not found. Expected at {test_set_path}')
    if not exists(train_set_path):
        error = True
        print(f'CIFAR-10-C training set not found. Expected at {train_set_path}')
    missing_files = []
    for corruption in corruptions:
        test_samples = test_set_path + f'{corruption}.npy'
        train_samples = train_set_path + f'{corruption}.npy'
        if not exists(test_samples):
            missing_files.append(test_samples)
        if not exists(train_samples):
            missing_files[:0] = [train_samples]
    if len(missing_files):
        error = True
        print('Missing the following CIFAR-10-C samples:')
        for f_path in missing_files:
            print(f_path)
    return error


def check_kitti(args):
    datasets.Kitti(root=args.dataroot, download=True)
    ...

