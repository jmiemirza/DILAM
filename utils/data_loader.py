import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import copy

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


def prepare_test_data(args, corruption=None):
    if corruption is not None:
        args.corruption = corruption

    if args.dataset == 'cifar10':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'initial':
            # print('Test on the original test set')
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                 train=False, download=False,
                                                 transform=te_transforms)

        elif args.corruption in common_corruptions:
            # print('Test on %s level %d' % (args.corruption, args.level))
            teset_raw = np.load(args.dataroot + '/CIFAR-10-C/test/%s.npy' % (args.corruption))
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                 train=False, download=False,
                                                 transform=te_transforms)

            teset.data = teset_raw

        else:
            raise Exception('Corruption not found!')
    # elif args.dataset == 'kitti':
    #     teset = torchvision.datasets.Kitti(root=args.dataroot,
    #                                        train=False, download=True,
    #                                        transform=te_transforms)
    else:
        raise Exception('Dataset not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1

    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.workers)

    return teset, teloader


def prepare_train_data(args):
    print('Preparing data...')
    if args.dataset == 'cifar10':
        trset = torchvision.datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                             train=True, download=True)
    else:
        raise Exception('Dataset not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.workers)
    return trset, trloader


def prepare_train_valid_loaders(args):
    if args.dataset != 'cifar10':
        raise Exception(f'Invalid dataset argument: \'{args.dataset}\'')
    if not args.corruption in common_corruptions:
        raise Exception('Corruption not found!')
    if not hasattr(args, 'workers'):
        args.workers = 1
    assert args.train_val_split > 0 and args.train_val_split < 1

    train_set_raw = np.load(args.dataroot + '/CIFAR-10-C/train/%s.npy' % (args.corruption))
    train_set = datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                 train=True, download=False)
    valid_set = datasets.CIFAR10(root=args.dataroot, transform=te_transforms,
                                 train=True, download=False)
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

