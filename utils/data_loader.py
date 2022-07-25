import copy
import logging
from os.path import exists
from torch import manual_seed, randperm
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
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


def prepare_test_data(args):
    if args.dataset == 'cifar10':
        tesize = 10000
        if not hasattr(args, 'task') or args.task == 'initial':
            # log.debug('Test on the original test set')
            teset = datasets.CIFAR10(root=args.dataroot, train=False,
                                     transform=te_transforms)
        elif args.task in TASKS:
            # log.debug('Test on %s level %d' % (args.task, args.level))
            teset_raw = np.load(args.dataroot + f'/CIFAR-10-C/test/{args.task}.npy')
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = datasets.CIFAR10(root=args.dataroot, train=False,
                                     transform=te_transforms)
            teset.data = teset_raw
        else:
            raise Exception('Task not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1
    teloader = DataLoader(teset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers)
    return teset, teloader


def prepare_train_data(args):
    log.debug('Preparing original training data...')
    if args.dataset == 'cifar10':
        trset = datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                 train=True)

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers)
    return trset, trloader


def prepare_train_valid_data(args):
    log.debug(f'Preparing training and validation data for task {args.task}')
    if not hasattr(args, 'workers'):
        args.workers = 1
    assert args.train_val_split > 0 and args.train_val_split < 1

    train_set = datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                 train=True)
    valid_set = datasets.CIFAR10(root=args.dataroot, transform=te_transforms,
                                 train=True)
    if args.task != 'initial':
        train_set_raw = np.load(args.dataroot + f'/CIFAR-10-C/train/{args.task}.npy')
        # This currently asumes files generated for level 5 only.
        # Uncomment the following 2 lines for files containing all levels.
        # tesize = 50000
        # train_set_raw = train_set_raw[(args.level - 1) * tesize: args.level * tesize]
        train_set.data = train_set_raw
        valid_set.data = copy.deepcopy(train_set_raw)

    if args.train_val_split_seed != 0:
        manual_seed(args.train_val_split_seed)
    indices = randperm(len(train_set))
    valid_size = round(len(train_set) * args.train_val_split)
    train_set = Subset(train_set, indices[:-valid_size])
    valid_set = Subset(valid_set, indices[-valid_size:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)
    return train_set.dataset, train_loader, valid_set.dataset, valid_loader


def prepare_joint_loader(args):
    log.debug(f'Preparing joint (training + test) data for task {args.task}')
    train_set_raw = np.load(args.dataroot + f'/CIFAR-10-C/train/{args.task}.npy')
    # This currently asumes files generated for level 5 only.
    # Uncomment the following 2 lines for files containing all levels.
    # tesize = 50000
    # train_set_raw = train_set_raw[(args.level - 1) * tesize: args.level * tesize]
    train_set = datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                 train=True)
    train_set.data = train_set_raw

    test_size = 10000
    test_set_raw = np.load(args.dataroot + f'/CIFAR-10-C/test/{args.task}.npy')
    test_set_raw = test_set_raw[(args.level - 1) * test_size: args.level * test_size]
    test_set = datasets.CIFAR10(root=args.dataroot, train=False,
                                transform=te_transforms)
    test_set.data = test_set_raw

    joint_set = ConcatDataset([train_set, test_set])

    if not hasattr(args, 'workers'):
        args.workers = 1

    joint_loader = DataLoader(joint_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)

    return joint_loader


def dataset_checks(args):
    if not args.dataset in VALID_DATASETS:
        raise Exception(f'Invalid dataset argument: {args.dataset}')

    error = False
    if args.dataset == 'cifar10':
        error = check_cifar_ten_c(args)

    if error:
        log.critical('Dataset checks unsuccessful!')
        raise Exception('Dataset checks unsuccessful!')
    else:
        log.info('Dataset checks successful!')


def check_cifar_ten_c(args):
    datasets.CIFAR10(root=args.dataroot, download=True)
    error = False
    test_set_path = args.dataroot + '/CIFAR-10-C/test/'
    train_set_path = args.dataroot + '/CIFAR-10-C/train/'
    if not exists(test_set_path):
        error = True
        log.error(f'CIFAR-10-C test set not found. Expected at {test_set_path}')
    if not exists(train_set_path):
        error = True
        log.error(f'CIFAR-10-C training set not found. Expected at {train_set_path}')
    missing_files = []
    for task in TASKS:
        test_samples = test_set_path + f'{task}.npy'
        train_samples = train_set_path + f'{task}.npy'
        if not exists(test_samples):
            missing_files.append(test_samples)
        if not exists(train_samples):
            missing_files[:0] = [train_samples]
    if len(missing_files):
        error = True
        log.error('Missing the following CIFAR-10-C samples:')
        for f_path in missing_files:
            log.error(f_path)
    return error


def check_kitti(args):
    datasets.Kitti(root=args.dataroot, download=True)
    ...

