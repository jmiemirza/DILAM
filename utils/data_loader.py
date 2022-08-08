import copy
import logging
import numpy as np
from os.path import exists
from torch import manual_seed, randperm
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import CIFAR10, ImageFolder, vision
import torchvision.transforms as transforms
from utils.dataset_wrappers import ImageFolderWrap, CIFAR10C
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


def get_test_loader(args):
    teset = fetch_dataset(args, train=False)

    if not hasattr(args, 'workers'):
        args.workers = 1
    teloader = DataLoader(teset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers)
    return teloader


def get_train_loader(args):
    trset = fetch_dataset(args, train=True)

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers)
    return trloader


def fetch_dataset(args, train=True):
    if hasattr(args, 'task') and args.task not in ['initial'] + TASKS:
        raise Exception(f'Invalid task: {args.task}')
    split = 'train' if train else 'val'

    if args.dataset == 'cifar10':
        trfs = tr_transforms if train else te_transforms
        if not hasattr(args, 'task') or args.task == 'initial':
            ds = CIFAR10(root=args.dataroot, transform=trfs, train=train)
        elif args.task in TASKS:
            # TODO use args.level only, with files containing all corruption levels
            level = 1 if train else args.level
            ds = CIFAR10C(args.dataroot + '/CIFAR-10-C', args.task, train=train,
                          transform=trfs, severity=level)

    elif args.dataset == 'tiny-imagenet':
        trfs = tr_transforms_tiny if train else te_transforms
        if not hasattr(args, 'task') or args.task == 'initial':
            ds = ImageFolderWrap(f'{args.dataroot}/tiny-imagenet-200/{split}', trfs)
        elif args.task in TASKS:
            path = f'{args.dataroot}/tiny-imagenet-200-c/{split}/{args.task}/{args.level}'
            ds = ImageFolderWrap(path, trfs)

    elif args.dataset == 'imagenet':
        trfs = tr_transforms_imgnet if train else te_transforms_imgnet
        if not hasattr(args, 'task') or args.task == 'initial':
            ds = ImageFolderWrap(f'{args.dataroot}/imagenet/{split}', trfs)
        elif args.task in TASKS:
            path = f'{args.dataroot}/imagenet-c/{split}/{args.task}/{args.level}'
            ds = ImageFolderWrap(path, trfs)

    return ds


def get_train_valid_loaders(args):
    if args.dataset == 'cifar10':
        return get_cifar10_train_valid_loaders(args)
    return get_train_loader(args), get_test_loader(args)


# TODO just split the cifar dataset...
from PIL import Image
def get_pil_image_from_idx(self, idx: int = 0):
        return Image.fromarray(self.dataset.data[idx])
Subset.get_pil_image_from_idx = get_pil_image_from_idx

def get_cifar10_train_valid_loaders(args):
    log.debug(f'Preparing training and validation data for task {args.task}')
    if not hasattr(args, 'workers'):
        args.workers = 1
    assert args.train_val_split > 0 and args.train_val_split < 1

    train_set = CIFAR10(root=args.dataroot, transform=tr_transforms, train=True)
    valid_set = CIFAR10(root=args.dataroot, transform=te_transforms, train=True)
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
    # train_set
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)
    return train_loader, valid_loader


def get_joint_loader(args, tasks):
    datasets = []
    if args.dataset == 'cifar10':
        if 'initial' in tasks:
            tasks.remove('initial')
        datasets.append(CIFAR10(root=args.dataroot, transform=tr_transforms, train=True))
        datasets.append(CIFAR10(root=args.dataroot, transform=te_transforms, train=False))
        datasets.append(CIFAR10C(args.dataroot + '/CIFAR-10-C', *tasks, train=True,
                                 transform=tr_transforms, severity=1))
        datasets.append(CIFAR10C(args.dataroot + '/CIFAR-10-C', *tasks, train=False,
                                 transform=te_transforms, severity=1))
    else:
        tmp = args.task
        for args.task in tasks:
            datasets.append(fetch_dataset(args, train=True))
            datasets.append(fetch_dataset(args, train=False))
        args.task = tmp

    if not hasattr(args, 'workers'):
        args.workers = 1
    joint_loader = DataLoader(ConcatDataset(datasets),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)
    return joint_loader


def dataset_checks(args):
    if not args.dataset in VALID_DATASETS:
        raise Exception(f'Invalid dataset argument: {args.dataset}')

    error = False
    if args.dataset == 'cifar10':
        error = check_cifar10_c(args)

    if error:
        log.critical('Dataset checks unsuccessful!')
        raise Exception('Dataset checks unsuccessful!')
    else:
        log.info('Dataset checks successful!')


def check_cifar10_c(args):
    CIFAR10(root=args.dataroot, download=True)
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


# def check_kitti(args):
#     datasets.Kitti(root=args.dataroot, download=True)
#     ...

