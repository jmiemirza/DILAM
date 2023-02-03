import logging
import os
from os.path import exists, join, normpath
from pathlib import Path

import torch

from torch import manual_seed, randperm
from torch.utils.data import DataLoader

import globals
from utils.datasets import LoadImagesAndLabels as Kitti
from utils.datasets import LoadImagesForTaskClassification
from utils.general import check_img_size, increment_path
from utils.torch_utils import torch_distributed_zero_first

log = logging.getLogger('MAIN.DATA')


def get_loader(args, split='train', joint=False, for_classification=False,
               pad=0.0, aug=False, rect=False, shuffle=True):
        if rect and shuffle and not for_classification:
            shuffle = False
        ds = get_dataset(args, split=split, for_classification=for_classification,
                         joint=joint, pad=pad, aug=aug, rect=rect)
        collate_fn = None if for_classification else Kitti.collate_fn
        # sampler = torch.utils.data.distributed.DistributedSampler(ds) #if rank != -1 else None
        loader = DataLoader# if args.image_weights else InfiniteDataLoader
        return loader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, collate_fn=collate_fn,
                      pin_memory=True, drop_last=True)


def get_dataset(args, split=None, joint=False, for_classification=False, pad=0.0, aug=False, rect=False):
    """
        Create dataset
    """
    if not hasattr(args, 'task'):
        args.task = 'initial'
    if args.task not in ['initial'] + globals.TASKS:
        raise Exception(f'Invalid task: {args.task}')

    path = join(args.dataroot, f'{split}.txt')
    img_size_idx = split != 'train' # idx = 0 for train, else 1
    img_size = check_img_size(img_size=args.img_size[img_size_idx], s=args.gs)
    img_dirs_path = []
    if joint:
        for t in ['initial'] + globals.TASKS:
            if t != 'initial':
                if args.severity_idx < len(globals.KITTI_SEVERITIES[args.task]):
                    args.severity = globals.KITTI_SEVERITIES[t][args.severity_idx]
                else:
                    continue
            img_dir = 'images' if t == 'initial' else f'{args.severity}'
            img_dirs_path.append(join(args.dataroot, f'{t}', img_dir))
            if t == args.task:
                break
    else:
        img_dir = 'images' if args.task == 'initial' else f'{args.severity}'
        img_dirs_path.append(join(args.dataroot, f'{args.task}', img_dir))

    ds_class = LoadImagesForTaskClassification if for_classification else Kitti
    with torch_distributed_zero_first(-1):
        ds = ds_class(path=path, img_size=img_size, batch_size=args.batch_size,
                      augment=aug, hyp=args.yolo_hyp(), rect=rect,
                      stride=int(args.gs), pad=pad, img_dirs=img_dirs_path)
    return ds


def set_yolo_save_dir(args, baseline, scenario):
    """
        Sets args.save_dir which is used in yolov3 training to save results
    """
    p = join(args.checkpoints_path, args.dataset, args.model, baseline,
             scenario, f'{args.task}_{args.severity}_train_results')
    args.save_dir = increment_path(Path(p), exist_ok=args.exist_ok)


def set_severity(args):
    """
        Sets args.severity to the current severity and returns True on success.
        For the KITTI dataset this will get the appropriate severity for the
        current task. In case of different number of severities among tasks,
        False is returned if current args.severity_idx does not exist for the
        current task.
    """
    if args.task == 'initial':
        return True

    if args.severity_idx < len(globals.KITTI_SEVERITIES[args.task]):
        args.severity = globals.KITTI_SEVERITIES[args.task][args.severity_idx]
        return True

    return False


def get_all_severities_str(args):
    all_severities_str = ''
    for task in globals.TASKS:
        if args.dataset != 'kitti':
            all_severities_str = f'{args.robustness_severities[args.severity_idx]}_'
            break
        elif args.severity_idx < len(globals.KITTI_SEVERITIES[task]):
            all_severities_str += f'{globals.KITTI_SEVERITIES[task][args.severity_idx]}_'
    return all_severities_str


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

