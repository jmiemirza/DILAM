import logging
from copy import deepcopy
from os.path import exists, join
from statistics import mean

from torch import load

import globals
from dua import dua
from utils.data_loader import get_all_severities_str, get_loader, set_severity
from utils.results_manager import ResultsManager
from utils.testing import test
from utils.testing_yolov3 import test as test_yolo

log = logging.getLogger('MAIN.DISC')


def disc(args, net):
    bn_stats_file_name = f'{get_all_severities_str(args)}BN_stats.pt'
    if not args.no_disc_adaption:
        disc_adaption(args, net, True, True, bn_stats_file_name)
    disc_plug_and_play(args, net, f'{get_all_severities_str(args)}BN_stats.pt')
    # fast_disc_plug_and_play(args, net, f'{get_all_severities_str(args)}BN_stats.pt')


def disc_adaption(args, net, save_bn_stats=True, use_training_data=True, save_fname=None):
    dua(args, net, save_bn_stats, use_training_data, save_fname)


def disc_plug_and_play(args, net, bn_file_name=None):
    ckpt = load(args.ckpt_path)
    metric = 'Error'
    if args.model == 'yolov3':
        metric = 'mAP@50'

    bn_file_path = join('checkpoints', args.dataset, args.model)
    if not bn_file_name:
        bn_file_name = 'BN_stats.pt'
    bn_file_path = join(bn_file_path, bn_file_name)
    if exists(bn_file_path):
        load_bn_stats_file(net, bn_file_path)
    if not all(task in net.bn_stats for task in globals.TASKS):
        raise Exception('BN Stats not containing all tasks')
    log.info(f'Using {bn_file_path} batch norm running estimates checkpoint')
    tasks = ['initial'] + globals.TASKS
    results = ResultsManager()

    log.info('::: DISC Plug & Play :::')
    for idx, args.task in enumerate(tasks):
        if not set_severity(args):
            continue
        severity_str = '' if args.task == 'initial' else f'Severity: {args.severity}'
        log.info(f'Start evaluation for Task-{idx} ({args.task}). {severity_str}')
        current_results = []

        for i in range(0, idx + 1):
            args.task = tasks[i]
            if not set_severity(args):
                continue
            load_bn_stats(net, args.task, ckpt)
            test_loader = get_loader(args, split='test', pad=0.5, rect=True)
            if args.model == 'yolov3':
                res = test_yolo(model=net, dataloader=test_loader,
                                iou_thres=args.iou_thres, conf_thres=args.conf_thres,
                                augment=args.augment)[0] * 100
            else:
                res = test(test_loader, net)[0] * 100
            current_results.append(res)
            log.info(f'    {metric} on Task-{i} ({tasks[i]}): {res:.1f}')

            if i == idx:
                mean_result = mean(current_results)
                log.info(f'Mean {metric.lower()} over current task ({tasks[i]}) '
                         f'and previously seen tasks: {mean_result:.1f}')
                severity_str = '' if args.task == 'initial' else f'{args.severity}'
                results.add_result('DISC', f'{tasks[i]} {severity_str}', mean_result, 'online')
                results.add_result('DISC', f'{tasks[i]} {severity_str}', mean_result, 'offline')


def load_bn_stats(net, task, ckpt=None):
    """
        Optionally loads a model checkpoint.
        Loads the running estimates for all batch norm layers
            from net.bn_stats attribute, for a given task.
            Unless the task is 'initial', in which case the function
            returns without loading anything.
        Sets network to evaluation mode.
    """
    if ckpt is not None:
        net.load_state_dict(ckpt)
    if task == 'initial':
        net.eval()
        return
    state_dict = net.state_dict()
    for layer_name, value in net.bn_stats[task].items():
        state_dict[layer_name + '.running_mean'] = value['running_mean']
        state_dict[layer_name + '.running_var'] = value['running_var']
    net.load_state_dict(state_dict)
    net.eval()


def load_bn_stats_file(net, file_path):
    """
        Loads a saved state of net.bn_stats attribute, located at file_path.
    """
    net.bn_stats = load(file_path)


def fast_disc_plug_and_play(args, net, bn_file_name=None):
    ckpt = load(args.ckpt_path)
    metric = 'Error'
    if args.model == 'yolov3':
        metric = 'mAP@50'

    bn_file_path = join('checkpoints', args.dataset, args.model)
    if not bn_file_name:
        bn_file_name = 'BN_stats.pt'
    bn_file_path = join(bn_file_path, bn_file_name)
    if exists(bn_file_path):
        load_bn_stats_file(net, bn_file_path)
    if not all(task in net.bn_stats for task in globals.TASKS):
        raise Exception('BN Stats not containing all tasks')
    log.info(f'Using {bn_file_path} batch norm running estimates checkpoint')
    tasks = ['initial'] + globals.TASKS
    results = ResultsManager()

    current_results = []
    log.info('::: DISC Plug & Play :::')
    for idx, args.task in enumerate(tasks):
        if not set_severity(args):
            continue
        severity_str = '' if args.task == 'initial' else f'Severity: {args.severity}'
        log.info(f'Start evaluation for Task-{idx} ({args.task}). {severity_str}')

        load_bn_stats(net, args.task, ckpt)
        test_loader = get_loader(args, split='test', pad=0.5, rect=True)
        if args.model == 'yolov3':
            res = test_yolo(model=net, dataloader=test_loader,
                            iou_thres=args.iou_thres, conf_thres=args.conf_thres,
                            augment=args.augment)[0] * 100
        else:
            res = test(test_loader, net)[0] * 100
        current_results.append(res)
        log.info(f'    {metric} on Task-{idx} ({tasks[idx]}): {res:.1f}')


        mean_result = mean(current_results)
        log.info(f'Mean {metric.lower()} over current task ({tasks[idx]}) '
                    f'and previously seen tasks: {mean_result:.1f}')
        severity_str = '' if args.task == 'initial' else f'{args.severity}'
        results.add_result('DISC', f'{tasks[idx]} {severity_str}', mean_result, 'online')
        results.add_result('DISC', f'{tasks[idx]} {severity_str}', mean_result, 'offline')

