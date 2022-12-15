import logging
from os.path import exists, join
from statistics import mean

from torch import load, save

import globals
import config
from utils.data_loader import get_loader, set_severity, set_yolo_save_dir
from utils.results_manager import ResultsManager
from utils.testing import test
from utils.testing_yolov3 import test as test_yolo
from utils.torch_utils import select_device
from utils.training import train
from utils.training_yolov3 import train as train_yolo
from utils.utils import make_dirs

log = logging.getLogger('BASELINE.FREEZING')


def freezing(net, args, scenario='online'):
    """
        Evaluate Freezing baseline.
    """
    tmp_epochs = args.epochs
    if scenario == 'online':
        args.epochs = 1

    metric = 'Error'
    if args.model == 'yolov3':
        metric = 'mAP@50'
        config.YOLO_HYP['lr0'] = args.initial_task_lr

    ckpt_folder = join(args.checkpoints_path, args.dataset, args.model, 'freezing', scenario)

    if not args.use_freezing_heads_ckpts:
        move_existing_head_checkpoints(ckpt_folder)

    results = ResultsManager()

    tasks = ['initial'] + globals.TASKS
    net.load_state_dict(load(args.ckpt_path))
    save_initial_task_heads(args, net, ckpt_folder)

    log.info(f'::: Baseline Freezing ({scenario}) :::')
    for idx, args.task in enumerate(tasks):
        if not set_severity(args):
            continue
        severity_str = '' if args.task == 'initial' else f'Severity: {args.severity}'
        log.info(f'Start evaluation for Task-{idx} ({args.task}). {severity_str}')
        current_results = []

        for i in range(0, idx +1):
            args.task = tasks[i]
            if not set_severity(args):
                continue
            setup_net(net, args, ckpt_folder, idx, scenario)
            test_loader = get_loader(args, split='test', pad=0.5, rect=True)
            if args.model == 'yolov3':
                res = test_yolo(model=net, dataloader=test_loader)[0] * 100
            else:
                res = test(test_loader, net)[0] * 100
            current_results.append(res)
            log.info(f'\t{metric} on Task-{i} ({tasks[i]}): {res:.1f}')

            if i == idx:
                mean_result = mean(current_results)
                log.info(f'\tMean {metric.lower()} over current task ({tasks[idx]}) '
                            f'and previously seen tasks: {mean_result:.1f}')
                severity_str = '' if args.task == 'initial' else f'{args.severity}'
                results.add_result('Freezing', f'{tasks[idx]} {severity_str}', mean_result, scenario)

    args.epochs = tmp_epochs
    config.YOLO_HYP['lr0'] = args.lr


def setup_net(net, args, ckpt_folder, idx, scenario):
    if args.task == 'initial':
        net.load_state_dict(load(args.ckpt_path))
    else:
        # TODO remove initial_dectect ckpt loading and saving
        severity_str = '' if args.task == 'initial' else f'_{args.severity}'
        ckpt_path = join(ckpt_folder, f'{args.task}{severity_str}_detect.pt')
        if not exists(ckpt_path):
            if args.model == 'yolov3':
                hyp = args.yolo_hyp()
                device = select_device(args.device, batch_size=args.batch_size)
                net.load_state_dict(load(args.ckpt_path))
                # detect layers contain '28' in their full parameter name therefore
                # we are selecting all layers that don't contain '28' to be frozen
                freeze = [n for n, _ in list(net.named_parameters()) if '28' not in n]
                set_yolo_save_dir(args, 'freezing', scenario)
                train_yolo(hyp, args, device, model=net, freeze=freeze, heads_save_path=ckpt_path)
            else:
                log.info(f'No checkpoint for Freezing Task-{idx} '
                        f'({args.task}) - Starting training.')
                net.load_state_dict(load(args.ckpt_path))
                train(net, args, result_path=ckpt_path, lr=args.initial_task_lr,
                    train_heads_only=True)
        else:
            if args.model == 'yolov3':
                ckpt = load(ckpt_path)
                for k, v in list(net.named_parameters()):
                    if '28' in k:
                        v.data = ckpt[k]
            else:
                net.get_heads().load_state_dict(load(ckpt_path))
    net.eval()


def save_initial_task_heads(args, net, ckpt_folder):
    make_dirs(ckpt_folder)
    ckpt_path = join(ckpt_folder, 'initial_detect.pt')
    if args.model == 'yolov3':
        ckpt = {}
        for k, v in net.named_parameters():
            if '28' in k:
                ckpt[k] = v
        save(ckpt, ckpt_path)
    else:
        save(net.get_heads().state_dict(), ckpt_path)
    net.eval()


def move_existing_head_checkpoints(ckpt_folder):
    import glob
    import shutil
    from pathlib import Path
    import os
    from utils.general import increment_path

    files = glob.glob(join(ckpt_folder, '*.pt'))
    dst = increment_path(Path( join(ckpt_folder, 'prev_run_heads')))
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.move(f, dst)

