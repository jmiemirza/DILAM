import logging
from os.path import exists, join
from statistics import mean

from torch import load

import globals
from utils.data_loader import get_loader, set_severity, set_yolo_save_dir
from utils.results_manager import ResultsManager
from utils.testing import test
from utils.testing_yolov3 import test as test_yolo
from utils.torch_utils import select_device
from utils.training import train
from utils.training_yolov3 import train as train_yolo

log = logging.getLogger('BASELINE.JOINT_TRAINING')


def joint_training(net, args, scenario='online'):
    """
        Evaluate Joint-Training baseline.
    """
    tmp_epochs = args.epochs
    if scenario == 'online':
        args.epochs = 1

    metric = 'Error'
    if args.model == 'yolov3':
        metric = 'mAP@50'

    ckpt_folder = join(args.checkpoints_path, args.dataset, args.model, 'joint_training', scenario)
    results = ResultsManager()
    log.info(f'::: Baseline Joint-Training ({scenario}) :::')
    tasks = ['initial'] + globals.TASKS
    for idx, args.task in enumerate(tasks):
        current_results = []
        if not set_severity(args):
            continue
        severity_str = '' if args.task == 'initial' else f'Severity: {args.severity}'
        log.info(f'Start evaluation for Task-{idx} ({args.task}). {severity_str}')
        setup_net(net, args, ckpt_folder, idx, scenario)

        for i in range(0, idx + 1):
            args.task = tasks[i]
            if not set_severity(args):
                continue
            test_loader = get_loader(args, split='test', pad=0.5, rect=True)

            if args.model == 'yolov3':
                res = test_yolo(model=net, dataloader=test_loader,
                                iou_thres=args.iou_thres, conf_thres=args.conf_thres,
                                augment=args.augment)[0] * 100
            else:
                res = test(test_loader, net)[0] * 100

            current_results.append(res)
            log.info(f'\t{metric} on Task-{i} ({tasks[i]}): {res:.1f}')

            if i == idx:
                mean_result = mean(current_results)
                log.info(f'\tMean {metric.lower()} over current task ({tasks[i]}) '
                            f'and previously seen tasks: {mean_result:.1f}')
                severity_str = '' if args.task == 'initial' else f'{args.severity}'
                results.add_result('Joint-Training', f'{tasks[i]} {severity_str}', mean_result, scenario)
    args.epochs = tmp_epochs


def setup_net(net, args, ckpt_folder, idx, scenario):
    net.load_state_dict(load(args.ckpt_path))
    if args.task != 'initial':
        ckpt_path = join(ckpt_folder, f'{args.task}_{args.severity}.pt')
        if args.model == 'yolov3':
            hyp = args.yolo_hyp()
            device = select_device(args.device, batch_size=args.batch_size)
            set_yolo_save_dir(args, 'joint_training', scenario)
            train_yolo(hyp, args, device, model=net, joint=True)
        elif not exists(ckpt_path):
            log.info(f'No checkpoint for Joint-Training Task-{idx} '
                     f'({args.task}) - Starting training.')
            log.debug(f'Training model on: {["initial"] + globals.TASKS[:idx]}')
            train(net, args, result_path=ckpt_path, joint=True)
        else:
            net.load_state_dict(load(ckpt_path))
    net.eval()

