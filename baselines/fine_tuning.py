import logging
from os.path import exists, join
from statistics import mean

from torch import load

import config
import globals
from utils.data_loader import get_loader, set_severity, set_yolo_save_dir
from utils.results_manager import ResultsManager
from utils.testing import test
from utils.testing_yolov3 import test as test_yolo
from utils.torch_utils import select_device
from utils.training import train
from utils.training_yolov3 import train as train_yolo

log = logging.getLogger('BASELINE.FINE_TUNING')


def fine_tuning(net, args, scenario='online'):
    """
        Evaluate Fine-Tuning baseline.
    """
    tmp_epochs = args.epochs
    if scenario == 'online':
        args.epochs = 1

    metric = 'mAP@50'
    config.YOLO_HYP['lr0'] = args.initial_task_lr

    ckpt_folder = join(args.checkpoints_path, args.dataset, args.model, 'fine_tuning', scenario)
    results = ResultsManager()

    log.info(f'::: Baseline Fine-Tuning ({scenario}) :::')
    tasks = ['initial'] + globals.TASKS
    for idx, args.task in enumerate(tasks):
        current_results = []
        current_results_map50to95 = []

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
            test_res = test_yolo(model=net, dataloader=test_loader,
                            iou_thres=args.iou_thres, conf_thres=args.conf_thres,
                            augment=args.augment)
            res = test_res[0] * 100
            res_map50to95 = test_res[1][3] * 100
            current_results.append(res)
            current_results_map50to95.append(res_map50to95)
            log.info(f'\t{metric} on Task-{i} ({tasks[i]}): {res:.1f}')

            if i == idx:
                mean_result = mean(current_results)
                mean_result_map50to95 = mean(current_results_map50to95)
                log.info(f'\tMean {metric.lower()} over current task ({tasks[idx]}) '
                         f'and previously seen tasks: {mean_result:.1f}')
                severity_str = '' if args.task == 'initial' else f'{args.severity}'
                results.add_result('Fine-Tuning', f'{tasks[idx]} {severity_str}', mean_result, scenario)
                results.add_result_map50to95('Fine-Tuning', f'{tasks[idx]} {severity_str}', mean_result_map50to95, scenario)

    args.epochs = tmp_epochs
    config.YOLO_HYP['lr0'] = args.lr


def setup_net(net, args, ckpt_folder, idx, scenario):
    if args.task == 'initial':
        net.load_state_dict(load(args.ckpt_path))
    else:
        device = select_device(args.device, batch_size=args.batch_size)
        set_yolo_save_dir(args, 'fine_tuning', scenario)
        train_yolo(args.yolo_hyp(), args, device, model=net)
    net.eval()

