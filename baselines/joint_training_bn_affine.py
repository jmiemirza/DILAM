import logging
from os.path import exists, join
from statistics import mean

from torch import load

import globals, config
from utils.data_loader import get_loader, set_severity, set_yolo_save_dir
from utils.results_manager import ResultsManager
from utils.testing import test
from utils.testing_yolov3 import test as test_yolo
from utils.torch_utils import select_device
from utils.training import train
from utils.training_yolov3 import train as train_yolo

log = logging.getLogger('BASELINE.JOINT_BN_AFFINE')


def joint_training_bn_affine(net, args, scenario='online'):
    """
        Evaluate Joint-Training-BN-Affine baseline.
    """
    tmp_epochs = args.epochs
    if scenario == 'online':
        args.epochs = 1

    metric = 'mAP@50'
    config.YOLO_HYP['lr0'] = args.initial_task_lr

    ckpt_folder = join(args.checkpoints_path, args.dataset, args.model, 'joint_training_bn_affine', scenario)
    results = ResultsManager()
    log.info(f'::: Baseline Joint-Training-BN-Affine ({scenario}) :::')
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
                log.info(f'\tMean {metric.lower()} over current task ({tasks[i]}) '
                            f'and previously seen tasks: {mean_result:.1f}')
                severity_str = '' if args.task == 'initial' else f'{args.severity}'
                results.add_result('Joint-Training-BN-Affine', f'{tasks[i]} {severity_str}', mean_result, scenario)
                results.add_result_map50to95('Joint-Training-BN-Affine', f'{tasks[i]} {severity_str}', mean_result_map50to95, scenario)
    args.epochs = tmp_epochs


def setup_net(net, args, ckpt_folder, idx, scenario):
    net.load_state_dict(load(args.ckpt_path))
    if args.task != 'initial':
        ckpt_path = join(ckpt_folder, f'{args.task}_{args.severity}.pt')
        hyp = args.yolo_hyp()
        device = select_device(args.device, batch_size=args.batch_size)
        set_yolo_save_dir(args, 'joint_training_bn_affine', scenario)

        # select all layers that aren't batchnorm to be frozen
        freeze = [n for n, _ in list(net.named_parameters()) if 'bn' not in n]
        train_yolo(hyp, args, device, model=net, joint=True, freeze=freeze,
                    freeze_all_bn_running_estimates=True)
    net.eval()

