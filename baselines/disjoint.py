import logging
from os.path import exists, join
from statistics import mean

import torch

import globals
from models.yolo import Model
from utils.data_loader import get_loader, set_severity, set_yolo_save_dir
from utils.google_utils import attempt_download
from utils.results_manager import ResultsManager
from utils.testing import test
from utils.testing_yolov3 import test as test_yolo
from utils.torch_utils import (intersect_dicts, select_device,
                               torch_distributed_zero_first)
from utils.training import train
from utils.training_yolov3 import train as train_yolo

log = logging.getLogger('BASELINE.DISJOINT')


def disjoint(net, args, scenario='online'):
    """
        Evaluate Disjoint baseline.
    """
    tasks = ['initial'] + globals.TASKS
    results = ResultsManager()
    tmp_epochs = args.epochs
    if scenario == 'online':
        args.epochs = 1

    metric = 'Error'
    if args.model == 'yolov3':
        metric = 'mAP@50'

    ckpt_folder = join(args.checkpoints_path, args.dataset, args.model, 'disjoint', scenario)

    log.info(f'::: Baseline Disjoint ({scenario}) :::')
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
                log.info(f'\tMean {metric.lower()} over current task ({tasks[idx]}) '
                         f'and previously seen tasks: {mean_result:.1f}')
                severity_str = '' if args.task == 'initial' else f'{args.severity}'
                results.add_result('Disjoint', f'{tasks[idx]} {severity_str}', mean_result, scenario)
    args.epochs = tmp_epochs


def setup_net(net, args, ckpt_folder, idx, scenario):
    if args.task == 'initial':
        net.load_state_dict(torch.load(args.ckpt_path))
    else:
        if args.model == 'yolov3':
            device = select_device(args.device, batch_size=args.batch_size)

            if scenario == 'offline' and not args.start_disjoint_offline_from_initial:
                log.info('Loading yolov3.pt weights')
                hyp = args.yolo_hyp()
                with torch_distributed_zero_first(args.global_rank):
                    attempt_download('yolov3.pt')  # download if not found locally
                ckpt = torch.load('yolov3.pt', map_location=device)  # load checkpoint
                if hyp.get('anchors'):
                    ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
                model = Model(args.cfg or ckpt['model'].yaml, ch=3, nc=args.nc).to(device)  # create
                exclude = ['anchor'] if args.cfg or hyp.get('anchors') else []  # exclude keys
                state_dict = ckpt['model'].float().state_dict()  # to FP32
                state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
                net.load_state_dict(state_dict, strict=False)  # load
                log.info('Transferred %g/%g items from %s' % (len(state_dict), len(net.state_dict()), 'yolov3.pt'))  # report
            else:
                net.load_state_dict(torch.load(args.ckpt_path))

            set_yolo_save_dir(args, 'disjoint', scenario)
            train_yolo(args.yolo_hyp(), args, device, model=net)
        else:
            ckpt_path = join(ckpt_folder, f'{args.task}_{args.severity}.pt')
            log.info(f'ckptpath:  {ckpt_path}')
            if not exists(ckpt_path):
                log.info(f'No checkpoint for Disjoint Task-{idx} '
                        f'({args.task}) - Starting training.')
                net.load_state_dict(torch.load(args.ckpt_path))
                train(net, args, result_path=ckpt_path)
            else:
                net.load_state_dict(torch.load(ckpt_path))
    net.eval()

