import logging
from statistics import mean

from torch import load

import globals
from utils.data_loader import get_loader, set_severity
from utils.results_manager import ResultsManager
from utils.testing import test
from utils.testing_yolov3 import test as test_yolo

log = logging.getLogger('BASELINE.SOURCE_ONLY')


def source_only(net, args):
    """
        Evaluate Source-Only baseline.
    """
    metric = 'mAP@50'
    net.load_state_dict(load(args.ckpt_path))
    net.eval()
    results = ResultsManager()
    tasks = ['initial'] + globals.TASKS
    log.info('::: Baseline Source-Only :::')
    all_results = []
    current_results_map50to95 = []
    for idx, args.task in enumerate(tasks):
        if not set_severity(args):
            continue
        severity_str = '' if args.task == 'initial' else f'Severity: {args.severity}'
        log.info(f'Start evaluation for Task-{idx} ({args.task}). {severity_str}')

        test_loader = get_loader(args, split='test', pad=0.5, rect=True)
        # test_loader = get_loader(args, split='test', pad=0.5, rect=False, shuffle=False) # TODO

        test_res = test_yolo(model=net, dataloader=test_loader,
                            iou_thres=args.iou_thres, conf_thres=args.conf_thres,
                            augment=args.augment)#[0] * 100
        res = test_res[0] * 100
        res_map50to95 = test_res[1][3] * 100
        all_results.append(res)
        current_results_map50to95.append(res_map50to95)
        log.info(f'\t{metric} on Task-{idx} ({args.task}): {res:.1f}')
        log.info(f'\tMean {metric.lower()} over current task ({args.task}) '
                 f'and previously seen tasks: {mean(all_results):.1f}')
        severity_str = '' if args.task == 'initial' else f'{args.severity}'
        results.add_result('Source-Only', f'{args.task} {severity_str}', mean(all_results), 'online')
        results.add_result('Source-Only', f'{args.task} {severity_str}', mean(all_results), 'offline')

        results.add_result_map50to95('Source-Only', f'{args.task} {severity_str}', mean(current_results_map50to95), 'online')
        results.add_result_map50to95('Source-Only', f'{args.task} {severity_str}', mean(current_results_map50to95), 'offline')

