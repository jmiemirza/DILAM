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
    metric = 'Error'
    if args.model == 'yolov3':
        metric = 'mAP@50'
    net.load_state_dict(load(args.ckpt_path))
    net.eval()
    results = ResultsManager()
    tasks = ['initial'] + globals.TASKS
    log.info('::: Baseline Source-Only :::')
    all_results = []
    for idx, args.task in enumerate(tasks):
        if not set_severity(args):
            continue
        severity_str = '' if args.task == 'initial' else f'Severity: {args.severity}'
        log.info(f'Start evaluation for Task-{idx} ({args.task}). {severity_str}')
        test_loader = get_loader(args, split='test', pad=0.5, rect=True)

        if args.model == 'yolov3':
            res = test_yolo(model=net, dataloader=test_loader)[0] * 100
        else:
            res = test(test_loader, net)[0] * 100

        all_results.append(res)
        log.info(f'\t{metric} on Task-{idx} ({args.task}): {res:.1f}')
        log.info(f'\tMean {metric.lower()} over current task ({args.task}) '
                 f'and previously seen tasks: {mean(all_results):.1f}')
        severity_str = '' if args.task == 'initial' else f'{args.severity}'
        results.add_result('Source-Only', f'{args.task} {severity_str}', mean(all_results), 'online')
        results.add_result('Source-Only', f'{args.task} {severity_str}', mean(all_results), 'offline')

