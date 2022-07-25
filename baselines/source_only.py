from statistics import mean
import logging
from torch import load
from utils.data_loader import prepare_test_data
from utils.testing import test
from globals import TASKS, SEVERTITIES

log = logging.getLogger('BASELINE.SOURCE_ONLY')


def source_only(net, args):
    """
        Evaluate Source-Only baseline.
    """
    net.load_state_dict(load(args.ckpt_path))
    net.eval()
    scenario = 'provided checkpoint'

    tasks = ['initial'] + TASKS

    log.info(f'::: Baseline Source-Only ({scenario}) :::')
    for level in SEVERTITIES:
        log.info(f'Corruption level of severity: {level}')
        all_errors = []
        for idx, args.task in enumerate(tasks):
            test_loader = prepare_test_data(args)[1]
            err_cls = test(test_loader, net)[0] * 100
            all_errors.append(err_cls)
            log.info(f'Error on Task-{idx} ({args.task}): {err_cls:.1f}')
            log.info(f'Mean error over current task ({args.task}) '
                     f'and previously seen tasks: {mean(all_errors):.2f}')

