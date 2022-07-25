from statistics import mean
from os.path import exists
import logging
from torch import load
from utils.training import train_joint
from utils.data_loader import prepare_test_data
from utils.testing import test
from globals import TASKS, SEVERTITIES

log = logging.getLogger('BASELINE.JOINT_TRAINING')


def joint_training(net, args, scenario='online'):
    """
        Evaluate Joint-Training baseline.
    """
    if scenario == 'online':
        args.epochs = 1
    elif scenario == 'offline':
        args.epochs = 150
    ckpt_folder = 'checkpoints/' + args.dataset + '/joint_training/' + scenario
    ckpt_folder += '/' + net.__class__.__name__ + '/'

    log.info(f'::: Baseline Joint-Training ({scenario}) :::')
    for level in SEVERTITIES:
        log.info(f'Corruption level of severity: {level}')
        all_errors = []
        net.load_state_dict(load(args.ckpt_path))
        net.eval()
        test_loader = prepare_test_data(args)[1]
        err_cls = test(test_loader, net)[0] * 100
        all_errors.append(err_cls)
        log.info(f'Error on initial task: {err_cls:.2f}')
        for idx, args.task in enumerate(TASKS):
            ckpt_path = ckpt_folder + args.task + '.pt'
            if not exists(ckpt_path):
                log.info(f'No checkpoint for Joint-Training Task-{idx + 1} '
                         f'({args.task}) - Starting training.')
                train_joint(net, args, results_folder_path=ckpt_folder)
            else:
                net.load_state_dict(load(ckpt_path))
            net.eval()
            test_loader = prepare_test_data(args)[1]
            err_cls = test(test_loader, net)[0] * 100
            all_errors.append(err_cls)
            log.info(f'Error on Task-{idx + 1} ({args.task}): {err_cls:.1f}')
            log.info(f'Mean error over current task ({args.task}) '
                     f'and previously seen tasks: {mean(all_errors):.2f}')