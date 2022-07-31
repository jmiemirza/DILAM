from statistics import mean
from os.path import exists
import logging
from torch import load
from utils.training import train
from utils.data_loader import prepare_test_data
from utils.testing import test
from utils.results_manager import ResultsManager
from globals import TASKS, SEVERTITIES

log = logging.getLogger('BASELINE.DISJOINT')


def disjoint(net, args, scenario='online'):
    """
        Evaluate Disjoint baseline.
    """
    if scenario == 'online':
        args.epochs = 1
    elif scenario == 'offline':
        args.epochs = 150
    ckpt_folder = 'checkpoints/' + args.dataset + '/' + net.__class__.__name__
    ckpt_folder += '/disjoint/' + scenario + '/'
    tasks = ['initial'] + TASKS
    results = ResultsManager()

    log.info(f'::: Baseline Disjoint ({scenario}) :::')
    for args.level in SEVERTITIES:
        log.info(f'Corruption level of severity: {args.level}')
        for idx, args.task in enumerate(tasks):
            log.info(f'Start evaluation for Task-{idx} ({args.task})')
            current_errors = []
            setup_net(net, args, ckpt_folder, idx)

            for i in range(0, idx + 1):
                args.task = tasks[i]
                test_loader = prepare_test_data(args)[1]
                err_cls = test(test_loader, net)[0] * 100
                current_errors.append(err_cls)
                log.info(f'\tError on Task-{i} ({tasks[i]}): {err_cls:.2f}')

                if i == idx:
                    mean_error = mean(current_errors)
                    log.info(f'Mean error over current task ({tasks[idx]}) '
                             f'and previously seen tasks: {mean_error:.2f}')
                    results.add_result('Disjoint', tasks[idx], mean_error, scenario)


def setup_net(net, args, ckpt_folder, idx):
    if args.task == 'initial':
        net.load_state_dict(load(args.ckpt_path))
    else:
        ckpt_path = ckpt_folder + args.task + '.pt'
        if not exists(ckpt_path):
            log.info(f'No checkpoint for Disjoint Task-{idx} '
                     f'({args.task}) - Starting training.')
            net.load_state_dict(load(args.ckpt_path))
            train(net, args, results_folder_path=ckpt_folder)
        else:
            net.load_state_dict(load(ckpt_path))
    net.eval()
