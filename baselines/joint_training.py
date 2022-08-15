from statistics import mean
from os.path import exists
import logging
from torch import load
from utils.training import train_joint
from utils.data_loader import get_test_loader
from utils.testing import test
from utils.results_manager import ResultsManager
from globals import TASKS, SEVERTITIES

log = logging.getLogger('BASELINE.JOINT_TRAINING')


# TODO joint dataset
def joint_training(net, args, scenario='online'):
    """
        Evaluate Joint-Training baseline.
    """
    if scenario == 'online':
        args.epochs = 1
    elif scenario == 'offline':
        args.epochs = 150
    ckpt_folder = 'checkpoints/' + args.dataset + '/' + args.model
    ckpt_folder += '/joint_training/' + scenario + '/'
    results = ResultsManager()

    log.info(f'::: Baseline Joint-Training ({scenario}) :::')
    tasks = ['initial'] + TASKS
    for args.level in SEVERTITIES:
        log.info(f'Corruption level of severity: {args.level}')
        for idx, args.task in enumerate(tasks):
            log.info(f'Start evaluation for Task-{idx} ({args.task})')
            current_errors = []
            setup_net(net, args, ckpt_folder, idx)

            for i in range(0, idx + 1):
                args.task = tasks[i]
                test_loader = get_test_loader(args)
                err_cls = test(test_loader, net)[0] * 100
                current_errors.append(err_cls)
                log.info(f'\tError on Task-{i} ({tasks[i]}): {err_cls:.2f}')

                if i == idx:
                    mean_error = mean(current_errors)
                    log.info(f'\tMean error over current task ({tasks[i]}) '
                             f'and previously seen tasks: {mean_error:.2f}')
                    results.add_result('Joint-Training', tasks[i], mean_error, scenario)


def setup_net(net, args, ckpt_folder, idx):
    net.load_state_dict(load(args.ckpt_path))
    if args.task != 'initial':
        ckpt_path = ckpt_folder + args.task + '.pt'
        if not exists(ckpt_path):
            log.info(f'No checkpoint for Joint-Training Task-{idx} '
                     f'({args.task}) - Starting training.')
            tasks = ['initial'] + TASKS[:idx]
            log.debug(f'Training model on: {tasks}')
            train_joint(net, args, results_folder_path=ckpt_folder, tasks=tasks)
        else:
            net.load_state_dict(load(ckpt_path))
    net.eval()

