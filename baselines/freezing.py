from statistics import mean
from os.path import exists
import logging
from torch import load, save
from utils.training import train
from utils.data_loader import prepare_test_data
from utils.testing import test
from utils.results_manager import ResultsManager
from globals import TASKS, SEVERTITIES

log = logging.getLogger('BASELINE.FREEZING')


def freezing(net, args, scenario='online'):
    """
        Evaluate Freezing baseline.
    """
    if scenario == 'online':
        args.epochs = 1
    elif scenario == 'offline':
        args.epochs = 150
    ckpt_folder = 'checkpoints/' + args.dataset + '/' + net.__class__.__name__
    ckpt_folder += '/freezing/' + scenario + '/'
    results = ResultsManager()

    tasks = ['initial'] + TASKS
    net.load_state_dict(load(args.ckpt_path))
    save_initial_task_head(net, ckpt_folder)

    log.info(f'::: Baseline Freezing ({scenario}) :::')
    for args.level in SEVERTITIES:
        log.info(f'Corruption level of severity: {args.level}')
        for idx, args.task in enumerate(tasks):
            log.info(f'Start evaluation for Task-{idx} ({args.task})')
            current_errors = []

            for i in range(0, idx +1):
                args.task = tasks[i]
                setup_net(net, args, ckpt_folder, idx)
                test_loader = prepare_test_data(args)[1]
                err_cls = test(test_loader, net)[0] * 100
                current_errors.append(err_cls)
                log.info(f'\tError on Task-{i} ({tasks[i]}): {err_cls:.2f}')

                if i == idx:
                    mean_error = mean(current_errors)
                    log.info(f'\tMean error over current task ({tasks[idx]}) '
                             f'and previously seen tasks: {mean_error:.2f}')
                    results.add_result('Freezing', tasks[idx], mean_error, scenario)


def setup_net(net, args, ckpt_folder, idx):
    ckpt_path = ckpt_folder + args.task + '.pt'
    if not exists(ckpt_path):
        log.info(f'No checkpoint for Freezing Task-{idx} '
                 f'({args.task}) - Starting training.')
        net.load_state_dict(load(args.ckpt_path))
        train(net, args, results_folder_path=ckpt_folder,
              lr=args.initial_task_lr, train_heads_only=True)
    else:
        # TODO multi layer heads
        net.get_heads().load_state_dict(load(ckpt_path))
    net.eval()


def save_initial_task_head(net, ckpt_folder):
    ckpt_path = ckpt_folder + 'initial.pt'
    save(net.get_heads().state_dict(), ckpt_path)
    net.eval()
