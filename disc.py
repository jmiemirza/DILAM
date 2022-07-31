from statistics import mean
import logging
from os.path import exists
from torch import load
from utils.data_loader import prepare_test_data
from utils.testing import test
from dua import dua
from utils.results_manager import ResultsManager
from globals import TASKS, SEVERTITIES

log = logging.getLogger('MAIN.DISC')


def disc_adaption(args, net, save_bn_stats=False, use_training_data=True):
    dua(args, net, save_bn_stats, use_training_data)


def disc_plug_and_play(args, net, bn_file_name=None):
    ckpt = load(args.ckpt_path)

    bn_file_path = 'checkpoints/' + args.dataset + '/' + net.__class__.__name__ + '/'
    if not bn_file_name:
        bn_file_name = 'BN_stats.pt'
    bn_file_path += bn_file_name
    if exists(bn_file_path):
        load_bn_stats_file(net, bn_file_path)
    elif not net.bn_stats.get(TASKS[0]):
        raise Exception('Could not find BN stats')

    tasks = ['initial'] + TASKS
    results = ResultsManager()

    log.info('::: DISC Plug & Play :::')
    for args.level in SEVERTITIES:
        log.info(f'Corruption level of severity: {args.level}')
        for idx, args.task in enumerate(tasks):
            log.info(f'Start evaluation for Task-{idx} ({args.task})')
            current_errors = []

            for i in range(0, idx + 1):
                args.task = tasks[i]
                load_bn_stats(net, args.task, ckpt)
                test_loader = prepare_test_data(args)[1]
                err_cls = test(test_loader, net)[0] * 100
                current_errors.append(err_cls)
                log.info(f'\tError on Task-{i} ({tasks[i]}): {err_cls:.2f}')

                if i == idx:
                    mean_error = mean(current_errors)
                    log.info(f'Mean error over current task ({tasks[i]}) '
                             f'and previously seen tasks: {mean_error:.2f}')
                    results.add_result('DISC', tasks[i], mean_error, 'online')
                    results.add_result('DISC', tasks[i], mean_error, 'offline')


def load_bn_stats(net, task, ckpt=None):
    """
        Optionally loads a model checkpoint.
        Loads the running estimates for all batch norm layers
            from net.bn_stats attribute, for a given task.
            Unless the task is 'initial', in which case the function
            returns without loading anything.
        Sets network to evaluation mode.
    """
    if ckpt is not None:
        net.load_state_dict(ckpt)
    if task == 'initial':
        return
    state_dict = net.state_dict()
    for layer_name, value in net.bn_stats[task].items():
        state_dict[layer_name + '.running_mean'] = value['running_mean']
        state_dict[layer_name + '.running_var'] = value['running_var']
    net.load_state_dict(state_dict)
    net.eval()


def load_bn_stats_file(net, file_path):
    """
        Loads a saved state of net.bn_stats attribute, located at file_path.
    """
    net.bn_stats = load(file_path)

