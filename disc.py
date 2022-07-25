from statistics import mean
import logging
from os.path import exists
from torch import load
from utils.data_loader import prepare_test_data
from utils.testing import test
from globals import TASKS, SEVERTITIES

log = logging.getLogger('MAIN.DISC')


def disc(args, net, bn_file_name=None):
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

    log.info('::: DISC Plug & Play :::')
    for args.level in SEVERTITIES:
        all_errors = []
        for idx, args.task in enumerate(tasks):
            task_errors = []
            load_bn_stats(net, args.task, ckpt)
            test_loader = prepare_test_data(args)[1]
            err_cls = test(test_loader, net)[0] * 100
            log.info(f'Error on Task-{idx} ({args.task}): {err_cls:.1f}')
            all_errors.append(err_cls)
            task_errors.append(err_cls)

            # previously seen tasks
            if idx > 0:
                log.info('\tPreviously seen tasks:')
                for i in range(0, idx):
                    load_bn_stats(net, tasks[i], ckpt)
                    args.task = tasks[i]
                    test_loader = prepare_test_data(args)[1]
                    prev_err = test(test_loader, net)[0] * 100
                    log.info(f'\tError on Task-{i} ({tasks[i]}): {prev_err:.1f}')
                    task_errors.append(prev_err)

                    if i == idx - 1:
                        log.info(f'\tMean error over current task '
                                 f'({tasks[idx]}) and previously '
                                 f'seen tasks: {mean(task_errors):.1f}')

            assert mean(task_errors) == mean(all_errors)


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

