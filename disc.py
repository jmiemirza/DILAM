from torch import load
from utils.data_loader import prepare_test_data
from utils.testing import test
from statistics import mean


def disc(args, net, severity, common_corruptions):
    ckpt = load(args.ckpt_path)
    bn_file_name = 'BN-' + net.__class__.__name__ + '-' + args.dataset + '.pt'
    load_bn_stats_file(net, bn_file_name)
    common_corruptions = ['initial'] + common_corruptions

    print('::: DISC Plug & Play :::')
    for args.level in severity:
        all_errors = []
        for idx, args.corruption in enumerate(common_corruptions):
            task_errors = []
            load_bn_stats(net, args.corruption, ckpt)
            test_loader = prepare_test_data(args)[1]
            err_cls = test(test_loader, net)[0] * 100
            print(f'Error on Task-{idx} ({args.corruption}): {err_cls:.1f}')
            all_errors.append(err_cls)
            task_errors.append(err_cls)

            # previously seen tasks
            if idx > 0:
                print('\tPreviously seen tasks:')
                for i in range(0, idx):
                    load_bn_stats(net, common_corruptions[i], ckpt)
                    args.corruption = common_corruptions[i]
                    test_loader = prepare_test_data(args)[1]
                    prev_err = test(test_loader, net)[0] * 100
                    print(f'\tError on Task-{i} ({common_corruptions[i]}): '
                          f'{prev_err:.1f}')
                    task_errors.append(prev_err)

                    if i == idx - 1:
                        print(f'\tMean error over current task '
                              f'({common_corruptions[idx]}) and previously '
                              f'seen tasks: {mean(task_errors):.1f}')

            assert mean(task_errors) == mean(all_errors)


def load_bn_stats(net, corruption, ckpt=None):
    """
        Optionally loads a model checkpoint.
        Loads the running estimates for all batch norm layers
            from net.bn_stats attribute, for a given corruption.
            Unless the corruption is 'initial', in which case the function
            returns without loading anything.
        Sets network to evaluation mode.
    """
    if ckpt is not None:
        net.load_state_dict(ckpt)
    if corruption == 'initial':
        return
    state_dict = net.state_dict()
    for layer_name, value in net.bn_stats[corruption].items():
        state_dict[layer_name + '.running_mean'] = value['running_mean']
        state_dict[layer_name + '.running_var'] = value['running_var']
    net.load_state_dict(state_dict)
    net.eval()


def load_bn_stats_file(net, file_path):
    """
        Loads a saved state of net.bn_stats attribute, located at file_path.
    """
    net.bn_stats = load(file_path)

