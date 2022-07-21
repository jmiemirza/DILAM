from statistics import mean
from os.path import exists
from torch import load
from utils.training import train
from utils.data_loader import prepare_test_data
from utils.testing import test


def freezing(net, severity, corruptions, args, scenario='online'):
    """
        Evaluate Freezing baseline.
    """
    if scenario == 'online':
        args.epochs = 1
    elif scenario == 'offline':
        args.epochs = 150
    ckpt_folder = 'checkpoints/' + args.dataset + '/freezing/' + scenario
    ckpt_folder += '/' + net.__class__.__name__ + '/'

    print(f'::: Baseline Freezing ({scenario}) :::')
    for level in severity:
        print(f'Corruption level of severity: {level}')
        all_errors = []
        net.load_state_dict(load(args.ckpt_path))
        net.eval()
        test_loader = prepare_test_data(args)[1]
        err_cls = test(test_loader, net)[0] * 100
        all_errors.append(err_cls)
        print(f'Error on initial task: {err_cls:.2f}')
        for idx, args.corruption in enumerate(corruptions):
            ckpt_path = ckpt_folder + args.corruption + '.pt'
            if not exists(ckpt_path):
                print(f'No checkpoint for Freezing Task-{idx + 1} '
                      f'({args.corruption}) - Starting training.')
                train(net, args, results_folder_path=ckpt_folder,
                      lr=args.initial_task_lr, train_heads_only=True)
            else:
                net.load_state_dict(load(ckpt_path))
            net.eval()
            test_loader = prepare_test_data(args)[1]
            err_cls = test(test_loader, net)[0] * 100
            all_errors.append(err_cls)
            print(f'Error on Task-{idx + 1} ({args.corruption}): {err_cls:.1f}')
            print(f'Mean error over current task ({args.corruption}) '
                  f'and previously seen tasks: {mean(all_errors):.2f}')

