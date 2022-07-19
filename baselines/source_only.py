import torch
from statistics import mean
from utils.data_loader import prepare_test_data
from utils.testing import test


def source_only(net, severity, corruptions, args):
    """
        Evaluate Source-Only baseline.
    """
    net.load_state_dict(torch.load(args.ckpt_path))
    net.eval()
    scenario = 'provided checkpoint'

    corruptions = ['initial'] + corruptions

    print(f'::: Baseline Source-Only ({scenario}) :::')
    for level in severity:
        print(f'Corruption level of severity: {level}')
        all_errors = []
        for idx, args.corruption in enumerate(corruptions):
            test_loader = prepare_test_data(args)[1]
            err_cls = test(test_loader, net)[0] * 100
            all_errors.append(err_cls)
            print(f'Error on Task-{idx} ({args.corruption}): {err_cls:.1f}')
            print(f'Mean error over current task ({args.corruption}) '
                  f'and previously seen tasks: {mean(all_errors):.2f}')

