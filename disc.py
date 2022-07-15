from dua import *


def disc(args, net, severity, common_corruptions):
    ckpt = torch.load(args.ckpt_path)
    bn_file_name = 'BN-' + net.__class__.__name__ + '-' + args.dataset + '.pt'
    load_bn_stats_file(net, bn_file_name)

    for args.level in severity:
        print(f'Starting DISC for Level {args.level}')
        all_errors = []
        for idx, args.corruption in enumerate(common_corruptions):
            task_errors = []
            print(f'Corruption - {args.corruption} :::: Level - {args.level}')
            load_bn_stats(net, args.corruption, ckpt)
            teloader = prepare_test_data(args)[1]
            err_cls = test(teloader, net)[0] * 100
            print(f'DISC: Error: {err_cls:.1f}')
            all_errors.append(err_cls)
            task_errors.append(err_cls)

            # previously seen tasks
            if idx > 0:
                print(f'\tPreviously seen')
                for i in range(0, idx):
                    load_bn_stats(net, common_corruptions[i], ckpt)
                    args.corruption = common_corruptions[i]
                    teloader = prepare_test_data(args)[1]
                    prev_err = test(teloader, net)[0] * 100
                    print(f'\tError for {common_corruptions[i]}: {prev_err:.1f}')
                    task_errors.append(prev_err)
                    if i == idx - 1:
                        print(f'\tCurrent and prev seen mean error: {sum(task_errors) / len(task_errors):.1f}')

        print(f'DISC Mean Error {(sum(all_errors) / len(all_errors)):.1f}')


def baseline_src_only(net, severity, corruptions, args, ckpt=None):
    """
        Evaluate Source-Only baseline.
    """
    print('::: Baseline Source-Only :::')
    if ckpt is not None:
        net.load_state_dict(ckpt)
        net.eval()
    for level in severity:
        all_errors = []
        for corruption in corruptions:
            args.corruption = corruption
            teloader = prepare_test_data(args)[1]
            err_cls = test(teloader, net)[0] * 100
            print(f'Corruption - {corruption} - Level {level}, Source-Only Error: {err_cls:.1f}')
            all_errors.append(err_cls)
        print(f'Level {level}, Source-Only Mean Error: {(sum(all_errors) / len(all_errors)):.1f}')


def load_bn_stats(net, corruption, ckpt=None):
    """
        Optionally loads a model checkpoint.
        Loads the running estimates for all batch norm layers
            from net.bn_stats attribute, for a given corruption.
        Sets network to evaluation mode.
    """
    if ckpt is not None:
        net.load_state_dict(ckpt)
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
    net.bn_stats = torch.load(file_path)
