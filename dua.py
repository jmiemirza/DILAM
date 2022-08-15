import logging
from utils.data_loader import *
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from utils.utils import make_dirs
from utils.testing import test
from utils.rotation import *
from globals import TASKS, SEVERTITIES

log = logging.getLogger('MAIN.DUA')


def dua(args, net, save_bn_stats=False, use_training_data=False):
    tr_transform_adapt = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*NORM)
    ])

    ckpt = torch.load(args.ckpt_path)

    decay_factor = args.decay_factor
    min_momentum_constant = args.min_mom

    no_imp = 0
    no_imp_cnt = 0

    for args.level in SEVERTITIES:
        log.info(f'Starting DUA for Level {args.level}')
        all_errors = []
        for args.task in TASKS:
            mom_pre = 0.1
            err = []
            log.info(f'Task - {args.task} :::: Level - {args.level}')
            net.load_state_dict(ckpt)

            if use_training_data:
                train_loader = get_train_loader(args)
                valid_loader = get_val_loader(args)
            else:
                train_loader = valid_loader = get_test_loader(args)

            err_cls = test(valid_loader, net)[0] * 100
            log.info(f'Error Before Adaptation: {err_cls:.1f}')

            for i in tqdm(range(1, args.num_samples + 1)):
                net.eval()
                image = train_loader.dataset.get_pil_image_from_idx(i - 1)
                mom_new = (mom_pre * decay_factor)
                for m in net.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.train()
                        m.momentum = mom_new + min_momentum_constant
                mom_pre = mom_new
                inputs = [(tr_transform_adapt(image)) for _ in range(64)]
                inputs = torch.stack(inputs)
                inputs = inputs.cuda()
                inputs_ssh, _ = rotate_batch(inputs, 'rand')
                inputs_ssh = inputs_ssh.cuda()
                _ = net(inputs_ssh)
                err_cls = test(valid_loader, net)[0] * 100

                err.append(err_cls)
                if err_cls <= min(err):
                    save_bn_stats_in_model(net, args.task)
                    no_imp = 0
                    no_imp_cnt = 0
                else:
                    no_imp += 1
                    if no_imp >= 10:
                        no_imp_cnt += no_imp
                        no_imp = 0
                        log.info(f'IT {i}/{args.num_samples}: NO Improvement '
                                 f'for {no_imp_cnt} consecutive iterations')

            adaptation_error = min(err)
            log.info(f'Error After Adaptation: {adaptation_error:.1f}')
            all_errors.append(adaptation_error)
        log.info(f'Mean Error after Adaptation {(sum(all_errors) / len(all_errors)):.1f}')

    if save_bn_stats:
        save_bn_stats_to_file(net, args.dataset, args.model)


def save_bn_stats_in_model(net, task):
    """
        Saves the running estimates of all batch norm layers for a given
        task, in the net.bn_stats attribute.
    """
    state_dict = net.state_dict()
    net.bn_stats[task] = {}
    for layer_name, m in net.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            net.bn_stats[task][layer_name] = {
                'running_mean': state_dict[layer_name + '.running_mean'].detach().clone(),
                'running_var': state_dict[layer_name + '.running_var'].detach().clone()
            }


def save_bn_stats_to_file(net, dataset_str=None, model_str=None, file_name=None):
    """
        Saves net.bn_stats content to a file.
    """
    ckpt_folder = 'checkpoints/' + dataset_str + '/' + model_str + '/'
    make_dirs(ckpt_folder)
    if not file_name:
        file_name = 'BN_stats.pt'
    torch.save(net.bn_stats, ckpt_folder + file_name)