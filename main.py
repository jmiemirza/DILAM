import argparse
from argparse import Namespace
import sys
import time
import logging
import logging.config
from os.path import exists
from torch import nn
import torch.backends.cudnn as cudnn
from dua import dua
from disc import disc
from models.wide_resnet import WideResNet
from models.resnet_26 import ResNetCifar
from utils.data_loader import dataset_checks
from utils.training import train
import baselines
from globals import LOGGER_CFG


logging.config.dictConfig(LOGGER_CFG)
log = logging.getLogger('MAIN')


def main():
    start_time = time.time()
    cudnn.benchmark = True

    log.debug('-- hi --')

    net = init_net(args)
    initial_checks(net, args)


    # disc adaption phase
    # args.num_samples = 5
    # args.num_samples = 320
    # args.batch_size = 16
    # dua(args, net, save_bn_stats=True, use_training_data=True)
    # dua(args, net, save_bn_stats=False, use_training_data=True)

    # disc plug and play
    disc(args, net)


    # Baselines
    # baselines.source_only(net, args)

    # baselines.disjoint(net, args, 'online')
    # baselines.disjoint(net, args, 'offline')

    # baselines.freezing(net, args, 'online')
    # baselines.freezing(net, args, 'offline')

    # baselines.fine_tuning(net, args, scenario='online')
    # baselines.fine_tuning(net, args, scenario='offline')

    # baselines.joint_training(net, args, scenario='online')
    # baselines.joint_training(net, args, scenario='offline')


    runtime = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    log.info(f'Exection finished in {runtime}')





def init_net(args):
    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(args.group_norm, planes)
        norm_layer = gn_helper

    def get_heads_classification():
        layers = [m for m in net.modules()]
        return layers[-1]

    if args.model == 'wrn':
        net = WideResNet(widen_factor=2, depth=40, num_classes=10)
        net.get_heads = get_heads_classification
    elif args.model == 'res':
        net = ResNetCifar(args.depth, args.width, channels=3, classes=10,
                          norm_layer=norm_layer)
        net.get_heads = get_heads_classification
    else:
        raise Exception(f'Invalid model argument: {args.model}')

    net = net.cuda()
    setattr(net.__class__, 'bn_stats', {})
    return net


def initial_checks(net, args):
    log.info('Running initial checks.')
    dataset_checks(args)

    if not exists(args.ckpt_path):
        log.info('Checkpoint trained on initial task not found - Starting training.')
        args.task = 'initial'
        args.ckpt_path = 'checkpoints/' + args.dataset
        train(net, args, args.ckpt_path)
        log.info(f'Checkpoint trained on initial task saved at {args.ckpt_path}/initial.pt')


# Log uncaught exceptions, that aren't keyboard interrupts
def handle_exception(exception_type, value, traceback):
    if issubclass(exception_type, KeyboardInterrupt):
        sys.__excepthook__(exception_type, value, traceback)
        return
    log.exception('Exception occured:', exc_info=(exception_type, value, traceback))
sys.excepthook = handle_exception


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--dataroot', default='X:/thesis/CIFAR-10-C')
    parser.add_argument('--ckpt_path', default='X:/thesis/Hendrycks2020AugMixWRN.pt')
    parser.add_argument('--model', default='wrn')
    parser.add_argument('--num_samples', default=80, type=int)
    parser.add_argument('--decay_factor', default=0.94, type=float)
    parser.add_argument('--min_mom', default=0.005, type=float)
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--level', default=5, type=int)


    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--initial_task_lr', default=1.5242e-06, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--patience', default=4, type=int)
    parser.add_argument('--lr_factor', default=1/3, type=float)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--train_val_split', default=0.2, type=float)
    parser.add_argument('--train_val_split_seed', default=42, type=int)
    parser.add_argument('--max_unsuccessful_reductions', default=3, type=int)

    args: Namespace = parser.parse_args()
    main()
