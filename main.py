from dua import dua
from disc import disc
import argparse
from argparse import Namespace
import time
import baselines
from torch import nn
from models.wide_resnet import WideResNet
from models.resnet_26 import ResNetCifar
import torch.backends.cudnn as cudnn
from utils.data_loader import dataset_checks


def main():
    start_time = time.time()
    cudnn.benchmark = True
    severity = [5]
    common_corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]
    dataset_checks(args, common_corruptions)

    net = init_net(args)

    # args.num_samples = 5
    # common_corruptions = common_corruptions[:2]


    # disc adaption phase
    # dua(args, net, severity, common_corruptions, save_bn_stats=True)
    # dua(args, net, severity, common_corruptions)

    # disc plug and play
    disc(args, net, severity, common_corruptions)


    # Baselines
    # baselines.source_only(net, severity, common_corruptions, args)

    # baselines.disjoint(net, severity, common_corruptions, args, 'online')
    # baselines.disjoint(net, severity, common_corruptions, args, 'offline')

    # baselines.freezing(net, severity, common_corruptions, args, 'online')
    # baselines.freezing(net, severity, common_corruptions, args, 'offline')


    runtime = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    print(f'Done! Execution time: {runtime}')


    # TODO
    #   reorganize data_loader
    #   corruption -> task
    #   globals
    #   logger
    #   validation set deepcpy needed?
    #   bn file path
    #   check if model ckpt exists, train if not
    #   error handling




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
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--patience', default=4, type=int)
    parser.add_argument('--lr_factor', default=1/3, type=float)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--train_val_split', default=0.2, type=float)
    parser.add_argument('--train_val_split_seed', default=42, type=int)
    parser.add_argument('--max_unsuccessful_reductions', default=3, type=int)

    args: Namespace = parser.parse_args()
    main()
