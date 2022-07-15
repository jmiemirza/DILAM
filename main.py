from dua import *
from disc import *
import argparse
from argparse import Namespace


def main():
    cudnn.benchmark = True
    severity = [5]
    common_corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
        'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    net = init_net(args)

    # disc adaption phase
    dua(args, net, severity, common_corruptions, True)

    # disc plug and play
    disc(args, net, severity, common_corruptions)

    ckpt = torch.load(args.ckpt_path)
    baseline_src_only(net, severity, common_corruptions, args, ckpt)




def init_net(args):
    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(args.group_norm, planes)

    if args.model == 'wrn':
        net = WideResNet(widen_factor=2, depth=40, num_classes=10)
    elif args.model == 'res':
        net = ResNetCifar(args.depth, args.width, channels=3, classes=10, norm_layer=norm_layer).cuda()
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

    args: Namespace = parser.parse_args()
    main()
