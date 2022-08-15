import argparse
from argparse import Namespace
import sys
import time
import logging
import logging.config
from os.path import exists, join
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from disc import disc_plug_and_play, disc_adaption
from models.wide_resnet import WideResNet
from models.resnet_26 import ResNetCifar
from utils.data_loader import dataset_checks, get_test_loader, get_train_loader
from utils.training import train
import baselines
from utils.results_manager import ResultsManager
import globals

logging.config.dictConfig(globals.LOGGER_CFG)
log = logging.getLogger('MAIN')


def main():
    start_time = time.time()
    cudnn.benchmark = True

    log.debug('-- hi --')

    results = ResultsManager()

    # ------------------------------------------
    args.usr = 'sl'

    # args.num_samples = 10
    args.num_samples = 20
    # args.num_samples = 320

    # args.batch_size = 64
    # args.batch_size = 128

    args.initial_task_lr = 0.001

    # args.dataset = 'imagenet-mini'
    # args.model = 'res18'

    # args.dataset = 'imagenet'
    # args.model = 'res18'
    # ------------------------------------------

    if args.dataset == 'kitti':
        globals.TASKS = globals.KITTI_TASKS
    elif args.dataset in ['imagenet', 'imagenet-mini']:
        from utils.dataset_wrappers import ImgNet
        ImgNet.initial_dir = args.dataset

    if not hasattr(args, 'workers'):
        args.workers = 1

    if args.usr:
        set_paths()
    net = init_net(args)
    initial_checks(net, args)



    # disc_adaption(args, net, save_bn_stats=True, use_training_data=False)
    # disc_adaption(args, net, save_bn_stats=True, use_training_data=True)
    # disc_adaption(args, net, save_bn_stats=False, use_training_data=True)
    # disc_plug_and_play(args, net)

    # results.save_to_file(file_name='disc_results.pkl')
    # results.load_from_file(file_name='disc_results.pkl')


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


    # results.plot_summary()
    # results.print_summary()
    # results.print_summary_latex()
    # results.plot_scenario_summary('online')
    # results.save_to_file(file_name='123456_res.pkl')
    # results.save_to_file(file_name='CIFAR10_online_ALL.pkl')


    runtime = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    log.info(f'Exection finished in {runtime}')



def set_paths():
    import platform
    args.dataroot = globals.PATHS[args.usr][platform.system()][args.dataset]['root']
    args.ckpt_path = globals.PATHS[args.usr][platform.system()][args.dataset]['ckpt']



def init_net(args):
    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(args.group_norm, planes)
        norm_layer = gn_helper

    def get_heads_classification(self):
        for m in self.modules(): pass
        return m

    if args.model == 'wrn':
        net = WideResNet(widen_factor=2, depth=40, num_classes=10)
        WideResNet.get_heads = get_heads_classification

    elif args.model == 'res':
        net = ResNetCifar(args.depth, args.width, channels=3, classes=10,
                          norm_layer=norm_layer)
        ResNetCifar.get_heads = get_heads_classification

    elif args.model == 'res18':
        num_classes = 200 if args.dataset == 'tiny-imagenet' else 1000
        # net = models.resnet18(weights='DEFAULT', norm_layer=norm_layer, num_classes=num_classes)
        net = models.resnet18(norm_layer=norm_layer, num_classes=num_classes)
        models.resnet.ResNet.get_heads = get_heads_classification

    else:
        raise Exception(f'Invalid model argument: {args.model}')

    net = net.cuda()
    setattr(net.__class__, 'bn_stats', {})
    return net


def initial_checks(net, args):
    log.info('Running initial checks.')
    dataset_checks(args)

    if not exists(args.ckpt_path):
        args.epochs = 350
        log.info('Checkpoint trained on initial task not found - Starting training.')
        args.task = 'initial'
        args.ckpt_path = join('checkpoints', args.dataset, args.model)
        train(net, args, args.ckpt_path)
        log.info(f'Checkpoint trained on initial task saved at {args.ckpt_path}/initial.pt')
        args.ckpt_path = join(args.ckpt_path, 'initial.pt')


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
    parser.add_argument('--max_unsuccessful_reductions', default=3, type=int)

    parser.add_argument('--split_ratio', default=0.35, type=float)
    parser.add_argument('--split_seed', default=42, type=int)

    parser.add_argument('--usr', default=None, type=str)

    args: Namespace = parser.parse_args()
    main()
