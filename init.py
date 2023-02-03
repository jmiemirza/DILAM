import logging
import logging.config
import os
from os.path import split, join, realpath, exists

import globals, config
from torch import nn, save
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device
from utils.training_yolov3 import train as train_yolo

log = logging.getLogger('MAIN')


def set_paths(args):
    args.dataroot = config.PATHS[args.usr][args.dataset]['root']
    args.ckpt_path = config.PATHS[args.usr][args.dataset]['ckpt']


def init_net(args, cls = False):
    device = select_device(args.device, batch_size=args.batch_size)

    if hasattr(args, 'orig_ckpt_path'):
        args.ckpt_path = args.orig_ckpt_path
    else:
        args.orig_ckpt_path = args.ckpt_path

    if cls:
        net = init_cls_net(args, device)
        args.gs = max(int(net.stride.max()), 32)
        args.img_size = [check_img_size(x, args.gs) for x in args.img_size]

        setattr(net.__class__, 'bn_affine', {})
        save(net.state_dict(), 'affine_yolo_kitti_state_dict_ckpt.pt')
        args.ckpt_path = join(split(realpath(__file__))[0], 'affine_yolo_kitti_state_dict_ckpt.pt')

        return net

    if exists(args.ckpt_path):
        net = attempt_load(args.ckpt_path, map_location=device)
        args.gs = max(int(net.stride.max()), 32)
        args.img_size = [check_img_size(x, args.gs) for x in args.img_size]

    else:
        net = init_yolov3(args, device)
        args.gs = max(int(net.stride.max()), 32)
        args.img_size = [check_img_size(x, args.gs) for x in args.img_size]
        train_initial(args, net)
    save(net.state_dict(), 'yolo_kitti_state_dict_ckpt.pt')
    args.ckpt_path = join(split(realpath(__file__))[0], 'yolo_kitti_state_dict_ckpt.pt')

    net = net.to(device)
    setattr(net.__class__, 'bn_stats', {})
    return net


def init_cls_net(args, device):
    import torch
    from models.yolo import Model

    ckpt = None
    if exists(args.ckpt_path):
        ckpt = torch.load(f'{args.ckpt_path}', map_location=device)  # load checkpoint

    net = Model(args.cfg, ch=3, nc=args.nc).to(device)  # create

    if ckpt:
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        net.load_state_dict(state_dict, strict=False)  # load
    else:
        net.to(device)
        args.gs = max(int(net.stride.max()), 32)
        args.img_size = [check_img_size(x, args.gs) for x in args.img_size]
        train_initial(args, net)
    net.to(device)
    return net


def init_yolov3(args, device):
    import torch
    from utils.torch_utils import intersect_dicts, torch_distributed_zero_first
    from utils.google_utils import attempt_download
    from models.yolo import Model

    log.info('Loading yolov3.pt weights.')
    hyp = args.yolo_hyp()
    with torch_distributed_zero_first(args.global_rank):
        attempt_download('yolov3.pt')  # download if not found locally
    ckpt = torch.load('yolov3.pt', map_location=device)  # load checkpoint
    if hyp.get('anchors'):
        ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
    net = Model(args.cfg or ckpt['model'].yaml, ch=3, nc=args.nc).to(device)  # create
    exclude = ['anchor'] if args.cfg or hyp.get('anchors') else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, net.state_dict(), exclude=exclude)  # intersect
    net.load_state_dict(state_dict, strict=False)  # load
    net.to(device)
    return net


def train_initial(args, net):
    args.epochs = 350

    log.info('Checkpoint trained on initial task not found - Starting training.')
    args.task = 'initial'
    save_dir_path = join('checkpoints', args.dataset, args.model, 'initial')

    device = select_device(args.device, batch_size=args.batch_size)
    args.save_dir = save_dir_path
    train_yolo(args.yolo_hyp(), args, device, model=net)
    args.ckpt_path = join(split(realpath(__file__))[0], save_dir_path, 'weights', 'best.pt')

    log.info(f'Checkpoint trained on initial task saved at {args.ckpt_path}')


def init_settings(args):
    args.baselines = [x.lower() for x in args.baselines]
    os.makedirs('results', exist_ok=True)
    os.makedirs(args.checkpoints_path, exist_ok=True)

    if not args.model:
        args.model = 'yolov3'

    if args.tasks:
        globals.TASKS = args.tasks
    else:
        globals.TASKS = config.KITTI_TASKS

    args.num_severities = max([len(args.fog_severities),
                                len(args.rain_severities),
                                len(args.snow_severities)])
    globals.KITTI_SEVERITIES['fog'] = args.fog_severities
    globals.KITTI_SEVERITIES['rain'] = args.rain_severities
    globals.KITTI_SEVERITIES['snow'] = args.snow_severities

    # set args.yolo_hyp to a function returning a copy of config.YOLO_HYP
    # as some values get changed during training, which would lead to
    # false values if multiple training sessions are started within one
    # execution of the script
    def get_yolo_hyp():
        return config.YOLO_HYP.copy()
    config.YOLO_HYP['lr0'] = args.lr
    args.yolo_hyp = get_yolo_hyp

    # opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    args.world_size = 1
    args.global_rank = -1

    args.img_size.extend([args.img_size[-1]] * (2 - len(args.img_size)))  # extend to 2 sizes (train, test)
    args.total_batch_size = args.batch_size
    args.nc = 8
    args.names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                    'Cyclist', 'Tram', 'Misc']


