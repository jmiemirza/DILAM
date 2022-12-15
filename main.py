import argparse
import logging
import logging.config
import sys
import time
from argparse import Namespace

import torch.backends.cudnn as cudnn
from datetime import datetime
import baselines
import config
from disc import disc
from init import init_net, init_settings, initial_checks, set_paths
from utils.results_manager import ResultsManager
from utils.utils import timedelta_to_str

# logging.config.dictConfig(config.LOGGER_CFG)
# log = logging.getLogger('MAIN')


def main(args):
    if args.kitti_to_yolo_labels:
        from utils.utils import kitti_labels_to_yolo
        kitti_labels_to_yolo(args.kitti_to_yolo_labels)
        exit()

    cudnn.benchmark = True
    start_time = datetime.now()

    log.info('------------------------------------ NEW RUN ------------------------------------')
    log.info(f'Running: {" ".join(sys.argv)}')
    log.info('Full args list:')
    for arg in vars(args):
        log.info(f'{arg}: {getattr(args, arg)}')
    log.info('---------------------------------------------------------------------------------')

    results = ResultsManager('mAP@50' if args.dataset == 'kitti' else 'Error')

    init_settings(args)
    if args.usr:
        set_paths(args)
    net = init_net(args)
    initial_checks(net, args)

    # TODO fixed initial task lr being set with yolo.. untested tho
    # this also changed lr adjustment in warmup

    # By default only the highest severity is used. This can be changed using
    # the --(robustness/fog/rain/snow)_severities commandline arguments.
    # Multiple severities can be listed, but for the KITTI dataset keep in mind
    # that the different severities are accessed by the same index.
    # Meaning if for example the results for two different fog severities,
    # with everything else staying the same, are needed, the other tasks
    # severities need to be listed twice, like this:
    # --fog_severities fog_30 fog_40
    # --rain_severities 200m 200m
    # --snow_severities 5 5
    #
    # Running DISC can be skipped by using --no_disc
    # Running DISC adaption phase can be skipped by using --no_disc_adaption
    # in which case the batch norm running estimates checkpoint from a previous
    # run will be used.
    # By default all baselines are ran.
    # Specific baselines to run can be listed using --baselines
    # Baselines can be skipped entirely by using --baselines without listing
    # any baselines to run
    #
    # "online" and/or "offline" scenario can be specified using --scenario
    # by default both scenarios are ran.
    #
    for args.severity_idx in range(args.num_severities):
        if not args.no_disc:
            disc(args, net)
        if 'source_only' in args.baselines:
            baselines.source_only(net, args)
        for scenario in args.scenario:
            if 'disjoint' in args.baselines:
                baselines.disjoint(net, args, scenario)
            if 'freezing' in args.baselines:
                baselines.freezing(net, args, scenario)
            if 'fine_tuning' in args.baselines:
                baselines.fine_tuning(net, args, scenario)
            if 'joint_training' in args.baselines:
                baselines.joint_training(net, args, scenario)

        # plot & log results summary for every severity index
        timestamp_str = time.strftime('%b-%d-%Y_%H%M', time.localtime())
        if results.has_results():
            results.save_to_file(file_name=f'{timestamp_str}_raw_results.pkl')
            results.print_summary_latex()
            results.plot_summary(f'{timestamp_str}_summary_plot.png')


    runtime = datetime.now() - start_time
    log.info(f'Execution finished in {timedelta_to_str(runtime)}')


# Log uncaught exceptions, that aren't keyboard interrupts
def handle_exception(exception_type, value, traceback):
    if issubclass(exception_type, KeyboardInterrupt):
        sys.__excepthook__(exception_type, value, traceback)
        return
    log.exception('Exception occured:', exc_info=(exception_type, value, traceback))
sys.excepthook = handle_exception


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr', default=None, type=str)
    parser.add_argument('--dataroot', default='path/to/dataroot')
    parser.add_argument('--ckpt_path', default='path/to/checkpoint.pt')
    parser.add_argument('--dataset', default='kitti')
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--logfile', default='log.txt', type=str)

    # General run settings
    all_baselines = ['source_only', 'disjoint', 'freezing', 'fine_tuning', 'joint_training']
    parser.add_argument('--baselines', default=all_baselines, type=str, nargs='*',
                        help='List of baselines to run')
    parser.add_argument('--scenario', default=['online', 'offline'], type=str, nargs='*',
                        help='Scenarios to run (online and/or offline)')
    parser.add_argument('--robustness_severities', default=['5'], type=str, nargs='*')
    parser.add_argument('--fog_severities', default=['fog_30'], type=str, nargs='*')
    parser.add_argument('--rain_severities', default=['200mm'], type=str, nargs='*')
    parser.add_argument('--snow_severities', default=['5'], type=str, nargs='*')
    parser.add_argument('--checkpoints_path', default='checkpoints', help='path where model checkpoints will be saved')

    # DUA/DISC adaption
    parser.add_argument('--num_samples', default=50, type=int)
    parser.add_argument('--decay_factor', default=0.94, type=float)
    parser.add_argument('--min_mom', default=0.005, type=float)
    parser.add_argument('--no_disc', action='store_true', help='do not run DISC')
    parser.add_argument('--no_disc_adaption', action='store_true',
                        help='skip DISC adaption phase (assumes existing BN running estimates checkpoint)')

    # Learning & Loading
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--initial_task_lr', default=0.01, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--workers', type=int, default=1, help='maximum number of dataloader workers')
    parser.add_argument('--yolo_lr_adjustment', type=str, default='thirds',
                        choices=['thirds', 'linear_lr', 'cosine'],
                        help='how yolov3 training reduces learning rate')

    # LR scheduler and early stopping
    parser.add_argument('--patience', default=4, type=int)
    parser.add_argument('--lr_factor', default=1/3, type=float)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--max_unsuccessful_reductions', default=3, type=int)

    # For creating a val/test set from train set for CIFAR/ImageNet
    parser.add_argument('--split_ratio', default=0.35, type=float)
    parser.add_argument('--split_seed', default=42, type=int)

    # ResNet
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--rotation_type', default='rand')

    # yolov3
    parser.add_argument('--weights', type=str, default='yolov3.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # disabled
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--start_disjoint_offline_from_coco', action='store_true',
                        help='start offline disjoint training from checkpoint trained on MS COCO')
    parser.add_argument('--use_freezing_heads_ckpts', action='store_true',
                        help='Use freezing baseline heads from a previous run. '
                             'Without this option previously saved heads are moved.')
    # yolov3 untested
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache_images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image_weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--multi_scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single_cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--sync_bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log_imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log_artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')

    # other
    parser.add_argument('--kitti_to_yolo_labels', default=None, type=str,
                        help='Generate YOLO style labels from KITTI labels, given original KITTI root dir')

    args: Namespace = parser.parse_args()

    config.LOGGER_CFG['handlers']['file_handler']['filename'] = args.logfile
    logging.config.dictConfig(config.LOGGER_CFG)
    log = logging.getLogger('MAIN')

    main(args)
