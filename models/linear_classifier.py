import logging
import logging.config
import time
import globals
from utils.data_loader import set_severity, get_loader
import torch.nn as nn
import torch.optim as optim
import torch

log = logging.getLogger('MAIN')
loss_ce = nn.CrossEntropyLoss()


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def test(teloader, model):
    device = next(model.parameters()).device
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(teloader), batch_time, top1, prefix='Test: ')
    one_hot = []
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    end = time.time()
    for i, (inputs, labels) in enumerate(teloader):
        with torch.no_grad():
            inputs, labels = inputs.to(device).float()/255, labels.to(device)
            outputs = model.forward_cls(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.train()
    return top1.avg


def test_cls(net, args, rect=True, verbose=False, task='fog', fname='cls_head.pt'):
    device = next(net.parameters()).device
    net.class_head.load_state_dict(torch.load(fname, map_location=device))
    args.task = task
    set_severity(args)
    from utils.general import check_img_size
    tmp_opt_imgsz = args.img_size
    args.img_size = [check_img_size(x, args.gs) for x in [840, 840]]
    dataloader_test = get_loader(args, split='val', pad=0.5, rect=rect)
    net.eval()
    with torch.no_grad():
        all = list()
        for batch_i, (img, targets, paths, shapes) in enumerate(dataloader_test):
            img = img.to(device).float()/255
            out = net.forward_cls(img, verbose=verbose)
            all.append(out.max(1)[1].cpu().numpy()[:])
        import numpy as np
        all = np.concatenate(all, axis=0)
        # print(all)
        correct = np.count_nonzero(all == globals.KITTI_CLS_WEATHER.index(task))
        # print(correct)
        log.info(f'{task} acc ::: {(100 * correct / len(all)):.2f}%')


def train_cls_head(net, args, rect=False, fname='cls_head.pt', pad=0.5):
    log.info('Start training classifier head')
    device = next(net.parameters()).device
    tmp_tasks = globals.TASKS
    globals.TASKS = globals.KITTI_CLS_WEATHER[1:]
    args.task = globals.KITTI_CLS_WEATHER[-1] # last task -> joint loader for all tasks

    set_severity(args)
    cls_loader_val = get_loader(args, split='val', joint=True, pad=pad, shuffle=True, rect=rect, for_classification=True)
    cls_loader_train = get_loader(args, split='train', joint=True, pad=pad, shuffle=True, rect=rect, for_classification=True)
    cls_loader_test = get_loader(args, split='test', joint=True, pad=pad, shuffle=True, rect=rect, for_classification=True)

    optimizer = optim.SGD(net.class_head.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
    accuracies = []
    for epoch in range(1, args.epochs+1):
        print(f'Epoch\t\titer\t\tloss')

        for m in net.modules():
            m.requires_grad_(False)
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
                m.eval()

        for m in net.class_head.modules():
            m.requires_grad_(True)
        # count = 0
        for m in net.class_head.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
                m.train()
        #         count += 1
        # print(count)
        # quit()
        for batch_idx, (imgs, cls_labels) in enumerate(cls_loader_train):
            optimizer.zero_grad()
            imgs = imgs.to(device).float() / 255.0
            out = net.forward_cls(imgs)
            loss = loss_ce(out, cls_labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f'{epoch}/{args.epochs}\t\t{batch_idx+1}\t\t{loss}')

        print('Evaluating ...')
        acc = test(cls_loader_test, model=net)
        accuracies.append(acc)
        log.info(f'Accuracy Epoch {epoch}: {acc*100}')
        if acc >= max(accuracies):
            log.info(f'New best - saving to {fname}')
            torch.save(net.class_head.state_dict(), fname)

    globals.TASKS = tmp_tasks