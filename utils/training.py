from typing import Iterable
from warnings import warn
from torch import save
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import chain
from utils.data_loader import prepare_train_valid_loaders
from utils.testing import test
from utils.utils import make_dirs


class ReduceLROnPlateauEarlyStop(ReduceLROnPlateau):
    """
        Extension of ReduceLROnPlateau to also implement early stopping.
        The argument max_unsuccessful_reductions defines how many lr reductions
        without improvement can be made before meeting the early stopping
        criteria, in which case the step() method returns False instead of True
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False,
                 max_unsuccessful_reductions=3):
        super().__init__(optimizer, mode, factor, patience,
                         threshold, threshold_mode, cooldown,
                         min_lr, eps, verbose)
        self.consecutive_lr_reductions = 0
        self.max_unsuccessful_reductions = max_unsuccessful_reductions

    # slightly modified ReduceLROnPlateau step() method, to keep track of
    # lr decreases and return False on no improvement after
    # max_unsuccessful_reductions lr reductions
    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warn(optim.lr_scheduler.EPOCH_DEPRECATION_WARNING)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            self.consecutive_lr_reductions = 0
        else:
            self.num_bad_epochs += 1
            if self.consecutive_lr_reductions >= self.max_unsuccessful_reductions:
                if self.verbose:
                    print("Early stopping criteria reached!")
                return False

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.consecutive_lr_reductions += 1
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return True


def get_heads_params(model):
    heads = model.get_heads()
    if isinstance(heads, Iterable):
        return chain.from_iterable([m.parameters() for m in heads])
    return heads.parameters()


def train(model, args, results_path='checkpoints/', train_heads_only=False):
    make_dirs(results_path)
    if results_path[-1] != '/':
        results_path += '/'

    if train_heads_only:
        optimizer = optim.SGD(get_heads_params(model), lr=args.lr, momentum=0.9,
                              weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().cuda()
    n = args.max_unsuccessful_reductions
    scheduler = ReduceLROnPlateauEarlyStop(optimizer, factor=args.lr_factor,
                                           patience=args.patience,
                                           verbose=args.verbose,
                                           max_unsuccessful_reductions = n)
    train_loader, valid_loader = prepare_train_valid_loaders(args)

    all_err_cls = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_one_epoch(model, epoch, optimizer, train_loader, criterion)
        err_cls = test(valid_loader, model)[0]
        all_err_cls.append(err_cls)
        if err_cls <= min(all_err_cls):
            save(model.state_dict(), f'{results_path}{args.corruption}.pt')

        print(('Epoch %d/%d:' % (epoch, args.epochs)).ljust(20) +
              '%.2f' % (err_cls * 100))

        if not scheduler.step(err_cls):
            if args.verbose:
                print("Finished training")
            return


def train_one_epoch(model, epoch, optimizer, train_loader, criterion):
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch} avg loss per batch: {total_loss / (batch_idx + 1):.4f}')
