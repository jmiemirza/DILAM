import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import torch
from tqdm import tqdm
from utils.data_loader import get_loader, set_severity
import globals
import logging

log = logging.getLogger('MAIN.KNN')


class KITTIWeatherKNNClassifier(KNeighborsClassifier):
    def __init__(self, n_neighbors=4, *, weights="uniform", algorithm="auto",
                 leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=-1):
        super().__init__(n_neighbors, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)


    def create_train_dataset_and_fit(self, args, split='train', pad=0.5, rect=True):
        """
            Uses yolo loader to iterate over train sets and creates a normalized
            and flattened 3D histogram from HSV channels for every image as the
            train set for this KNN classifier.
            The dataset and labels created here are then saved and assigned as this
            KNeighborsClassifier training set/labels using it's "fit" method.
        """
        log.info(f'Generating KNN train set and labels.')
        args.severity_idx = 0
        args.gs = 32
        hists = []
        labels = []
        for label, args.task in enumerate(globals.KITTI_CLS_WEATHER):
            set_severity(args)
            dataloader = get_loader(args, split='train', pad=pad, rect=rect)
            ldr = tqdm(enumerate(dataloader), total=len(dataloader), desc=args.task)
            for batch_i, (img, targets, paths, shapes) in ldr:
                hists += self.get_histograms(img.to(torch.float))
                labels += [label for _ in range(args.batch_size)]

        hists = np.array(hists, dtype=np.float16)
        log.info(f'hists shape: {hists.shape}, hists size: {(hists.nbytes / (1024 * 1000.0)):.2f}MB')
        np.save(f'{args.checkpoints_path}/knn_{split}_hists.npy', hists)
        np.save(f'{args.checkpoints_path}/knn_{split}_labels.npy', labels)

        self.fit(hists, labels)


    def predict_batch(self, in_batch, verbose=False):
        """
            Predict classes in in_batch
            Returns (majority voted prediction, individual predictions)
        """
        hists = self.get_histograms(in_batch)
        predictions_per_hist = self.predict(hists)

        # predictions majority vote
        most_predicted_in_batch = np.argmax(np.bincount(predictions_per_hist))

        if verbose:
            log.info(f'Prediction: {globals.KITTI_CLS_WEATHER[most_predicted_in_batch]} '
                     f'(Predictions: {predictions_per_hist})')

        return most_predicted_in_batch, list(predictions_per_hist)


    def eval_per_class_accuracy(self, args, split='val', pad=0.5, rect=True):
        """
            Method to test KNN classifier performance.
            Use yolo loader for given split and classify all images by weather condition.
            Reports accuracies with and without batch majority votes.
            Batch size is set by args.batch_size
        """
        args.severity_idx = 0
        args.gs = 32
        for args.task in ['initial'] + globals.TASKS:
            majority_predictions = []
            all_predictions = []
            set_severity(args)
            dataloader_test = get_loader(args, split=split, pad=pad, rect=rect)
            ldr = tqdm(enumerate(dataloader_test), total=len(dataloader_test), desc=args.task)
            for batch_i, (img, targets, paths, shapes) in ldr:
                majority_pred, per_hist_preds = self.predict_batch(img)
                majority_predictions.append(majority_pred)
                all_predictions += per_hist_preds

            gt = globals.KITTI_CLS_WEATHER.index(args.task)
            num_misses = sum(1 for n in majority_predictions if n != gt)
            batch_acc = 100 - (num_misses/len(majority_predictions))
            num_misses = sum(1 for n in all_predictions if n != gt)
            per_img_acc = 100 - (num_misses/len(all_predictions))

            most_predicted_all_batches = np.argmax(np.bincount(majority_predictions))
            most_predicted_str = f'(most predicted: {globals.KITTI_CLS_WEATHER[most_predicted_all_batches]})'.ljust(26)
            batch_acc_str = f'batch acc: {batch_acc:.2f}%'.ljust(18)
            per_img_acc_str = f'per img acc: {per_img_acc:.2f}%'.ljust(21)
            log.info(f'{args.task.ljust(7)} {most_predicted_str}'
                     f'{batch_acc_str}, {per_img_acc_str}')


    @staticmethod
    def get_histograms(img_batch: torch.Tensor, bins=(8, 8, 8)):
        """
            Takes an image batch tensor (B x C x H x W) as input and returns
            a list of normalized and flattened HSV 3D histograms (one for each image in the batch)
        """
        hists = []
        hsv = KITTIWeatherKNNClassifier.rgb2hsv_torch(img_batch).numpy()
        hsv = np.moveaxis(hsv, 1, -1)
        num_imgs = hsv.shape[0]
        for img_idx in range(num_imgs):
            img_reshaped = hsv[img_idx].reshape(hsv[img_idx].shape[0] * hsv[img_idx].shape[1], hsv[img_idx].shape[2])
            hist = np.histogramdd(img_reshaped, bins=bins, range=[(0, 180), (0, 256), (0, 256)], density=True)[0]
            hists.append(hist.flatten())
        return hists


    @staticmethod
    def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
        """
            Source: https://github.com/limacv/RGB_HSV_HSL
            Convert a batch of RGB images (B x C x H x W) to HSV.
        """
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :], dtype=torch.float)
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(hsv_h), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)
