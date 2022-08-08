import os
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder, vision


class ImageFolderWrap(ImageFolder):
    def __init__(self, root, transform, **kwargs):
        super().__init__(root, transform=transform, **kwargs)


    def get_pil_image_from_idx(self, idx: int = 0):
        return Image.open(self.imgs[idx][0])


class CIFAR10C(vision.VisionDataset):
    def __init__(self, root: str, *tasks, train=True, transform=None,
                 target_transform=None, severity=1):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        if train:
            self.split = 'train'
            self.num_samples = 50000
        else:
            self.split = 'test'
            self.num_samples = 10000

        self.num_tasks = len(tasks)
        self.data = []
        self.targets = []
        start = self.num_samples * (severity - 1)
        end = self.num_samples * severity
        labels_path = os.path.join(self.root, self.split, 'labels.npy')
        targets = np.load(labels_path)[start : end]
        if self.num_tasks == 1:
            f_path = os.path.join(self.root, self.split, f'{tasks[0]}.npy')
            self.data = np.load(f_path)[start : end]
            self.targets = np.load(labels_path)[start : end]
        else:
            for task in tasks:
                f_path = os.path.join(self.root, self.split, f'{task}.npy')
                self.data.extend(np.load(f_path, mmap_mode='c')[start : end])
                self.targets.extend(targets)


    def __getitem__(self, idx: int):
        img, target = self.data[idx], self.targets[idx]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return self.num_samples * self.num_tasks


    def get_pil_image_from_idx(self, idx: int = 0):
        return Image.fromarray(self.data[idx])

