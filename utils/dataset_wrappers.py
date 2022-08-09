import os
import numpy as np
from PIL import Image
import csv
from torchvision.datasets import ImageFolder, vision


class ImageFolderWrap(ImageFolder):
    def __init__(self, root, transform, **kwargs):
        super().__init__(root, transform=transform, **kwargs)


    def get_pil_image_from_idx(self, idx: int = 0):
        return Image.open(self.imgs[idx][0])



class KITTI(vision.VisionDataset):
    image_dir_name = "image_2"
    labels_dir_name = "label_2"
    def __init__(self, root, split: str, task, severity, transform=None,
                 target_transform=None, transforms=None):
        super().__init__(root, transform=transform, transforms=transforms,
                         target_transform=target_transform)
        self.images = []
        self.targets = []
        self.root = root
        self.split = split
        if task == 'initial':
            raw_folder = os.path.join(self.root, 'Kitti' , "raw")
        else:
            raw_folder = os.path.join(self.root, 'Kitti-c', task, str(severity))

        image_dir = os.path.join(raw_folder, self.split, self.image_dir_name)
        labels_dir = os.path.join(raw_folder, self.split, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))


    def __getitem__(self, index: int):
        image = Image.open(self.images[index])
        target = self._parse_target(index)
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


    def _parse_target(self, index: int):
        target = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "type": line[0],
                        "truncated": float(line[1]),
                        "occluded": int(line[2]),
                        "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                        "rotation_y": float(line[14]),
                    }
                )
        return target


    def get_pil_image_from_idx(self, idx: int = 0):
        return Image.open(self.images[idx])



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

