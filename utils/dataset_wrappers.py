import os
from posixpath import split
import numpy as np
from PIL import Image
import csv
from torchvision.datasets import ImageFolder, vision
import pickle


class ImgNet(ImageFolder):
    """
        Class for Imagenet and Imagenet-mini dataset.
        For training and test split, the entire training set is choosen
        and expected to be filtered for an appropriate subset by the creator.
    """
    initial_dir = ''

    def __init__(self, root, split, task, severity, transform, **kwargs):
        split_dir = 'val' if split == 'val' else 'train'
        if task == 'initial':
            root = os.path.join(root, self.initial_dir, split_dir)
        else:
            root = os.path.join(root, self.initial_dir + '-c', split_dir, task,
                                str(severity))
        super().__init__(root, transform, **kwargs)


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



class CIFAR(vision.VisionDataset):
    """
        Class for Cifar10 dataset.
        For training and validation split, the entire training set is choosen
        and expected to be filtered for an appropriate subset by the creator.
    """
    corruptions_dir = 'CIFAR-10-C'
    def __init__(self, root: str, task, split='train', transform=None,
                 target_transform=None, severity=1):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        self.num_samples = 10000 if split == 'test' else 50000
        split_dir = 'test' if split == 'test' else 'train'

        self.data = []
        self.targets = []

        if task == 'initial':
            self._load_initial_task(split)
            return

        start = self.num_samples * (severity - 1)
        end = self.num_samples * severity
        targets_path = os.path.join(self.root, self.corruptions_dir, split_dir, 'labels.npy')
        self.targets = np.load(targets_path)[start : end]
        f_path = os.path.join(self.root, self.corruptions_dir, split_dir, f'{task}.npy')
        self.data = np.load(f_path, mmap_mode='c')[start : end]



    def _load_initial_task(self, split):
        base_folder = "cifar-10-batches-py"
        train_list = ["data_batch_1", "data_batch_2", "data_batch_3",
                      "data_batch_4", "data_batch_5"]
        test_list = ["test_batch"]

        for file_name in test_list if split == 'test' else train_list:
            file_path = os.path.join(self.root, base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))


    def __getitem__(self, idx: int):
        img, target = self.data[idx], self.targets[idx]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return self.num_samples


    def get_pil_image_from_idx(self, idx: int = 0):
        return Image.fromarray(self.data[idx])

