Datasets Directory structures
=
CIFAR-10-C
-
```
args.dataroot
├── cifar-10-batches-py
|   ├── batches.meta
|   ├── data_batch_1
|   ├── ...
|
└── CIFAR-10-C
    ├── test
    |   ├── brightness.npy
    |   ├── contrast.npy
    |   ├── ...
    |
    └── train
        ├── brightness.npy
        ├── contrast.npy
        ├── ...

```


Tiny-Imagenet-200-C
-
```
args.dataroot
├── tiny-imagenet-200
|   ├── train
|   |   ├── n01443537
|   |   |   └──images
|   |   |       ├── *.JPEG
|   |   |       ├── ...
|   |   |
|   |   ├── n01629819
|   |   ├── ...
|   |
|   └── val
|       ├── n01443537
|       |   └── images
|       |       ├── *.JPEG
|       |       ├── ...
|       |
|       ├── n01629819
|       ├── ...
|
└── tiny-imagenet-200-c
    ├── val
    |   ├── brightness
    |   |   ├── 1
    |   |   |   ├── n01443537
    |   |   |   |   ├── *.JPEG
    |   |   |   |   ├── ...
    |   |   |   |
    |   |   |   ├── n01629819
    |   |   |   ├── ...
    |   |   |
    |   |   ├── 2
    |   |   ├── 3
    |   |   ├── 4
    |   |   └── 5
    |   |
    |   ├── contrast
    |   ├── ...
    |
    └── train
        ├── ... same as tiny-imagenet-200-c/val

```


Imagenet
-
```
args.dataroot
├── imagenet
|   ├── train
|   |   ├── n01443537
|   |   |   ├── *.JPEG
|   |   |   ├── ...
|   |   |
|   |   ├── n01629819
|   |   ├── ...
|   |
|   └── val
|       ├── n01443537
|       |   ├── *.JPEG
|       |   ├── ...
|       |
|       ├── n01629819
|       ├── ...
|
└── imagenet-c
    ├── val
    |   ├── brightness
    |   |   ├── 1
    |   |   |   ├── n01443537
    |   |   |   |   ├── *.JPEG
    |   |   |   |   ├── ...
    |   |   |   |
    |   |   |   ├── n01629819
    |   |   |   ├── ...
    |   |   |
    |   |   ├── 2
    |   |   ├── 3
    |   |   ├── 4
    |   |   └── 5
    |   |
    |   ├── contrast
    |   ├── ...
    |
    └── train
        ├── ... same as imagenet-c/val

```