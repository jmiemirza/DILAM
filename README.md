Datasets Directory structures
=
CIFAR-10-C
-
Excpected directory structure for CIFAR-10-C

If the `cifar-10-batches-py` directory and its contents are not present, they will be downloaded.

The files in the `train` directory are expected to be the **corrupted** version of the original training set.

**Currently**, the files in `train` are also expected to only contain their level 5 corruption.
For files containing all levels, the necessary code additions have been added as comments
in `data_loader.py`. This will be changed to only use files with all 5 severity levels in the near future.

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