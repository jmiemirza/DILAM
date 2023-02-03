# Dataset Directory structure

## KITTI
```
args.dataroot
├── fog
|   ├── fog_30
|   |   ├── *.png
|   |
|   ├── ... other severities
|
├── initial
|   └── images
|       ├── *.png
|
├── labels_caches [this is an initially empty directory]
|
├── labels_yolo_format
|   ├── *.txt
|
├── rain
|   ├── 200mm
|   |   ├── *png
|   |
|   ├── ... other severities
|
├── snow
|   ├── 5
|   |   ├── *png
|   |
|   ├── ... other severities
|
├── test.txt
├── train.txt
└── val.txt
```
The .txt files contain a list of image names defining the train/val/test splits.

The splits used by us can be copied from [here](dataset_splits).


