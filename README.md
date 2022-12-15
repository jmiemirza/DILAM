# Installation

1) `git clone` this repository.
2) `pip install -r requirements.txt` to install required packages


# Preparing Datasets

## For KITTI dataset
* Download Clear (Original) [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).
* Download [KITTI-Fog/Rain](https://team.inria.fr/rits/computer-vision/weather-augment/) datasets.
* Super-impose snow on KITTI dataset through this [repository](https://github.com/hendrycks/robustness).
* Generate labels YOLO can use (see [Dataset directory structures](#dataset-directory-structures) subsection).


## For ImageNet and CIFAR datasets
* Download the original train and test set for [ImageNet](https://image-net.org/download.php) & [ImageNet-C](https://zenodo.org/record/2235448#.Yn5OTrozZhE) datasets.
* Download the original train and test set for [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) & [CIFAR-10C](https://zenodo.org/record/2535967#.Yn5QwbozZhE) datasets.
* Generate _corrupted_ version of train sets through this [repository](https://github.com/hendrycks/robustness).

## Dataset directory structures
### For KITTI labels:
To generate labels YOLO can use from the original KITTI labels run

`python main.py --kitti_to_yolo_labels /path/to/original/kitti`

This is expecting the path to the original KITTI directory structure
```
path_to_specify
└── raw
    └── training
        ├── image_2
        └── label_2
```
Which will create a `yolo_style_labels` directory in the `raw` directory, containing
the KITTI labels in a format YOLO can use.

### For all datasets:
Structure the choosen dataset(s) as described [here](directory_scructures.md).

# Running Experiments

We recommend first setting up user specific paths in the `PATHS` dictionary in `config.py`
By following the existing entry as an example. This will lead to less commandline
arguments. Alternatively all paths can be passed explicitly as commandline
arguments instead.

Assuming paths have been added to the `PATHS` dictionary for a user `sl`, all KITTI experiments
for the highest _weather severity_ can be ran like this:
```
python main.py --usr sl --dataset kitti
```
All KITTI experiments for the highest _weather severity_ with DISC adaption phase only using 10 samples
can be ran like this:
```
python main.py --usr sl --dataset kitti --num_samples 10
```

Running only a selection of Baselines for a different severity of fog can be ran like this:
```
python main.py --usr sl --dataset kitti --baselines source_only disjoint --fog_severities fog_40
```
Note that severities also correspond to the directory names where their data is located.

All KITTI experiments with the highest _weather severity_ with specifying paths in commandline arguments
can be done like this:
```
python main.py --dataset kitti --dataroot /path/to/dataroot --ckpt_path /path/to/checkpoint.pt
```

Many more options, such as batch size, learning rate, workers, are accesible through commandline arguments.
They can be seen in `main.py` or by running `python main.py -h`