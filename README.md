# Installation

1) `git clone` this repository.
2) `pip install -r requirements.txt` to install required packages


# Preparing Datasets

* Download Clear (Original) [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).
* Download [KITTI-Fog/Rain](https://team.inria.fr/rits/computer-vision/weather-augment/) datasets.
* Super-impose snow on KITTI dataset through this [repository](https://github.com/hendrycks/robustness).
* Generate labels YOLOv3 can use (see [KITTI labels](#kitti-labels) section).


## Dataset directory structure
### KITTI labels:
To generate labels YOLOv3 can use from the original KITTI labels run

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
the KITTI labels in a format YOLOv3 can use. These labels will have to be moved
to the dataset directory, as described in the following section ([Directory structure](#directory-structure)).

### Directory structure:
Dataset direcory structure as described [here](directory_scructures.md).

# Pretrained Checkpoints

Pretrained checkpoints can be downloaded. [TODO: links]

* Download KITTI pre-trained [Yolov3](https://arxiv.org/abs/1804.02767) from [here](https://drive.google.com/file/d/1NWwhX7zmsQh0791VUL_5mB9cF29Xd0SU/view?usp=sharing).

* Memory bank with all batchnorm layers adapted.

* Memory bank with all except the first two batchnorm layers adapted.

* Classification head when using the linear classification head.

We recommend placing all checkpoints in `./checkpoints/`.

# Running Experiments

Recommended Python verion: `Python 3.6.9`

We recommend first setting up user specific paths in the `PATHS` dictionary in `config.py`
By following the existing entries as example. This will lead to less commandline
arguments. Alternatively all paths can be passed explicitly as commandline
arguments.


Assuming paths have been added to the `PATHS` dictionary for a user `sl`,
DILAM adaptation and plug & play can be executed by running:
```
python main.py --usr sl
```
Asuming DILAM adaptation has been ran at one point or the memory bank file
was aquired by different means, DILAM adaptation can be omitted by providing
`--no_dilam_adapt`. Baselines to run can be specified using `--baselines`.
Tasks and their order can be specified using `--tasks`
```
python main.py --usr sl --no_dilam_adapt --baselines disc source_only disjoint --tasks snow fog rain
```

Most relevant options may include:

GPU to run on: `--device`, Batch size: `--batch_size`, Number of runs: `--num_runs`, Image size: `--img_size`

DILAM adaption has a separately specifyable batch size `--dilam_adapt_batch_size`.

By default DILAM adapts all except the first two batchnorm layers, this can be
changed to adapt all by using `--dilam_adapt_all`

For a full list of options check `main.py` or run `python main.py -h`
```
python main.py --usr sl --batch_size 8 --device 0 --num_runs 10 --img_size 840
```

Specifying paths in commandline arguments rather than using the `PATHS` dictionary in `config.py`:
```
python main.py --dataroot /path/to/dataroot --ckpt_path /path/to/checkpoint.pt
```

To train a new checkpoint on the initial task, `--ckpt_path` can be omitted when
using command line arguments or when using the `PATHS` dictionary in `config.py`
the path specified under `ckpt_path` needs to not exist.
```
python main.py --dataroot /path/to/dataroot
```
To train a new classification head the existing checkpoint, by default found in the
`./checkpoints/` directory, can be removed/renamed or the `--cls_ckpt_path` argument
can be used to specify a path to the new checkpoint, if that path leads to a checkpoint
that does not exist a new checkpoint will be trained and saved at that location.

