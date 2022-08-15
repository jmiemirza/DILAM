
VALID_DATASETS = ['cifar10', 'imagenet', 'kitti', 'imagenet-mini']

SEVERTITIES = [5]

RAIN_SEVERITIES = ['1mm', '5mm', '17mm', '25mm', '50mm', '75mm', '100mm', '200mm']
FOG_SEVERITIES = ['30m', '40m', '50m', '75m', '150m', '375m', '750m']

TASKS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

KITTI_TASKS = ['fog', 'rain', 'snow']

# TASKS = TASKS[:3]


PATHS = {
    'sl': {
        'Windows': {
            'cifar10': {
                'root': 'X:/thesis/CIFAR-10-C',
                'ckpt': 'X:/thesis/framework/checkpoints/cifar10/wrn/Hendrycks2020AugMixWRN.pt',
            },
            'imagenet-mini': {
                'root': 'X:/thesis/imagenet-mini',
                # 'ckpt': 'X:/thesis/framework/checkpoints/imagenet-mini/res18/initialDefaultLearn.pt',
                'ckpt': 'X:/thesis/res18_imgnet_default.pt',
            },
            'imagenet': {
                'root': 'X:/thesis/imagenet',
                'ckpt': 'X:/thesis/res18_imgnet_default.pt',
            },
        },
        'Linux': {
            'cifar10': {
                'root': '/mnt/linux_data/datasets/cifar10',
                'ckpt': '/home/sleitner/space/framework/checkpoints/cifar10/wrn/Hendrycks2020AugMixWRN.pt',
            },
            'imagenet-mini': {
                'root': '/mnt/linux_data/datasets/imagenet-mini',
                # 'ckpt': '/home/sleitner/space/framework/checkpoints/imagenet-mini/res18/initialDefaultLearn.pt',
                'ckpt': '/home/sleitner/space/framework/res18_imgnet_default.pt',
            },
            'imagenet': {
                'root': '/mnt/linux_data/datasets/imagenet',
                'ckpt': '/home/sleitner/space/framework/res18_imgnet_default.pt',
            },
        },
    },
}


LOGGER_CFG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(name)s - %(levelname)s] %(message)s'
        },
        'timestamped': {
            'format': '%(asctime)s [%(name)s - %(levelname)s] %(message)s'
        },
        'minimal': {
            'format': '[%(name)s] %(message)s'
        }
    },
    'filters': {
        'name': {
            '()': 'globals.ContextFilter'
        }
    },
    'handlers': {
        'console_handler': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'minimal',
            'stream': 'ext://sys.stdout',
            'filters': ['name']
        },
        'file_handler': {
            'level': 'DEBUG',
            'formatter': 'minimal',
            'class': 'logging.FileHandler',
            'filename': 'log.txt',
            'mode': 'a',
            'filters': ['name']
        },
    },
    'loggers': {
        '': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'WARNING',
            'propagate': False
        },

        'MAIN': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'DEBUG',
            'propagate': False
        },
        'MAIN.DISC': {},
        'MAIN.DUA': {},
        'MAIN.DATA': {},
        'MAIN.RESULTS': {},

        'BASELINE': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'DEBUG',
            'propagate': False
        },
        'BASELINE.FREEZING': {},
        'BASELINE.DISJOINT': {},
        'BASELINE.JOINT_TRAINING': {},
        'BASELINE.SOURCE_ONLY': {},
        'BASELINE.FINE_TUNING': {},

        'TRAINING': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}


class ContextFilter:
    def filter(self, record):
        split_name = record.name.split('.', 1)
        if split_name[0] == 'BASELINE' or split_name[0] == 'MAIN':
            if len(split_name) > 1:
                record.name = split_name[1]
        return True

