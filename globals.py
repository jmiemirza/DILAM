
VALID_DATASETS = ['cifar10', 'tiny-imagenet', 'imagenet']

SEVERTITIES = [5]

TASKS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# TASKS = TASKS[:4]

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

