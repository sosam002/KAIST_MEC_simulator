import logging
import logging.config


def initialize_mecs():
    init_logger()


def init_logger():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s %(levelname)-5s %(name)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'default',
                'stream': 'ext://sys.stderr',
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'filename': 'mecs.log',
                'mode': 'w',
                'formatter': 'default'
            },
            'null': {
                'class': 'logging.NullHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
            },
        },
    })
