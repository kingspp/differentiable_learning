import os
from abc import ABCMeta
from munch import Munch

DIFFLR_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
DIFFLR_DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/'
DIFFLR_EXPERIMENTS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/experiments/'
DIFFLR_EXPERIMENTS_RUNS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/experiments/runs'




class Singleton(type, metaclass=ABCMeta):
    """
    Singleton Class
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Config(Munch, metaclass=Singleton):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)


CONFIG = Config()
CONFIG.DRY_RUN = False

    
    
    