import argparse

from __init__ import *
from src.utils import config
from src.utils.tools import create_logger
from src.ML.models import Models

if __name__ == '__main__':
    m = Models(config.root_path + '/model/ml_model/' + "emsemble")
    m.unbalance_helper(imbalance_method="ensemble",
                       search_method='grid')