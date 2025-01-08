# flake8: noqa
import os.path as osp

from basicsr.train import train_pipeline
from drct.archs import *
from drct.data import *
from drct.models import *

# import hat.archs
# import hat.data
# import hat.models


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
