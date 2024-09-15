import os
import sys

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ.keys():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import tensorflow as tf

from modules.dataloader import parse_cfg
from modules.nn import ClassifyModel
