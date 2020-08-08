import math
import time
import datetime
import random

import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getlocaltime():
    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())

def seconds_to_hms(second):
    return str(datetime.timedelta(seconds=second))