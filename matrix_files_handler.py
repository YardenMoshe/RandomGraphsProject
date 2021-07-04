import os

import numpy
import numpy as np
from distribution import Distribution

from datetime import datetime

subdir = ""


def save(weights, dist, N, extra_details="", subdir_prefix="saved/"):
    current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    dist = (str(dist)).replace("Distribution.", "")
    subdir = "{0}/{1}".format(subdir_prefix, dist)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    file_name = "{0}/{1}_{2}___{3}.npy".format(subdir, str(N), str(extra_details),str(current_time))
    file = open(file_name, 'wb')
    numpy.save(file, weights)


def load(file_name):
    file = open(file_name, 'rb')
    s= numpy.load(file)
    print(s)
    return s
