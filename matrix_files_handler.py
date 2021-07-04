import os
import numpy
from datetime import datetime


def save(weights, dist, N, extra_details="", subdir_prefix="saved/"):
    subdir = "{0}/{1}".format(subdir_prefix, (str(dist)).replace("Distribution.", ""))
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    file_name = "{0}/{1}_{2}___{3}.npy".format(subdir,
                                               str(N),
                                               str(extra_details),
                                               str(current_time))
    file = open(file_name, 'wb')
    numpy.save(file, weights)


def load(file_name):
    file = open(file_name, 'rb')
    return numpy.load(file)
