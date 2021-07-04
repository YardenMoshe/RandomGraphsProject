import os
import numpy
import uuid

def save(input_graph, dist, extra_details="", subdir_prefix="saved/"):
    N = len(input_graph)
    subdir = "{0}/{1}".format(subdir_prefix, (str(dist)).replace("Distribution.", ""))
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    file_name = "{0}/{1}_{2}_{3}.npy".format(subdir,
                                               str(N),
                                               str(extra_details),
                                               str(uuid.uuid4()))
    file = open(file_name, 'wb')
    numpy.save(file, input_graph)


def load(file_name):
    file = open(file_name, 'rb')
    return numpy.load(file)
