import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from time import time
from collections import defaultdict
from functools import partial
from sklearn.utils import shuffle
import os, glob, skimage, cv2, shutil
from matplotlib import pyplot as plt

import dataUtils
#import keras

# Image dimension
SQUARE_SIDE_LENGTH = 227

# Categories for neural network to learn
categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

