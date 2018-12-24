"""
the error value of a move is not the current engine evaluation,
but instead the slope at which the position is deteriorating.

Given the starting, current, and ending outcome percentile vectors,
we can determine where the inaccurate moves were made.

"""

from functions import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
	open = [.346, .257, .397]   # from CCRL

