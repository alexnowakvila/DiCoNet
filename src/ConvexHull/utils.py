import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)


def compute_dicard_rate(Perms):
    discard_rates = []
    for perm in Perms[1:]:
        rate = (perm[:, 1:] > 0).type(dtype).sum(1) / perm.size()[1]
        discard_rates.append(rate.mean().data.cpu().numpy())
    return discard_rates


def compute_accuracy(output, target):
    # convert to numpy arrays
    tar = target.data.cpu().numpy()
    out = output.data.cpu().numpy()
    return np.mean(np.all(np.equal(tar, out[:, 1:]), axis=1))
