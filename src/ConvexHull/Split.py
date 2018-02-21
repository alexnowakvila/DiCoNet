import numpy as np
from scipy import sparse
import csv
from scipy.spatial import ConvexHull

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Pytorch requirements
import unicodedata
import string
import re
import random
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor
dtype_l = torch.cuda.LongTensor
torch.cuda.manual_seed(0)


class SplitLayer(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(SplitLayer, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        W1 = nn.Parameter(torch.randn(hidden_size + input_size,
                          hidden_size))
        self.register_parameter('W1', W1)
        W2 = nn.Parameter(torch.randn(hidden_size + input_size,
                          hidden_size))
        self.register_parameter('W2', W2)
        b = nn.Parameter(torch.randn(hidden_size))
        self.register_parameter('b', b)

    def forward(self, input_n, hidden, phi, nh):
        self.batch_size = input_n.size()[0]
        hidden = torch.cat((hidden, input_n), 2)
        # Aggregate reresentations
        h_conv = torch.div(torch.bmm(phi, hidden), nh)
        hidden = hidden.view(-1, self.hidden_size + self.input_size)
        h_conv = h_conv.view(-1, self.hidden_size + self.input_size)
        # h_conv has shape (batch_size, n, hidden_size + input_size)
        m1 = (torch.mm(hidden, self.W1)
              .view(self.batch_size, -1, self.hidden_size))
        m2 = (torch.mm(h_conv, self.W2)
              .view(self.batch_size, -1, self.hidden_size))
        m3 = self.b.unsqueeze(0).unsqueeze(1).expand_as(m2)
        hidden = torch.sigmoid(m1 + m2 + m3)
        return hidden


class Split(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, n_layers=1):
        super(Split, self).__init__()
        print('Initializing Parameters Split')
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.n = 32
        layers = [SplitLayer(input_size, hidden_size, batch_size)
                  for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.linear_b = nn.Linear(hidden_size, 1, bias=True).type(dtype)

    def forward(self, e, input, mask, scale=0):
        hidden = Variable(torch.randn(self.batch_size, self.n,
                                      self.hidden_size)).type(dtype)
        if scale == 0:
            e = Variable(torch.zeros(self.batch_size, self.n)).type(dtype)
        Phi = self.build_Phi(e, mask)
        N = torch.sum(Phi, 2).squeeze()
        N += (N == 0).type(dtype)  # avoid division by zero
        Nh = N.unsqueeze(2).expand(self.batch_size, self.n,
                                   self.hidden_size + self.input_size)
        # Normalize inputs, important part!
        mask_inp = mask.unsqueeze(2).expand_as(input)
        input_n = self.Normalize_inputs(Phi, input) * mask_inp
        # input_n = input * mask_inp
        for i, layer in enumerate(self.layers):
            hidden = layer(input_n, hidden, Phi, Nh)
        hidden_p = hidden.view(self.batch_size * self.n, self.hidden_size)
        scores = self.linear_b(hidden_p)
        probs = torch.sigmoid(scores).view(self.batch_size, self.n) * mask
        # probs has shape (batch_size, n)
        return scores, probs, input_n, Phi

    def build_Phi(self, e, mask):
        e_rows = e.unsqueeze(1).expand(self.batch_size, self.n, self.n)
        e_cols = e.unsqueeze(2).expand(self.batch_size, self.n, self.n)
        Phi = Variable(torch.abs(e_rows - e_cols).data == 0).type(dtype)
        # mask attention matrix
        mask_rows = mask.unsqueeze(2).expand_as(Phi)
        mask_cols = mask.unsqueeze(1).expand_as(Phi)
        Phi = Phi * mask_rows * mask_cols
        return Phi

    def Normalize_inputs(self, phis, input):
        # phis defines the clusters
        length = phis.size()[1]
        mask = phis.unsqueeze(3).expand(self.batch_size, length,
                                        length, self.input_size)
        inp_masked = mask * input.unsqueeze(1).expand_as(mask)
        N = mask.sum(2)
        N += (N == 0).type(dtype)
        means = inp_masked.sum(2) / N
        dif = inp_masked - mask * means.unsqueeze(2).expand_as(inp_masked)
        var = (dif * dif).sum(2) / N
        var += (var == 0).type(dtype)
        inp_norm = (input - means) / (3 * var.sqrt()) + 0.5
        return inp_norm
