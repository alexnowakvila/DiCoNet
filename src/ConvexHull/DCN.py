#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
from Merge import PtrNet_tanh as Merge
from Split import Split
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import init
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


class DivideAndConquerNetwork(nn.Module):
    def __init__(
                 self, input_size, batch_size,
                 num_units_merge, rnn_layers, grad_clip_merge,
                 num_units_split, split_layers, grad_clip_split, beta=1.0
                 ):
        super(DivideAndConquerNetwork, self).__init__()
        # General
        self.input_size = input_size
        self.batch_size = batch_size
        # Merge
        self.num_units_merge = num_units_merge
        self.rnn_layers = rnn_layers
        self.merge = Merge(input_size, num_units_merge, batch_size)
        # Split
        self.num_units_split = num_units_split
        self.split_layers = split_layers
        self.beta = beta
        self.split = Split(input_size, num_units_split,
                           batch_size, split_layers)
        # Training
        self.grad_clip_split = grad_clip_split
        self.optim_split = optim.RMSprop(self.split.parameters())
        self.grad_clip_merge = grad_clip_merge
        self.optim_merge = optim.Adam(self.merge.parameters())

    ###########################################################################
    #                           Load Parameters                               #
    ###########################################################################

    def load_split(self, path):
        path = os.path.join(path, 'parameters/params_split.pt')
        print('Reading split parameters from {}'.format(path))
        if os.path.exists(path):
            self.split = torch.load(path)
        else:
            raise ValueError('path for split {} does not exist'.format(path))

    def load_merge(self, path):
        path = os.path.join(path, 'parameters/params_ptr.pt')
        print('Reading merge parameters from {}'.format(path))
        if os.path.exists(path):
            self.merge = torch.load(path)
        else:
            raise ValueError('path for merge {} does not exist'.format(path))

    ###########################################################################
    #                           Save Parameters                               #
    ###########################################################################

    def save_split(self, path):
        directory = os.path.join(path, 'parameters/')
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        path = os.path.join(directory, 'params_split.pt')
        torch.save(self.split, path)

    def save_merge(self, path):
        directory = os.path.join(path, 'parameters/')
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        path = os.path.join(directory, 'params_ptr.pt')
        torch.save(self.merge, path)

    ###########################################################################
    #                        Optimization Steps                               #
    ###########################################################################

    def upd_learning_rate(self, epoch):
        # split
        lr = 0.01 / float(epoch + 1)
        self.optim_split = optim.RMSprop(self.split.parameters(), lr=lr)
        # merge
        lr = 0.001 / float(epoch + 1)
        self.optim_merge = optim.Adam(self.merge.parameters(), lr=lr)
        return lr

    def step_split(self, cost, variances, regularize=False):
        self.split.zero_grad()
        loss = cost
        if regularize:
            loss -= self.beta * variances
        loss.backward(retain_variables=True)
        nn.utils.clip_grad_norm(self.split.parameters(), self.grad_clip_split)
        self.optim_split.step()

    def step_merge(self, loss):
        self.merge.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.merge.parameters(), self.grad_clip_merge)
        self.optim_merge.step()

    ###########################################################################
    #                             Split Phase                                 #
    ###########################################################################

    def log_probabilities(self, Bs, Samples, mask, depth):
        LogProbs = []
        lengths = mask.sum(1)
        for scale in range(depth):
            probs = Bs[scale]
            sample = Samples[scale]
            probs_act = probs * sample + (1 - probs) * (1 - sample)
            logprobs = torch.log(probs_act + (1 - mask) + 1e-6)
            logprobs = logprobs.sum(1) / lengths
            LogProbs.append(logprobs)
        return LogProbs

    def compute_variance(self, probs, mask):
        N = mask.sum(1)
        mean = torch.sum(probs, 1) / N
        mean_squared = mean * mean
        mean_std = torch.mean(torch.sum(probs*probs, 1) / N -
                              mean_squared, 0)
        return mean_std.squeeze()

    def fwd_split(self, input, batch, depth,
                  random_split=False, mode='train', epoch=0):
        length = self.split.n
        var = 0.0
        # Iterate over scales
        e = Variable(torch.zeros(self.batch_size, length)).type(dtype)
        mask = (input[:, :, 0] >= 0).type(dtype).squeeze()
        Phis, Bs, Inputs_N, Samples = ([] for ii in range(4))
        for scale in range(depth):
            logits, probs, input_n, Phi = self.split(e, input,
                                                     mask, scale=scale)
            # Sample from probabilities and update embeddings
            if random_split:
                rand = (Variable(torch.zeros(self.batch_size, length))
                        .type(dtype))
                init.uniform(rand)
                sample = (rand > 0.5).type(dtype)
            else:
                rand = (Variable(torch.zeros(self.batch_size, length))
                        .type(dtype))
                init.uniform(rand)
                sample = (probs > rand).type(dtype)
            e = 2 * e + sample
            # Appends
            Samples.append(sample)
            Phis.append(Phi)
            Bs.append(probs)
            Inputs_N.append(input_n)
            # variance of bernouilli probabilities
            var += self.compute_variance(probs, mask)
        # computes log probabilities of binary actions for the policy gradient
        Log_Probs = self.log_probabilities(Bs, Samples, mask, depth)
        # pad embeddings with infinity to not affect embeddings argsort
        infty = 1e6
        e = e * mask + (1 - mask) * infty
        return var, Phis, Bs, Inputs_N, e, Log_Probs

    ###########################################################################
    #                             Merge Phase                                 #
    ###########################################################################

    def eliminate_rows(self, prob_sc, ind, phis):
        """ eliminate rows of phis and prob_matrix scale """
        length = prob_sc.size()[1]
        mask = (prob_sc[:, :, 0] > 0.85).type(dtype)
        rang = (Variable(torch.range(0, length - 1).unsqueeze(0)
                .expand_as(mask)).
                type(dtype))
        ind_sc = torch.sort(rang * (1-mask) + length * mask, 1)[1]
        # permute prob_sc
        m = mask.unsqueeze(2).expand_as(prob_sc)
        mm = m.clone()
        mm[:, :, 1:] = 0
        prob_sc = (torch.gather(prob_sc * (1 - m) + mm, 1,
                   ind_sc.unsqueeze(2).expand_as(prob_sc)))
        # compose permutations
        ind = torch.gather(ind, 1, ind_sc)
        active = torch.gather(1-mask, 1, ind_sc)
        # permute phis
        active1 = active.unsqueeze(2).expand_as(phis)
        ind1 = ind.unsqueeze(2).expand_as(phis)
        active2 = active.unsqueeze(1).expand_as(phis)
        ind2 = ind.unsqueeze(1).expand_as(phis)
        phis_out = torch.gather(phis, 1, ind1) * active1
        phis_out = torch.gather(phis_out, 2, ind2) * active2
        return prob_sc, ind, phis_out, active

    def sort_by_embeddings(self, Phis, Inputs_N, e):
        ind = torch.sort(e, 1)[1].squeeze()
        for i, phis in enumerate(Phis):
            # rearange phis
            phis_out = (torch.gather(Phis[i], 1, ind.unsqueeze(2)
                        .expand_as(phis)))
            Phis[i] = (torch.gather(phis_out, 2, ind.unsqueeze(1)
                       .expand_as(phis)))
            # rearange inputs
            Inputs_N[i] = torch.gather(Inputs_N[i], 1,
                                       ind.unsqueeze(2).expand_as(Inputs_N[i]))
        return Phis, Inputs_N

    def reindex_target(self, target, e):
        """ Reindex target by embedding to be coherent. We have to invert
        a permutation and add some padding to do it correctly. """
        ind = torch.sort(e, 1)[1].squeeze()
        first = Variable(torch.zeros(self.batch_size, 1)).type(dtype_l)
        ind = torch.cat((first, ind + 1), 1)
        # target = new_target(ind) -> new_target = target(ind_inv)
        # invert permutation
        ind_inv = torch.sort(ind, 1)[1]
        last = np.zeros((self.batch_size, 1))
        target = np.concatenate((target, last), axis=1)
        for example in range(self.batch_size):
            tar = target[example].astype(int)
            ind_inv_n = ind_inv[example].data.cpu().numpy()
            tar = ind_inv_n[tar]
            tar_aux = tar[np.where(tar > 0)[0]]
            argmin = np.argsort(tar_aux)[0]
            tar_aux = np.array(list(tar_aux[argmin:]) + list(tar_aux[:argmin]))
            tar[:tar_aux.shape[0]] = tar_aux
            target[example] = tar
        return target[:, :-1]

    def discretize(self, prob_matrix):
        indexes = torch.max(prob_matrix, 2)[1].squeeze()
        return indexes

    def combine_matrices(self, prob_matrix, prob_matrix_scale,
                         perm, last=False):
        # prob_matrix shape is bs x length x length + 1. Add extra column.
        length = prob_matrix_scale.size()[2]
        first = Variable(torch.zeros([self.batch_size, 1, length])).type(dtype)
        first[:, 0, 0] = 1.0
        prob_matrix_scale = torch.cat((first, prob_matrix_scale), 1)
        # argmax
        new_perm = self.discretize(prob_matrix_scale)
        perm = torch.gather(perm, 1, new_perm)
        # combine
        prob_matrix = torch.bmm(prob_matrix_scale, prob_matrix)
        return prob_matrix, prob_matrix_scale, perm

    def outputs(self, input, prob_matrix, perm):
        hard_output = (torch.gather(input, 1, perm.unsqueeze(2)
                       .expand_as(input)))
        # soft argmax
        soft_output = torch.bmm(prob_matrix, input)
        return hard_output, soft_output

    def fwd_merge(self, Inputs_N, target, Phis, Bs, lp,
                  batch, depth, mode='train', epoch=0):
        # Flow backwards
        Phis, Bs, Inputs_N = Phis[::-1], Bs[::-1], Inputs_N[::-1]
        length = self.merge.n
        perm = (torch.range(0.0, length)
                .unsqueeze(0).expand(self.batch_size, length + 1))
        perm = Variable(perm, requires_grad=False).type(dtype_l)
        ind = perm[:, :-1].clone()
        prob_matrix = Variable(torch.eye(length + 1)).type(dtype)
        prob_matrix = prob_matrix.unsqueeze(0).expand(self.batch_size,
                                                      length + 1, length + 1)
        # concatenate pad_token to input
        pad_token = (self.merge.pad_token[:-1].unsqueeze(0)
                     .expand(self.batch_size, 1, self.input_size))
        input = torch.cat((pad_token, Inputs_N[0]), 1)
        phis = Phis[0]
        input_target = torch.cat((pad_token, Inputs_N[-1]), 1)
        input_scale = input
        input_norm = input_scale
        Perms = [perm]
        Points = [input_scale]
        for i, scale in enumerate(range(depth)):
            if scale < depth - 1:
                # fine scales
                prob_sc = self.merge(input_scale, phis)
                input_norm = torch.cat((pad_token, Inputs_N[scale + 1]), 1)
                phis = Phis[scale + 1]
                prob_sc, ind, phis, _ = self.eliminate_rows(prob_sc, ind, phis)
                comb = self.combine_matrices(prob_matrix, prob_sc, perm,
                                             last=False)
                prob_matrix, _, perm = comb
                # postprocess before feeding to next scale
                hard_out, soft_out = self.outputs(input_norm,
                                                  prob_matrix, perm)
                input_scale = hard_out
            else:
                # coarsest scale
                if mode == 'test':
                    prob_sc = self.merge(input_scale, phis,
                                         input_target=None,
                                         target=None)
                else:
                    prob_sc = self.merge(input_scale, phis,
                                         input_target=input_target,
                                         target=target)
                comb = self.combine_matrices(prob_matrix, prob_sc, perm,
                                             last=True)
                prob_matrix, prob_sc, perm = comb
                hard_out, soft_out = self.outputs(input, prob_matrix, perm)
                loss, pg_loss = self.merge.compute_loss(prob_matrix, target,
                                                        lp=lp)
            Perms.append(perm)
            Points.append(input_norm)
        return loss, pg_loss, Perms

    ###########################################################################
    #                            Forward pass                                 #
    ###########################################################################

    def forward(self, input, tar, length, depth, it=0, epoch=0,
                random_split=False, mode='train', dynamic=False):
        self.merge.n, self.split.n = [length] * 2
        input = (Variable(torch.from_numpy(input), requires_grad=False)
                 .type(dtype))
        # forward split
        out_split = self.fwd_split(input, it, depth, random_split=random_split,
                                   mode=mode, epoch=epoch)
        var, Phis, Bs, Inputs_N, e, lp = out_split
        # reindex at the leaves of the computation tree
        if dynamic:
            Phis, Inputs_N = self.sort_by_embeddings(Phis, Inputs_N, e)
            tar = self.reindex_target(tar, e)
        target = (Variable(torch.from_numpy(tar), requires_grad=False)
                  .type(dtype_l))
        # forward merge
        out_merge = self.fwd_merge(Inputs_N, target, Phis, Bs, lp, it, depth,
                                   mode=mode, epoch=epoch)
        loss, pg_loss, Perms = out_merge
        return Phis, Inputs_N, target, Perms, e, loss, pg_loss, var
