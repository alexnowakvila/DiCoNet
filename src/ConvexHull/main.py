#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
from data_generator import Generator
from Logger import Logger
from DCN import DivideAndConquerNetwork
import utils
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


parser = argparse.ArgumentParser()

parser.add_argument('--dynamic', action='store_true',
                    help='Use DCN. If not set, run baseline. (depth=0)')
parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=1048576)
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=4096)
parser.add_argument('--num_epochs', nargs='?', const=1, type=int, default=35)
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=128)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
parser.add_argument('--path_dataset', nargs='?', const=1, type=str, default='')
parser.add_argument('--path', nargs='?', const=1, type=str, default='')


###############################################################################
#                             PtrNet arguments                                #
###############################################################################

parser.add_argument('--load_merge', nargs='?', const=1, type=str)
parser.add_argument('--num_units_merge', nargs='?', const=1, type=int,
                    default=512)
parser.add_argument('--rnn_layers', nargs='?', const=1, type=int, default=1)
parser.add_argument('--grad_clip_merge', nargs='?', const=1,
                    type=float, default=2.0)
parser.add_argument('--merge_sample', action='store_true',
                    help='Sample inputs to connect different PtrNets when'
                    'cascading PtrNets')

###############################################################################
#                              split arguments                                #
###############################################################################

parser.add_argument('--load_split', nargs='?', const=1, type=str)
parser.add_argument('--split_layers', nargs='?', const=1, type=int, default=5)
parser.add_argument('--num_units_split', nargs='?', const=1, type=int,
                    default=15)
parser.add_argument('--grad_clip_split', nargs='?', const=1,
                    type=float, default=40.0)
parser.add_argument('--regularize_split', action='store_true',
                    help='regularize the split training with variance prior.')
parser.add_argument('--beta', nargs='?', const=1, type=float, default=1.0)
parser.add_argument('--random_split', action='store_true',
                    help='Do not train split. Take uniform random samples')

args = parser.parse_args()

num_epochs = args.num_epochs
batch_size = args.batch_size
input_size = 2

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)


def train(DCN, logger, gen):
    Loss, Loss_reg = [], []
    Discard_rates = [[[] for jj in range(ii)] for ii in gen.scales['train']]
    Accuracies_tr = [[] for ii in gen.scales['train']]
    Accuracies_te = [[] for ii in gen.scales['test']]
    iterations_tr = int(gen.num_examples_train / batch_size)
    for epoch in range(num_epochs):
        lr = DCN.upd_learning_rate(epoch)
        for it in range(iterations_tr):
            losses = 0.0
            variances = 0.0
            start = time.time()
            for i, scales in enumerate(gen.scales['train']):
                # depth tells how many times the dynamic model will be unrolled
                depth = 1
                if args.dynamic:
                    depth = scales
                _, length = gen.compute_length(scales, mode='train')
                DCN.merge.n, DCN.split.n = [length] * 2
                input, tar = gen.get_batch(batch=it, scales=scales,
                                           mode='train')
                # forward DCN
                out = DCN(input, tar, length, depth, it=it, epoch=epoch,
                          random_split=args.random_split,
                          mode='train', dynamic=args.dynamic)
                Phis, Inputs_N, target, Perms, e, loss, pg_loss, var = out
                # backward DCN
                if args.dynamic:
                    DCN.step_split(pg_loss, var,
                                   regularize=args.regularize_split)
                DCN.step_merge(loss)
                losses += loss
                variances += var
                # Save discard rates
                discard_rates = utils.compute_dicard_rate(Perms)
                for sc in range(len(Perms) - 1):
                    Discard_rates[i][sc].append(discard_rates[sc])
                Accuracies_tr[i].append(utils.compute_accuracy(Perms[-1],
                                                               target))
            losses /= len(gen.scales['train'])
            variances /= len(gen.scales['train'])
            # optimizer.step()
            Loss.append(losses.data.cpu().numpy())
            Loss_reg.append(variances.data.cpu().numpy())
            elapsed = time.time() - start
            if it % 64 == 0:
                print('TRAINING --> | Epoch {} | Batch {} / {} | Loss {} |'
                      ' Accuracy Train {} | Elapsed {} | dynamic {} |'
                      ' Learning Rate {}'
                      .format(epoch, it, iterations_tr,
                              losses.data.cpu().numpy(), Accuracies_tr[-1][-1],
                              elapsed, args.dynamic, lr))
                logger.plot_Phis_sparsity(Phis, fig=0)
                logger.plot_norm_points(Inputs_N, e, Perms,
                                        gen.scales['train'][-1], fig=1)
                logger.plot_losses(Loss, Loss_reg, fig=2)
                logger.plot_rates(Discard_rates, fig=3)
                logger.plot_accuracies(Accuracies_tr,
                                       scales=gen.scales['train'],
                                       mode='train', fig=4)
                # keep track of output examples
                # for perm in Perms:
                #     print('permutation', perm.data[0, 1:].cpu().numpy())
                # print('target', target[0].data.cpu().numpy())
            if it % 1000 == 1000 - 1:
                print('Saving model parameters')
                DCN.save_split(args.path)
                DCN.save_merge(args.path)
                accuracies_test = test(DCN, gen)
                for i, scales in enumerate(gen.scales['test']):
                    Accuracies_te[i].append(accuracies_test[i])
                show_accs = ' | '.join(['Acc Len {} -> {}'
                                        .format(scales, Accuracies_te[i][-1])
                                        for i, scales
                                        in enumerate(gen.scales['test'])])
                print('TESTING --> epoch {} '.format(epoch) + show_accs)
                logger.plot_accuracies(Accuracies_te,
                                       scales=gen.scales['test'],
                                       mode='test', fig=2)
                logger.save_results(Loss, Accuracies_te, Discard_rates)


def test(DCN, gen):
    accuracies_test = [[] for ii in gen.scales['test']]
    iterations_te = int(gen.num_examples_test / batch_size)
    for it in range(iterations_te):
        for i, scales in enumerate(gen.scales['test']):
            # depth tells how many times the dynamic model will be unrolled
            depth = 1
            if args.dynamic:
                depth = scales
            _, length = gen.compute_length(scales, mode='test')
            DCN.merge.n, DCN.split.n = [length] * 2
            input, tar = gen.get_batch(batch=it, scales=scales,
                                       mode='test')
            # forward DCN
            out = DCN(input, tar, length, depth, it=it,
                      random_split=args.random_split,
                      mode='test', dynamic=args.dynamic)
            Phis, Inputs_N, target, Perms, e, loss, pg_loss, var = out
            acc = utils.compute_accuracy(Perms[-1], target)
            accuracies_test[i].append(acc)
            print(sum(accuracies_test[i]) / float(it + 1))
    accuracies_test = [sum(accs) / iterations_te
                       for accs in accuracies_test]
    print('acc test:', accuracies_test)
    return accuracies_test

if __name__ == '__main__':
    logger = Logger(args.path)
    logger.write_settings(args)
    DCN = DivideAndConquerNetwork(input_size, args.batch_size,
                                  args.num_units_merge, args.rnn_layers,
                                  args.grad_clip_merge,
                                  args.num_units_split, args.split_layers,
                                  args.grad_clip_split, beta=args.beta)
    if args.load_split is not None:
        DCN.load_split(args.load_split)
    if args.load_merge is not None:
        DCN.load_merge(args.load_merge)
    if torch.cuda.is_available():
        DCN.cuda()
    gen = Generator(args.num_examples_train, args.num_examples_test,
                    args.path_dataset, args.batch_size)
    gen.load_dataset()
    DCN.batch_size = args.batch_size
    DCN.merge.batch_size = args.batch_size
    DCN.split.batch_size = args.batch_size
    if args.mode == 'train':
        train(DCN, logger, gen)
    elif args.mode == 'test':
        test(DCN, gen)
