#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
import os
# import dependencies
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
import math
from subprocess import run, PIPE

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

class Generator():
    def __init__(self, path_dataset, num_examples_train, num_examples_test, N, C_min, C_max, test=False):
        self.path_dataset = path_dataset
        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test
        self.data_train = []
        self.data_test = []
        self.N = N
        self.C_min = C_min
        self.C_max = C_max
        self.test = test
        self.Z = 1e4
    
    def solve(self, W, V, C):
        W = np.ceil(W*self.Z).astype(np.int64)
        V = np.ceil(V*self.Z).astype(np.int64)
        C = np.ceil(C*self.Z).astype(np.int64)
        
        def np2string(A):
            return ' '.join(A.astype(str))
        problem_input = str(self.N) + ' ' + str(C) + ' ' + np2string(W) + ' ' + np2string(V) + '\n'
        p = run(['./solver'], stdout=PIPE, input=problem_input, encoding='ascii')
        output = list(map(int, p.stdout.split()))
        OptW = output[-1]
        OptV = output[-2]
        chosen = np.asarray(output[:-2])
        return OptW/self.Z, OptV/self.Z, chosen
        
    
    def compute_example(self):
        example = {}
        
        weights = np.random.uniform(size=self.N)
        volumes = np.random.uniform(size=self.N)
        C = np.random.uniform(low=self.C_min, high=self.C_max)
        
        example['weights'] = np.ceil(weights * self.Z) / self.Z
        example['volumes'] = np.ceil(volumes * self.Z) / self.Z
        example['capacity'] = np.ceil(C * self.Z) / self.Z
        
        OptW, OptV, chosen = self.solve(weights, volumes, C)
        example['OptW'] = OptW
        example['OptV'] = OptV
        chosen = np.flip(chosen, 0)
        example['chosen'] = chosen
        is_chosen = np.zeros(self.N)
        is_chosen[chosen] = 1
        example['is_chosen'] = is_chosen
        
        return example
        

    def create_dataset_train(self):
        for i in range(self.num_examples_train):
            example = self.compute_example()
            self.data_train.append(example)
            if i % 100 == 0:
                print('Train example {} of length {} computed.'
                      .format(i, self.N))

    def create_dataset_test(self):
        for i in range(self.num_examples_test):
            example = self.compute_example()
            self.data_test.append(example)
            if i % 100 == 0:
                print('Test example {} of length {} computed.'
                      .format(i, self.N))

    def load_dataset(self):
        if not self.test:
            # load train dataset
            filename = 'Dataset_Knapsack{}_train_capacity_{}_{}.np'.format(self.N, self.C_min, self.C_max)
            path = os.path.join(self.path_dataset, filename)
            if os.path.exists(path):
                print('Reading training dataset at {}'.format(path))
                self.data_train = np.load(open(path, 'rb'))
            else:
                print('Creating training dataset.')
                self.create_dataset_train()
                print('Saving training datatset at {}'.format(path))
                np.save(open(path, 'wb'), self.data_train)
        # load test dataset
        filename = 'Dataset_Knapsack{}_test_capacity_{}_{}.np'.format(self.N, self.C_min, self.C_max)
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading testing dataset at {}'.format(path))
            self.data_test = np.load(open(path, 'rb'))
        else:
            print('Creating testing dataset.')
            self.create_dataset_test()
            print('Saving testing datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_test)

    def sample_batch(self, num_samples, is_training=True, it=0, cuda=True, volatile=False):
        
        W = torch.zeros(num_samples, self.N)
        V = torch.zeros(num_samples, self.N)
        C = torch.zeros(num_samples)
        OptW = torch.zeros(num_samples)
        OptV = torch.zeros(num_samples)
        #chosen = torch.zeros(num_samples)
        is_chosen = torch.zeros(num_samples, self.N)
        
        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        
        for b in range(num_samples):
            if is_training:
                # random element in the dataset
                ind = np.random.randint(0, len(dataset))
            else:
                ind = it * num_samples + b
            W[b] = torch.from_numpy(dataset[ind]['weights'])
            V[b] = torch.from_numpy(dataset[ind]['volumes'])
            C[b] = dataset[ind]['capacity']
            OptW[b] = dataset[ind]['OptW']
            OptV[b] = dataset[ind]['OptV']
            is_chosen[b] = torch.from_numpy(dataset[ind]['is_chosen'])
        # wrap as variables
        W = Variable(W, volatile=volatile)
        V = Variable(V, volatile=volatile)
        C = Variable(C, volatile=volatile)
        OptW = Variable(OptW, volatile=volatile)
        OptV = Variable(OptV, volatile=volatile)
        is_chosen = Variable(is_chosen, volatile=volatile)
        if cuda:
            return W.cuda(), V.cuda(), C.cuda(), OptW.cuda(), OptV.cuda(), is_chosen.cuda()
        else:
            return W, V, C, OptW, OptV, is_chosen

