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
    def __init__(self, path_dataset, num_examples_train, num_examples_test, N, clusters, dim):
        self.path_dataset = path_dataset
        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test
        self.data_train = []
        self.data_test = []
        self.N = N
        self.clusters = clusters
        self.dim = dim
    
    
    def gaussian_example(self, N, clusters):
        centers = np.random.uniform(0, 1, [clusters, self.dim])
        per_cl = N // clusters
        Pts = []
        cov = 0.001 * np.eye(self.dim, self.dim)
        target = np.zeros([N])
        for c in range(clusters):
            points = np.random.multivariate_normal(centers[c], cov, per_cl)
            target[c * per_cl: (c + 1) * per_cl] = c
            Pts.append(points)
        points = np.reshape(Pts, [-1, self.dim])
        rand_perm = np.random.permutation(N)
        points = points[rand_perm]
        target = target[rand_perm]
        return points, target
    
    def uniform_example(self, N):
        points = np.random.uniform(0, 1, 2*N)
        points = np.reshape(points, (2, N))
        target = np.zeros(N)
        return points, target
    
    def compute_example(self):
        example = {}
        if self.clusters > 0:
            points, target = self.gaussian_example(self.N, self.clusters)
        else:
            points, target = self.uniform_example(self.N)
        example['points'] = points
        example['target'] = target
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
        # load train dataset
        filename = 'KMEANS{}_clusters{}_dim{}_train.np'.format(self.N, self.clusters, self.dim)
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
        filename = 'KMEANS{}_clusters{}_dim{}_test.np'.format(self.N, self.clusters, self.dim)
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
        points = torch.zeros(num_samples, self.N, self.dim)
        target = torch.zeros(num_samples, self.N)
        
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
            points[b] = torch.from_numpy(dataset[ind]['points'])
            target[b] = torch.from_numpy(dataset[ind]['target'])
        # wrap as variables
        points = Variable(points, volatile=volatile)
        target = Variable(target, volatile=volatile)
        if cuda:
            return points.cuda(), target.cuda()
        else:
            return points, target

if __name__ == '__main__':
    # Test Generator module
    gen = Generator('/data/folque/dataset/', 20000, 1000, 50, 5, 2)
    gen.load_dataset()
    print(gen.sample_batch(1))
    


