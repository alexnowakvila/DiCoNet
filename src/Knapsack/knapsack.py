
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os
import sys
import time
import math
import argparse
import time
from subprocess import run, PIPE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex

from model import Split_GNN
from data_generator import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--save_file_path', nargs='?', const=1, type=str, default='')
parser.add_argument('--load_file_path', nargs='?', const=1, type=str, default='')
parser.add_argument('--dataset_path', nargs='?', const=1, type=str, default='')
parser.add_argument('--solver_path', nargs='?', const=1, type=str, default='')
parser.add_argument('--logs_path', nargs='?', const=1, type=str, default='')
parser.add_argument('--splits', nargs='?', const=1, type=int, default=3)
parser.add_argument('--N', nargs='?', const=1, type=int, default=50)
args = parser.parse_args()


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

template_train1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} '
template_train2 = '{:<10} {:<10} {:<10.3f} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.3f} '
template_train3 = '{:<10} {:<10} {:<10} {:<10.5f} {:<10.5f} {:<10.5f} {:<10} \n'
info_train = ['TRAIN', 'iteration', 'loss', 'weight', 'opt', 'trivial', 'elapsed']


if args.logs_path != '':
    class Logger2(object):
        def __init__(self, path):
            self.terminal = sys.stdout
            self.log = open(path, 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass    

    sys.stdout = Logger2(os.path.join(args.logs_path, 'output.txt'))

class Logger():
    dicc = {}
    def add(self, name, val):
        if name in self.dicc:
            lis = self.dicc[name]
            lis.append(val)
            self.dicc[name] = lis
        else:
            self.dicc[name] = [val]
    def empty(self, name):
        self.dicc[name] = []
    def empty_all(self):
        self.dicc = {}
    def get(self, name):
        return np.asarray(self.dicc[name])
    
def plot_train_logs(path_train_plot, weight, trivial, opt):
    plt.figure(1, figsize=(8,6))
    plt.clf()
    iters = range(len(weight))
    plt.plot(iters, opt-trivial, 'g')
    plt.plot(iters, opt-weight, 'b')
    plt.xlabel('iterations')
    plt.ylabel('Average Mean Opt-Reward')
    plt.title('Average Mean Opt-Reward Training')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    plt.savefig(path_train_plot)


def create_input(weights, volumes, C, masks):
    bs, N = weights.size()
    Ns, NNs = masks
    weights = weights*NNs
    maxs = weights.max(1, True)[0].expand_as(weights)
    weights = weights / maxs.clamp(min=1e-6)
    volumes = volumes * NNs
    volumes = volumes / C.clamp(min=1e-4).unsqueeze(1).expand_as(volumes)
    NNNs = torch.bmm(NNs.unsqueeze(2),NNs.unsqueeze(1))
    W = torch.zeros(bs, N, N, 2).type(dtype)
    W[:,:,:,0] = torch.eye(N).type(dtype).unsqueeze(0).expand(bs,N,N) * NNNs
    #W[:,:,:,1] = W[:,:,:,0]*weights.unsqueeze(2).expand(bs,N,N) * NNNs
    #W[:,:,:,2] = W[:,:,:,0]*volumes.unsqueeze(2).expand(bs,N,N) * NNNs
    W[:,:,:,1] = NNNs / Ns.float().clamp(min=1).unsqueeze(1).unsqueeze(2).expand_as(NNNs)
    W = Variable(W)
    prods = weights / volumes.clamp(min=1e-6)
    x = Variable(torch.cat((weights.unsqueeze(2), volumes.unsqueeze(2), prods.unsqueeze(2)), 2))
    Y = W[:,:,:,1].clone()
    return W, x, Y

def trivial_algorithm(weights, volumes, C):
    bs, N = weights.size()
    scores = weights / volumes.clamp(min=1e-6)
    _, inds = scores.sort(1, descending=True)
    ord_volumes = volumes.gather(1, inds)
    sums = torch.zeros(bs, N).type(dtype)
    sums[:,0] = ord_volumes[:,0]
    for i in range(1, N):
        sums[:,i] = sums[:,i-1] + ord_volumes[:,i]
    mask_chosen = sums <= C.unsqueeze(1).expand_as(sums)
    w = (weights.gather(1, inds) * mask_chosen.float()).sum(1)
    return w

def decide(prob_scores, volumes, C):
    bs, N = prob_scores.size()
    _, inds = prob_scores.sort(1, descending=True)
    ord_volumes = volumes.gather(1, inds)
    sums = torch.zeros(bs, N).type(dtype)
    sums[:,0] = ord_volumes[:,0]
    for i in range(1, N):
        sums[:,i] = sums[:,i-1] + ord_volumes[:,i]
    mask_chosen = sums <= C.unsqueeze(1).expand_as(sums)
    chosen = mask_chosen.long() * inds
    return mask_chosen, inds

def decide2(prob_scores, volumes, C, n_samples):
    bs, N = prob_scores.size()
    sample = torch.zeros(bs, N).type(dtype)
    nn.init.uniform(sample)
    sample = sample*F.softmax(prob_scores).data
    _, inds = sample.sort(1, descending=True)
    volumes = volumes.gather(1, inds)
    M = torch.tril(torch.ones(N,N).type(dtype)).unsqueeze(0).expand(bs,N,N)
    sums = torch.bmm(M,volumes.unsqueeze(2)).squeeze(2)
    mask_chosen = sums <= C.unsqueeze(1).expand_as(sums)
    return mask_chosen, inds

def rearange(prob_scores, weights, volumes, masks, inds, mask_chosen):
    prob_scores = prob_scores.gather(1, Variable(inds))
    weights = weights.gather(1, inds)
    volumes = volumes.gather(1, inds)
    Ns, NNs = masks
    NNs = NNs.gather(1, inds)
    NNs *= (1-mask_chosen).float()
    Ns = NNs.sum(1)
    return prob_scores, weights, volumes, (Ns, NNs)

def compute_stuff(mask_chosen, scores, weights, volumes):
    bs = weights.size(0)
    mask_chosen = Variable(mask_chosen.float())
    probs = 1e-6 + (1-2e-6) * F.softmax(scores)
    lgp = (torch.log(probs) * mask_chosen + torch.log(1-probs) * (1-mask_chosen)).sum(1)
    w = (weights * mask_chosen).sum(1)
    v = (volumes * mask_chosen).sum(1)
    return lgp, w, v

def execute(Knap, scales, weights, volumes, C, masks, n_samples_base, mode='test'):
    Ns, NNs = masks
    bs = C.size(0)
    C2 = C
    w_total = Variable(torch.zeros(bs).type(dtype))
    if mode == 'train':
        loss_total = Variable(torch.zeros(1).type(dtype))
    for s in range(scales):
        n_samples = n_samples_base
        last_scale = s == scales-1
        if last_scale:
            C2 = C
        else:
            C2 = C / 2
        input = create_input(weights.data, volumes.data, C2.data, masks)
        prob_scores = Knap(input)
        if mode == 'train':
            ws = Variable(torch.zeros(bs, n_samples).type(dtype))
            lgps = Variable(torch.zeros(bs, n_samples).type(dtype))
            for i in range(n_samples):
                mask_chosen2, inds2 = decide2(prob_scores.data, volumes.data, C2.data, n_samples)
                prob_scores2, weights2, volumes2, masks2 = rearange(prob_scores, weights, volumes, masks, inds2, mask_chosen2)
                lgp, w, v = compute_stuff(mask_chosen2, prob_scores2, weights2, volumes2)
                C3 = C - v
                if not last_scale:
                    w_rec, C_rec, _ = execute(Knap, scales-1-s, weights, volumes, C3, masks, n_samples)
                    w = w + w_rec
                    C3 = C_rec
                ws[:,i] = w
                lgps[:,i] = lgp
            b = ws.mean(1,True).expand_as(ws)
            loss = -(lgps * Variable((ws-b).data)).sum(1).sum(0) / n_samples / bs
            loss_total = loss_total + loss
        mask_chosen, inds = decide(prob_scores.data, volumes.data, C2.data)
        prob_scores, weights, volumes, masks = rearange(prob_scores, weights, volumes, masks, inds, mask_chosen) #reordenar inputs i actualitzar masks
        _, w, v = compute_stuff(mask_chosen, prob_scores, weights, volumes)
        w_total = w_total + w
        C = C - v
    if mode=='train':
        return loss_total, w_total, C, masks
    else:
        return w_total, C, masks
    

def save_model(path, model):
    torch.save(model.state_dict(), path)
    print('Model Saved.')

def load_model(path, model):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print('GNN successfully loaded from {}'.format(path))
        return model
    else:
        raise ValueError('Parameter path {} does not exist.'.format(path))

if __name__ == '__main__':
    
    N = args.N
    num_examples_train = 20000
    num_examples_test = 1000
    mult = N // 50
    C_min, C_max = (10*mult,15*mult)
    clip_grad_norm = 40.0
    batch_size = 500
    num_features = 32
    num_layers = 5
    n_samples = 20
    scales = args.splits+1
    if N >= 200:
        num_examples_test = 100
        batch_size = 100
    
    test = args.test
    
    gen = Generator(args.dataset_path, args.solver_path, num_examples_train, num_examples_test, N, C_min, C_max, test=test)
    gen.load_dataset()
    num_iterations = 100000
    
    Knap = Split_GNN(num_features, num_layers, 3, dim_input=3)
    if args.load_file_path != '':
        Knap = load_model(args.load_file_path, Knap)
    optimizer = optim.Adamax(Knap.parameters(), lr=1e-3)
    
    log = Logger()
    log2 = Logger()
    path_train_plot = os.path.join(args.logs_path, 'training.png')
    
    if test:
        num_iterations = num_examples_test // batch_size
    
    start = time.time()
    for it in range(num_iterations):
        batch = gen.sample_batch(batch_size, is_training=not test, it=it)
        weights, volumes, C, OptW, OptV, is_chosen_opt = batch
        
        bs, N = weights.size()
        Ns = torch.ones(batch_size).type(dtype_l)*N
        NNs = Ns.float().unsqueeze(1).expand(bs,N)
        NNs = torch.ge(NNs, torch.arange(1,N+1).type(dtype).unsqueeze(0).expand(bs,N)).float()
        
        if test:
            loss = Variable(torch.zeros(1).type(dtype))
            w, c, (Ns, NNs) = execute(Knap, scales, weights, volumes, C,(Ns, NNs),  n_samples, 'test')
        else:
            loss, w, c, (Ns, NNs) = execute(Knap, scales, weights, volumes, C,(Ns, NNs),  n_samples, 'train')
        
        trivial_w = trivial_algorithm(weights.data, volumes.data, C.data)
        
        
        
        if not test:
            Knap.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(Knap.parameters(), clip_grad_norm)
            optimizer.step()
        
        log.add('w', w.data.mean())
        log.add('tw', trivial_w.mean())
        log.add('opt', OptW.data.mean())
        log.add('loss', loss.data.cpu().numpy()[0])
        log.add('ratioW', (OptW.data/w.data).mean())
        log.add('ratioT', (OptW.data/trivial_w).mean())

        if not test:
            if it%50 == 0:
                elapsed = time.time() - start
                loss = log.get('loss').mean()
                w = log.get('w').mean()
                opt = log.get('opt').mean()
                tw = log.get('tw').mean()
                ratioW = log.get('ratioW').mean()
                ratioT = log.get('ratioT').mean()

                out1 = ['---', it, loss, w, opt, tw, elapsed]
                out2 = ['', '', '', ratioW, opt/opt, ratioT, '']
                print(template_train1.format(*info_train))
                print(template_train2.format(*out1))
                print(template_train3.format(*out2))
                
                if it > 0:
                    log2.add('W',w)
                    log2.add('TW',tw)
                    log2.add('OPT',opt)
                    if args.logs_path != '':
                        plot_train_logs(path_train_plot, log2.get('W'),log2.get('TW'),log2.get('OPT'))
                
                log.empty_all()
                start = time.time()
            if it%1000 == 0 and it >= 0:
                if args.save_file_path != '':
                    save_model(args.save_file_path, Knap)
    if test:
        elapsed = time.time() - start
        loss = log.get('loss').mean()
        w = log.get('w').mean()
        opt = log.get('opt').mean()
        tw = log.get('tw').mean()
        ratioW = log.get('ratioW').mean()
        ratioT = log.get('ratioT').mean()

        out1 = ['---', it, loss, w, opt, tw, elapsed]
        out2 = ['', '', '', ratioW, opt/opt, ratioT, '']
        print(template_train1.format(*info_train))
        print(template_train2.format(*out1))
        print(template_train3.format(*out2))
            




















