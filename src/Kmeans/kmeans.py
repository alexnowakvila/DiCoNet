
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os
import sys
import pdb
import time
import math
import argparse
import time
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex

from model import Split_GNN, Split_BaselineGNN
from data_generator import Generator
from data_generator_cifar import Generator as GeneratorCIFAR



parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--save_file', nargs='?', const=1, type=str, default='')
parser.add_argument('--load_file', nargs='?', const=1, type=str, default='')
parser.add_argument('--output_file', nargs='?', const=1, type=str, default='')
parser.add_argument('--dataset', nargs='?', const=1, type=str, default='GM')

parser.add_argument('--dim', nargs='?', const=1, type=int, default=27)
parser.add_argument('--num_examples_train', nargs='?', const=1, type=int, default=10000)
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--N', nargs='?', const=1, type=int, default=200)
parser.add_argument('--K', nargs='?', const=1, type=int, default=2)
parser.add_argument('--clusters', nargs='?', const=1, type=int, default=4)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float, default=40.0)
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=32)
parser.add_argument('--sigma2', nargs='?', const=1, type=float, default=1.)
parser.add_argument('--reg_factor', nargs='?', const=1, type=int, default=0.0)
parser.add_argument('--k_step', nargs='?', const=1, type=int, default=0)
parser.add_argument('--n_samples', nargs='?', const=1, type=int, default=10)
parser.add_argument('--last', action='store_false')
parser.add_argument('--baseline', action='store_true')

###############################################################################
#                             GNN Arguments                                   #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int, default=32)
parser.add_argument('--num_layers', nargs='?', const=1, type=int, default=20)
parser.add_argument('--normalize', action='store_true')

args = parser.parse_args()

args.save_file = '/home/anowak/DCN-for-KMEANS/model/exp1'
# args.load_file = '/home/anowak/DCN-for-KMEANS/model/exp1'

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
info_train = ['TRAIN', 'iteration', 'loss', 'samples', 'best_smpl', 'trivial', 'elapsed']


if args.output_file != '':
    class Logger2(object):
        def __init__(self, path):
            self.terminal = sys.stdout
            self.log = open(path, 'a')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass    

    sys.stdout = Logger2(args.output_file)

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
        return self.dicc[name]

def plot_train_logs(cost_train):
    plt.figure(1, figsize=(8,6))
    plt.clf()
    iters = range(len(cost_train))
    plt.semilogy(iters, cost_train, 'b')
    plt.xlabel('iterations')
    plt.ylabel('Average Mean cost')
    plt.title('Average Mean cost Training')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    path = os.path.join('plots/logs', 'training.png') 
    plt.savefig(path)


def plot_clusters(num, e, centers, points, fig, model):
    plt.figure(0)
    plt.clf()
    plt.gca().set_xlim([-0.05,1.05])
    plt.gca().set_ylim([-0.05,1.05])
    clusters = e[fig].max()+1
    colors = cm.rainbow(np.linspace(0,1,clusters))
    for i in range(clusters):
        c = colors[i][:-1]
        mask = e[fig] == i
        x = torch.masked_select(points[fig,:,0], mask)
        y = torch.masked_select(points[fig,:,1], mask)
        plt.plot(x.cpu().numpy(), y.cpu().numpy(), 'o', c=rgb2hex(c))
        if centers is not None:
            center = centers[i]
            plt.plot([center.data[0]], [center.data[1]], '*', c=rgb2hex(c))
    plt.title('clustering')
    plt.savefig('./plots/clustering_it_{}_{}.png'.format(num, model))
    
    

def create_input(points, sigma2):
    bs, N, _ = points.size() #points has size bs,N,2
    OP = torch.zeros(bs,N,N,4).type(dtype)
    E = torch.eye(N).type(dtype).unsqueeze(0).expand(bs,N,N)
    OP[:,:,:,0] = E
    W = points.unsqueeze(1).expand(bs,N,N,dim) - points.unsqueeze(2).expand(bs,N,N,dim)
    dists2 = (W * W).sum(3)
    dists = torch.sqrt(dists2)
    W = torch.exp(-dists2 / sigma2)
    OP[:,:,:,1] = W
    D = E * W.sum(2,True).expand(bs,N,N)
    OP[:,:,:,2] = D
    U = (torch.ones(N,N).type(dtype)/N).unsqueeze(0).expand(bs,N,N)
    OP[:,:,:,3] = U
    OP = Variable(OP)
    x = Variable(points)
    Y = Variable(W.clone())

    # Normalize inputs
    if normalize:
        mu = x.sum(1)/N
        mu_ext = mu.unsqueeze(1).expand_as(x)
        var = ((x - mu_ext)*(x - mu_ext)).sum(1)/N
        var_ext = var.unsqueeze(1).expand_as(x)
        x = x - mu_ext
        x = x/(10 * var_ext)

    return (OP, x, Y), dists

def sample_K(probs, K, mode='test'):
    probs = 1e-6 + probs*(1 - 2e-6) # to avoid log(0)
    probs = probs.view(-1, 2**K)
    if mode == 'train':
        bin_sample = torch.multinomial(probs, 1).detach()
    else:
        bin_sample = probs.max(1)[1].detach().unsqueeze(1)
    sample = bin_sample.clone().type(dtype)
    log_probs_samples = torch.log(probs).gather(1, bin_sample).squeeze()
    log_probs_samples = log_probs_samples.view(batch_size, N).sum(1)
    return bin_sample.data.view(batch_size, N), log_probs_samples

def sample_one(probs, mode='test'):
    probs = 1e-6 + probs*(1 - 2e-6) # to avoid log(0)
    if mode == 'train':
        rand = torch.zeros(*probs.size()).type(dtype)
        nn.init.uniform(rand)
    else:
        rand = torch.ones(*probs.size()).type(dtype) / 2
    bin_sample = probs > Variable(rand)
    sample = bin_sample.clone().type(dtype)
    log_probs_samples = (sample*torch.log(probs) + (1-sample)*torch.log(1-probs)).sum(1)
    return bin_sample.data, log_probs_samples

def update_input(input, dists, sample, sigma2, e, k):
    OP, x, Y = input
    bs = x.size(0)
    N = x.size(1)
    sample = sample.float()
    mask = sample.unsqueeze(1).expand(bs,N,N)*sample.unsqueeze(2).expand(bs,N,N)
    mask += (1-sample).unsqueeze(1).expand(bs,N,N)*(1-sample).unsqueeze(2).expand(bs,N,N)
    U = (OP.data[:,:,:,3]>0).float()*mask
    
    W = dists*U
    Wm = W.max(2,True)[0].expand_as(W).max(1,True)[0].expand_as(W)
    W = W / Wm.clamp(min=1e-6) * np.sqrt(2)
    W = torch.exp(- W*W / sigma2)
    
    OP[:,:,:,1] = Variable(W)
    D = OP.data[:,:,:,0] * OP.data[:,:,:,1].sum(2,True).expand(bs,N,N)
    OP[:,:,:,2] = Variable(D)
    
    U = U / U.sum(2,True).expand_as(U)
    OP[:,:,:,3] = Variable(U)
    Y = Variable(OP[:,:,:,1].data.clone())

    # Normalize inputs
    if normalize:
        z = Variable(torch.zeros((bs, N, 2**k))).type(dtype)
        e = e.unsqueeze(2)
        o = Variable(torch.ones((bs, N, 1))).type(dtype)
        z = z.scatter_(2, e, o)
        z = z.unsqueeze(2).expand(bs, N, 2, 2**k)
        z_bar = z * x.unsqueeze(3).expand_as(z)
        Nk = z.sum(1)
        mu = z_bar.sum(1)/Nk
        mu_ext = mu.unsqueeze(1).expand_as(z)*z
        var = ((z_bar - mu_ext)*(z_bar - mu_ext)).sum(1)/Nk
        var_ext = var.unsqueeze(1).expand_as(z)*z
        x = x - mu_ext.sum(3)
        x = x/(10 * var_ext.sum(3))
        # plt.figure(1)
        # plt.clf()
        # plt.plot(x[0,:,0].data.cpu().numpy(), x[0,:,1].data.cpu().numpy(), 'o')
        # plt.savefig('./plots/norm.png')
        # pdb.set_trace()
    return OP, x, Y

def compute_variance(e, probs):
    bs, N = probs.size()
    variance = Variable(torch.zeros(bs).type(dtype))
    for i in range(e.max()+1):
        mask = Variable((e == i).float())
        Ns = mask.sum(1).clamp(min=1)
        masked_probs = probs*mask
        probs_mean = (masked_probs).sum(1) / Ns
        v = (masked_probs*masked_probs).sum(1) / Ns - probs_mean*probs_mean
        variance += v
    return variance

def compute_reward(e, K, points):
    bs, N, _ = points.size()
    reward2 = Variable(torch.zeros(bs).type(dtype))
    reward3 = Variable(torch.zeros(bs).type(dtype))
    c = []
    for k in range(2**K):
        mask = Variable((e == k).float()).unsqueeze(2).expand_as(points)
        N1 = mask.sum(1)
        center = points*mask
        center = center.sum(1) / N1.clamp(min=1)
        c.append(center[0])
        subs = ((points-center.unsqueeze(1).expand_as(points)) * mask)
        subs2 = (subs * subs).sum(2).sum(1) / N
        subs3 = torch.abs(subs * subs * subs).sum(2).sum(1) / N
        reward2 += subs2
        reward3 += subs3
    return reward2, reward3, c

def execute(points, K, n_samples, sigma2, reg_factor, mode='test'):
    bs, N, _ = points.size()
    e = torch.zeros(bs, N).type(dtype_l)
    input, dists = create_input(points.data, sigma2)
    loss_total = Variable(torch.zeros(1).type(dtype))
    for k in range(K):
        scores,_ = gnn(input)
        probs = F.sigmoid(scores)
        if mode == 'train':
            variance = compute_variance(e, probs)
            variance = variance.sum() / bs
            Lgp = Variable(torch.zeros(n_samples, bs).type(dtype))
            Reward2 = Variable(torch.zeros(n_samples, bs).type(dtype))
            Reward3 = Variable(torch.zeros(n_samples, bs).type(dtype))
            for i in range(n_samples):
                Samplei, Lgp[i] = sample_one(probs, 'train')
                Ei = e*2 + Samplei.long()
                Reward2[i], _,_ = compute_reward(Ei, k+1, points)
            baseline = Reward2.mean(0,True).expand_as(Reward3)
            loss = 0.0
            if (last and k == K-1) or not last:
                loss = ((Reward2-baseline) * Lgp).sum(1).sum(0) / n_samples / bs
            loss_total = loss_total + loss - reg_factor*variance
            show_loss = Reward2.data.mean()
        sample, lgp = sample_one(probs, 'test')
        e = e*2 + sample.long()
        reward,_,c = compute_reward(e, k+1, points)
        if mode == 'test':
            show_loss = reward.data.mean()
        if k < K-1:
            input = update_input(input, dists, sample, sigma2, e, k+1)
    if mode == 'test':
        return e, None, show_loss, c
    else:
        return e, loss_total, show_loss, c

def execute_baseline(points, K, n_samples, sigma2, reg_factor, mode='test'):
    bs, N, _ = points.size()
    e = torch.zeros(bs, N).type(dtype_l)
    input, dists = create_input(points.data, sigma2)
    loss_total = Variable(torch.zeros(1).type(dtype))
    scores,_ = gnn(input)
    probs = F.softmax(scores.permute(2, 1, 0)).permute(2, 1, 0)
    if mode == 'train':
        Lgp = Variable(torch.zeros(n_samples, bs).type(dtype))
        Reward2 = Variable(torch.zeros(n_samples, bs).type(dtype))
        Reward3 = Variable(torch.zeros(n_samples, bs).type(dtype))
        for i in range(n_samples):
            Samplei, Lgp[i] = sample_K(probs, K, 'train')
            Reward2[i], _,_ = compute_reward(Samplei, K, points)
        baseline = Reward2.mean(0,True).expand_as(Reward3)
        loss = ((Reward2-baseline) * Lgp).sum(1).sum(0) / n_samples / bs
        loss_total = loss_total + loss
        show_loss = Reward2.data.mean()
    sample, lgp = sample_K(probs, K, 'test')
    reward, _, c = compute_reward(sample, K, points)
    if mode == 'test':
        show_loss = reward.data.mean()
    if mode == 'test':
        return sample, None, show_loss, c
    else:
        return sample, loss_total, show_loss, c


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


def Lloyds(input, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters)
    nb_pbs, nb_samples, d = input.shape
    Costs = []
    for i in range(nb_pbs):
        inp = input[i]
        labels = kmeans.fit_predict(inp)
        cost = 0
        for cl in range(n_clusters):
            ind = np.where(labels==cl)[0]
            if ind.shape[0] > 0:
                x = inp[ind]
                mean = x.mean(axis=0)
                cost += np.mean(np.sum((x - mean)**2, axis=1), axis=0)*ind.shape[0]
                # cost += np.var(inp[ind], axis=0)*ind.shape[0]
    Costs.append(cost/nb_samples)
    Cost = sum(Costs)/len(Costs)
    return Cost

def Lloyds2(input, ind, E, k, K=2):
    # split at first place
    inp = input[ind]
    if inp.shape[0] >= 2:
        kmeans = KMeans(n_clusters=2, max_iter=20)
        labels = kmeans.fit_predict(inp)
    else:
        labels = np.zeros(ind.shape[0])
    E[ind] = 2*E[ind] + labels
    # recursion
    if k == K-1:
        return E
    else:
        ind1 = ind[np.where(labels == 0)[0]]
        ind2 = ind[np.where(labels == 1)[0]]
        E = Lloyds2(input, ind1, E, k+1, K=K)
        E = Lloyds2(input, ind2, E, k+1, K=K)
        return E

def recursive_Lloyds(input, K=2):
    n_clusters = 2**K
    nb_pbs, nb_samples, d = input.shape
    Costs = []
    Labels = []
    for i in range(nb_pbs):
        inp = input[i]
        ind = np.arange(nb_samples)
        labels = np.zeros(nb_samples)
        labels = Lloyds2(inp, ind, labels, 0, K=K)
        Labels.append(labels)
        # pdb.set_trace()
        cost = 0
        for cl in range(n_clusters):
            ind = np.where(labels==cl)[0]
            if ind.shape[0] > 0:
                x = inp[ind]
                mean = x.mean(axis=0)
                cost += np.mean(np.sum((x - mean)**2, axis=1), axis=0)*ind.shape[0]
                # cost += np.var(inp[ind], axis=0)*ind.shape[0]
        Costs.append(cost/nb_samples)
    Cost = sum(Costs)/len(Costs)
    Labels = np.reshape(Labels, [nb_pbs, nb_samples])
    return Cost, Labels

if __name__ == '__main__':
    
    dim = args.dim
    num_examples_train = args.num_examples_train
    num_examples_test = args.num_examples_test
    N = args.N
    clusters = args.clusters
    clip_grad_norm = args.clip_grad_norm
    batch_size = args.batch_size
    num_features = args.num_features
    num_layers = args.num_layers
    sigma2 = args.sigma2
    reg_factor = args.reg_factor
    K = args.K
    k_step = args.k_step
    n_samples = args.n_samples
    normalize = args.normalize
    last = args.last
    baseline = args.baseline
    
    if args.dataset == 'GM':
        gen = Generator('/data/anowak/dataset/', num_examples_train, num_examples_test, N, clusters, dim)
    elif args.dataset == "CIFAR":
        gen = GeneratorCIFAR('/data/anowak/dataset/', num_examples_train, num_examples_test, N, clusters, dim)
        dim = 27
    gen.load_dataset()
    num_iterations = 100000
    
    if not baseline:
        gnn = Split_GNN(num_features, num_layers, 5, dim_input=dim)
    else:
        gnn = Split_BaselineGNN(num_features, num_layers, 5, K, dim_input=dim)
    if args.load_file != '':
        gnn = load_model(args.load_file, gnn)
    optimizer = optim.RMSprop(gnn.parameters(), lr=1e-3)
    # optimizer = optim.Adam(gnn.parameters())
    
    test = args.test
    if test:
        num_iterations = num_examples_test // batch_size
    
    
    log = Logger()
    start = time.time()
    for it in range(num_iterations):
        if it % 50 == 0:
            mode = 'test'
        batch = gen.sample_batch(batch_size, is_training=True)
        points, target = batch
        if k_step > 0:
            k = min(K,1+it//k_step)
        else:
            k = K
        if not baseline:
            e, loss, show_loss, c = execute(points, k, n_samples, sigma2, reg_factor, mode=mode)
        else:
            e, loss, show_loss, c = execute_baseline(points, k, n_samples, sigma2, reg_factor, mode=mode)
        log.add('show_loss', show_loss)
        
        if mode == 'train':
            gnn.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(gnn.parameters(), clip_grad_norm)
            optimizer.step()
            
        # if not test:
        if it%50 == 0:
            elapsed = time.time()-start
            # print('iteration {}, var {}, loss {}, elapsed {}'.format(it, show_loss, loss.data.mean(), elapsed))
            plot_clusters(it, e, c, points.data, 0, 'gnn')
            #out1 = ['---', it, loss, w, wt, tw, elapsed]
            #print(template_train1.format(*info_train))
            #print(template_train2.format(*out1))
            start = time.time()
            # Lloyds
            points_np = points.data.cpu().numpy()
            cost_lloyd = Lloyds(points_np, n_clusters=clusters)
            cost_rec_lloyd, labels = recursive_Lloyds(points_np, K=K)
            labels = torch.from_numpy(labels).type(dtype_l)
            plot_clusters(it, labels, None, points.data, 0, 'lloyds')
            print('gnn: {:.5f}, lloyds: {:.5f}, rec_lloyds: {:.5f}, ratio_lloyd: {:.5f}'
                  'ratio_rec_lloyd: {:.5f}'
                  .format(show_loss, cost_lloyd, cost_rec_lloyd,
                          show_loss/cost_lloyd, show_loss/cost_rec_lloyd))
            mode = 'train'

        if it%300 == 0:
            plot_train_logs(log.get('show_loss'))
            if args.save_file != '':
                save_model(args.save_file, gnn)
        
    if test:
        a = 1
        #ensenyar resultats
            
    



















