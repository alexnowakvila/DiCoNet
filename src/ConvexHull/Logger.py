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


class Logger(object):
    def __init__(self, path):
        directory = os.path.join(path, 'plots/')
        self.path = directory
        # Create directory if necessary
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)

    def write_settings(self, args):
        # write info
        path = self.path + 'settings.txt'
        with open(path, 'w') as file:
            for arg in vars(args):
                file.write(str(arg) + ' : ' + str(getattr(args, arg)) + '\n')

    def plot_losses(self, losses, losses_reg, scales=[], fig=0):
        # discriminative losses
        plt.figure(fig)
        plt.clf()
        plt.semilogy(range(0, len(losses)), losses, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.title('discriminative loss')
        path = os.path.join(self.path, 'losses.png')
        plt.savefig(path)
        # reg loss
        plt.figure(fig + 1)
        plt.clf()
        plt.semilogy(range(0, len(losses_reg)), losses_reg, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.title('split regularization loss')
        path = os.path.join(self.path, 'split_variances.png')
        plt.savefig(path)

    def plot_accuracies(self, accuracies, scales=[], mode='train', fig=0):
        plt.figure(fig)
        plt.clf()
        colors = cm.rainbow(np.linspace(0, 1, len(scales)))
        l = []
        names = [str(sc) for sc in scales]
        for i, acc in enumerate(accuracies):
            ll, = plt.plot(range(len(acc)), acc, color=colors[i])
            l.append(ll)
        plt.ylabel('accuracy')
        plt.legend(l, names, loc=2, prop={'size': 6})
        if mode == 'train':
            plt.xlabel('iterations')
        else:
            plt.xlabel('iterations x 1000')
        path = os.path.join(self.path, 'accuracies_{}.png'.format(mode))
        plt.savefig(path)

    def plot_Phis_sparsity(self, Phis, fig=0):
        Phis = [phis[0].data.cpu().numpy() for phis in Phis]
        plt.figure(fig)
        plt.clf()
        for i, phi in enumerate(Phis):
            plt.subplot(1, len(Phis), i + 1)
            # plot first element of the batch
            plt.spy(phi, precision=0.001, marker='o', markersize=2)
            plt.xticks([])
            plt.yticks([])
            plt.title('k={}'.format(i))
        path = os.path.join(self.path, 'Phis.png')
        plt.savefig(path)

    def plot_norm_points(self, Inputs_N, e, Perms, scales, fig=1):
        input = Inputs_N[0][0].data.cpu().numpy()
        e = torch.sort(e, 1)[0][0].data.cpu().numpy()
        Perms = [perm[0].data.cpu().numpy() for perm in Perms]
        plt.figure(fig)
        plt.clf()
        ee = e.copy()
        for i, perm in enumerate(Perms):
            plt.subplot(1, len(Perms), i + 1)
            colors = cm.rainbow(np.linspace(0, 1, 2 ** (scales - i)))
            perm = perm[np.where(perm > 0)[0]] - 1
            points = input[perm]
            e_scale = ee[perm]
            for node in range(2 ** (scales - i)):
                ind = np.where(e_scale == node)[0]
                pts = points[ind]
                plt.scatter(pts[:, 0], pts[:, 1], c=colors[node])
            ee //= 2
        path = os.path.join(self.path, 'visualize_example.png')
        plt.savefig(path)

    def plot_rates(self, discard_rates, fig=1):
        plt.figure(fig)
        plt.clf()
        plt.title('Discard Rates')
        for i, rate in enumerate(discard_rates):
            plt.subplot(len(discard_rates), 1, i + 1)
            colors = cm.rainbow(np.linspace(0, 1, len(rate)))
            l = [[] for ii in rate]
            names = ['scale {}'.format(ii) for ii in range(len(rate))]
            for j, sc in enumerate(rate):
                l[j], = plt.plot(sc, c=colors[j])
            plt.legend(l, names, loc=2, prop={'size': 6})
        plt.xlabel('Iterations')
        plt.ylabel('discard rates')
        path = os.path.join(self.path, 'rates.png')
        plt.savefig(path)

    def save_results(self, losses, accuracies_test, discard_rates):
        np.savez(self.path + 'results.npz', Loss=losses,
                 Accuracies=accuracies_test, Discard_rates=discard_rates)

    def save_test_results(self, accuracies_test, Discard_rates, scales):
        path = self.path + 'test_results.txt'
        with open(path, 'w') as file:
            file.write('--------------TEST RESULTS-------------- \n')
            for i, accs in enumerate(accuracies_test):
                result_acc = ('Accuracy for {} scales: {} \n'
                              .format(scales[i], accs))
                result_rate = ('Discard rates for {} scales:'
                               .format(scales[i]))
                rates_str = ', '.join([str(rate) for rate in Discard_rates[i]])
                file.write(result_acc + result_rate + rates_str + '\n')
