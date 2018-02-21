import numpy as np
import os
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Generator(object):
    def __init__(
                 self, num_examples_train, num_examples_test,
                 path_dataset, batch_size
                 ):
        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test
        self.batch_size = batch_size
        self.path_dataset = path_dataset
        self.input_size = 2
        # self.input_size = 3
        self.task = 'convex_hull'
        scales_train = [1, 2, 3]
        scales_test = [5]
        self.scales = {'train': scales_train, 'test': scales_test}
        self.data = {'train': {}, 'test': {}}

    def load_dataset(self):
        for mode in ['train', 'test']:
            for sc in self.scales[mode]:
                path = os.path.join(self.path_dataset, mode + str(sc))
                if self.input_size == 2:
                    path = path + 'def.npz'
                elif self.input_size == 3:
                    path = path = 'def3d.npz'
                if os.path.exists(path):
                    print('Reading {} dataset for {} scales'
                          .format(mode, sc))
                    npz = np.load(path)
                    self.data[mode][sc] = {'x': npz['x'], 'y': npz['y']}
                else:
                    x, y = self.create(scales=sc, mode=mode)
                    self.data[mode][sc] = {'x': x, 'y': y}
                    # save
                    np.savez(path, x=x, y=y)
                    print('Created {} dataset for {} scales'
                          .format(mode, sc))

    def get_batch(self, batch=0, scales=3, mode="train"):
        bs = self.batch_size
        batch_x = self.data[mode][scales]['x'][batch * bs: (batch + 1) * bs]
        batch_y = self.data[mode][scales]['y'][batch * bs: (batch + 1) * bs]
        return batch_x, batch_y

    def compute_length(self, scales, mode='train'):
        if mode == 'train':
            length = np.random.randint(3 * 2 ** scales, 6 * 2 ** (scales) + 1)
            max_length = 6 * 2 ** scales
        else:
            if scales == 2:
                length, max_length = 25, 25
            elif scales == 3:
                length, max_length = 50, 50
            elif scales == 4:
                length, max_length = 100, 100
            elif scales == 5:
                length, max_length = 200, 200
        return length, max_length

    def convexhull_example(self, length, scales):
        points = np.random.uniform(0, 1, [length, self.input_size])
        target = -1 * np.ones([length])
        ch = ConvexHull(points).vertices
        argmin = np.argsort(ch)[0]
        ch = list(ch[argmin:]) + list(ch[:argmin])
        target[:len(ch)] = np.array(ch)
        target += 1
        return points, target

    def create(self, scales=3,  mode='train'):
        if mode == 'train':
            num_examples = self.num_examples_train
        else:
            num_examples = self.num_examples_test
        _, max_length = self.compute_length(scales, mode=mode)
        x = -1 * np.ones([num_examples, max_length, self.input_size])
        y = np.zeros([num_examples, max_length])
        for ex in range(num_examples):
            length, max_length = self.compute_length(scales, mode=mode)
            if self.task == "convex_hull":
                x_ex, y_ex = self.convexhull_example(length, scales)
                if ex % 500000 == 499999:
                    print('Created example {}'.format(ex))
            else:
                raise ValueError("task {} not implemented"
                                 .format(self.task))
            x[ex, :length], y[ex, :length] = x_ex, y_ex
        return x, y
