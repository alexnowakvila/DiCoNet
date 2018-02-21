import numpy as np 
import pickle
import pdb
from sklearn.cluster import KMeans

file = '/home/anowak/DCN-for-KMEANS/test_batch'

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

dat = unpickle(file)
x = dat[b'data']/256
Y = dat[b'labels']
N = x.shape[0]
X = np.zeros((N, 3, 32, 32))

for i in range(N):
  X[i, 0] = np.reshape(x[i, :1024], [32, 32])
  X[i, 1] = np.reshape(x[i, 1024:2048], [32, 32])
  X[i, 2] = np.reshape(x[i, 2048:], [32, 32])

j = 10
k = 10

n_samples_train = 10000
n_samples_test = 1000
DAT_train = np.zeros((n_samples_train, 3*3*3))
DAT_test = np.zeros((n_samples_test, 3*3*3))
for l in range(n_samples_train):
  i = np.random.randint(N)
  j = np.random.randint(29)
  k = np.random.randint(29)
  r = X[l, 0, j:j+3, k:k+3].flatten()
  g = X[l, 1, j:j+3, k:k+3].flatten()
  b = X[l, 2, j:j+3, k:k+3].flatten()
  rgb = np.concatenate((r, g, b), axis=0)
  DAT_train[l] = rgb
for l in range(n_samples_test):
  i = np.random.randint(N)
  j = np.random.randint(29)
  k = np.random.randint(29)
  r = X[l, 0, j:j+3, k:k+3].flatten()
  g = X[l, 1, j:j+3, k:k+3].flatten()
  b = X[l, 2, j:j+3, k:k+3].flatten()
  rgb = np.concatenate((r, g, b), axis=0)
  DAT_test[l] = rgb

out_train = '/data/anowak/dataset/KMEANS_cifar_train'
out_test = '/data/anowak/dataset/KMEANS_cifar_test'
np.save(out_train, DAT_train)
np.save(out_test, DAT_test)

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

# cost1 =  Lloyds(np.reshape(DAT_train, [1, 20000, 27]), n_clusters=32)
# print(cost1)
# cost2, _ =  recursive_Lloyds(np.reshape(DAT_train, [1, 20000, 27]), K=5)
# print(cost2)