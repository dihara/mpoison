# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 09:18:43 2021

@author: dihar
"""
import math
import numpy as np
import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd

def randitem(L):
    return L[np.random.randint(len(L))]

N = 1000

np.random.seed(0)
X = np.empty(shape=(N, 2))
X[:500, :] = np.random.randn(500, 2) - 10
X[500:, :] = np.random.randn(500, 2)


Y = np.empty(shape=(N))
Y[:500 ] = np.ones(500)
Y[500:] = np.zeros(500)

labels = np.unique(Y)
#plt.scatter(X[:,0], X[:,1], c=Y)

from sklearn import datasets
da = datasets.load_iris()
# da = datasets.load_wine()
# da = datasets.load_digits()
# da = datasets.load_breast_cancer()
# X = da.data[:, :2]  # we only take the first two features.

if False:
    da = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00488/Live_20210128.csv')
    da['status_type'] = pd.Categorical(da.status_type)
    X_ = da.to_numpy()
    X = X_[:,3:-5]
    
    Y = pd.DataFrame(da['status_type'].cat.codes).to_numpy().flatten()
else:
    # X = PCA(n_components=2).fit_transform(da.data)
    X=da.data
    
    Y = da.target

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)

plt.scatter(X_test[:,0], X_test[:,1], c=Y_test)

# plt.savefig('histogram.pgf')

#%%
knn = KNeighborsClassifier(n_neighbors=3)

T = 100
perm  = np.random.permutation(np.arange(len(Y_train)))
stops = (np.arange(11) + 0) * 5
poison_perc = np.arange(11) * 5
acc = np.zeros(shape=(len(stops), len(poison_perc), T))

j = 0
for i, s in enumerate(stops):
    # add noise
    while j < math.floor(len(Y_train) * s / 100):
        Y_train[perm[j]] = randitem(labels)#abs(Y_train[perm[j]] - 1)
        j += 1

    for t in range(T):
        Y_pois = np.array(Y_train)    
        poison_points = np.random.permutation(np.arange(len(Y_train)))
        pois_count = 0

        #poison        
        for k, pf in enumerate(poison_perc):
            while pois_count < math.floor(len(Y_train) * pf / 100):
                Y_pois[poison_points[pois_count]] = randitem(labels) #abs(Y_pois[poison_points[pois_count]] - 1)
                pois_count += 1
                
            Y_pred = knn.fit(X_train, Y_pois).predict(X_test)
            acc[i, k, t] = accuracy_score(Y_test, Y_pred)
            # print('.', end='', flush=True)
    
results = np.mean(acc, axis=2)    
print(results)
#%%

plt.plot(results)
plt.ylabel('accuracy')
plt.xlabel('percentage of poison added')
plt.xticks(np.arange(11), poison_perc)
plt.title('BCanc')
#%%
np.random.seed(100)
X = np.random.randn(1000, 2)
X[:500, 0] = X[:500, 0]  - 6

Y = np.empty(shape=(N))
Y[:500 ] = np.ones(500)
Y[500:] = np.zeros(500)

from sklearn.model_selection import train_test_split

# plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)

# plt.scatter(X_test[:,0], X_test[:,1], c=Y_test)
plt.scatter(X[:,0], X[:,1], c=Y)
#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
Y_pred = knn.fit(X_train, Y_train).predict(X_test)
ac = accuracy_score(Y_test, Y_pred)
print(ac)

#%%
X[:500, 0] = X[:500, 0]  + 1
plt.scatter(X[:,0], X[:,1], c=Y)
#%%
knn = KNeighborsClassifier(n_neighbors=3)
T = 100
perm  = np.random.permutation(np.arange(len(Y_train)))
stops = np.arange(6)
poison_perc = np.arange(11) * 5
acc_2 = np.zeros(shape=(len(stops), len(poison_perc), T))

j = 0
for i, s in enumerate(stops):
    if i>0:
        X[:500, 0] = X[:500, 0]  + 1
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    #poison        
    for t in range(T):
        Y_pois = np.array(Y_train)    
        poison_points = np.random.permutation(np.arange(len(Y_train)))
        pois_count = 0

        for k, pf in enumerate(poison_perc):
            while pois_count < math.floor(len(Y_train) * pf / 100):
                Y_pois[poison_points[pois_count]] = abs(Y_pois[poison_points[pois_count]] - 1)
                pois_count += 1
                
            Y_pred = knn.fit(X_train, Y_pois).predict(X_test)
            acc_2[i, k, t] = accuracy_score(Y_test, Y_pred)
            print('.', end='', flush=True)
    
print(np.mean(acc_2, axis=2))