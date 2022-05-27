# -*- coding: utf-8 -*-

from itertools import permutations, product
from kcenter_poisoning import KCPoison
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from greedy_kcenter import GreedyCenters
from kcenter import Gonzalez, GreedyCenters, CentersWOutliers
from random import sample
from sklearn import datasets
#%%
from keras.datasets import mnist

#%%

import requests, gzip, os, hashlib



#%%

synthetic = False
if synthetic:
    np.random.seed(10)
    X = np.empty(shape=(0,2))
    k = 3
    t = 50
    for i in range(k):
        center = 0.1 + np.random.random_sample((2)) * 0.8
        
        x = np.random.randn(t, 2)
        n = np.linalg.norm(x, axis=1)
        r = np.sqrt(np.random.random_sample((t)))*0.05
        
        p = center + np.array([(x[i]/n[i])*r[i] for i in range(t)])  
       
        X = np.concatenate((X, p))

    #plot
    plt.scatter(X[:, 0], X[:, 1])
    plt.ylim([0,1])
    plt.xlim([0,1])

else:
    if False:
        da = datasets.load_iris()
        d = len(da.data[0])
        
        X = PCA(n_components=d).fit_transform(da.data)
        Y = da.target
    
        # Normalize to fit in unit-square
        for i in range(len(X[0])):
            X[:,i] += abs(np.min(X[:,i]))
            
        q = np.max(X)
        X /= q
    
        scale = False
        # Scale so poisoning can be added "far"
        if scale:
            X[:,0] /= 100
            X[:,1] /= 100
            
            X[:,0] += 0.5 - X[:,0].max() / 2
            X[:,1] += 0.5 - X[:,1].max() / 2

    else:
        da = datasets.load_diabetes()
        
        d = len(da.data[0])
        
        X = PCA(n_components=d).fit_transform(da.data)
        Y = da.target
        
        # Normalize to fit in unit-square
        for i in range(len(X[0])):
            X[:,i] += abs(np.min(X[:,i]))
            
        q = np.max(X)
        X /= q
        
    plt.scatter(X[:,0], X[:,1], c=Y)

# if scale:
#     plt.ylim([0.48,0.52])
#     plt.xlim([0.48,0.52])

def compare_clustering(A, B, poison):
    dif=[]
       
    k = len(A)
    # list of indices
    K = [z for z in range(k)]
    # for each permutation of k clusters
    for P in permutations(K, k):
        c = 0
        for j in range(k):
            for g in A[P[j]].points:
                if g in poison:
                    continue
                
                if not g in B[j].points:
                    c+=1
               
        dif.append(c)
    
    return min(dif)

def find_bounded_r(X, k, upper_r):
    T = 10
    gkc = GreedyCenters(X, k, 0, [])
    z = np.sort(np.unique(gkc.distances))
    z = z[z<=upper_r]
    z = z[z>=(upper_r/2)]
    
    high = len(z) - 1
    low = 0
    
    r = z[0]
    gkc.r = r
    if gkc.kcenters() != math.inf:
        print("Success")
        best = r
    
    while (low + 1) < high:
        q = (low + high) // 2
        r = z[q]

        gkc = GreedyCenters(X.tolist(), k, r, [])

        if gkc.kcenters() != math.inf:
            low = q
        else:
            high = q
            best = r
        
    return best


def find_r(X, p, k, m):    
    distances = [[math.dist(X[i], X[j]) for i in range(len(X))] for j in range(len(X))]
    z = np.sort(np.unique(distances))

    eps = 0.000000000
    high = len(z) - 1
    low = 0
    
    r = z[-1] + eps
    if CentersWOutliers(X.tolist() + p, r, k, m):
        best = r
        
    
    while (low + 1) < high:
        q = (low + high) // 2
        r = z[q] + eps
        ro = CentersWOutliers(X.tolist() + p, r, k, m)
        if not ro.robust_clustering():
            low = q
        else:
            high = q
            best = r
        
    # print(best, z[-1] + eps)    
    return best        

T = 10
Q = 10
# steps = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
steps = np.arange(11) * 0.01
results_cost = np.zeros((6, len(steps)))
results_diff = np.zeros((6, len(steps)))
results_cost_std = np.zeros((6, len(steps)))
results_diff_std = np.zeros((6, len(steps)))

w_cost = -1
w_poison = []
b_cost = math.inf
b_poison = []
k=3
# r = 0.140
# r = 0.498
# r = 0.0023717
randseed=0

kc = Gonzalez(X, k)
kc.kcenters()
r = kc.cost
print(r)
min_r = find_bounded_r(X, k, r)
# min_r = find_r(X, k, r)
print(min_r)
# r = 0.05
# min_r = 0.05
#%%
gkc = GreedyCenters(X, k, min_r, [])
print(gkc.kcenters())
ro = CentersWOutliers(X.tolist(), k, min_r, 0)
print(ro.kcenters())

#%%
# kcg = k-center Gonzalez
# kcr = k-center Robust
# kcgs = k-center Greedy Sampling

# measure_cost = True

for s, f in enumerate(steps):
    m = math.floor(f * len(X))

    ba = Gonzalez(X.tolist(), k)
    ba.kcenters(randseed)
    base_clusters = ba.clusters

    # random poison results
    t_cost = []
    t_cost_ro = []
    t_kcgs_cost = []
    t_diff = []
    t_diff_ro = []
    t_kcgs_diff = []
    
    for t in range(T):
        print('.', end='')
        
        if m>0:
            poison = np.random.rand(m, len(X[0])).tolist()
        else:
            poison = []
            
        # r = find_r(X, poison, k, m)
        # r = min_r
        q_cost = []
        q_cost_ro = []
        q_kcgs_cost = []
        q_diff = []
        q_diff_ro = []
        q_kcgs_diff = []

        for q in range(Q):
            # apply gonzalez to data with random poison
            kc = Gonzalez(X.tolist() + poison, k)
            kc.kcenters()
            
            q_cost.append(kc.cost)
            q_diff.append(compare_clustering(base_clusters, kc.clusters, poison))
            
            # apply robust clustering to data with random poison
            ro = CentersWOutliers(X.tolist() + poison, k, r, m)
            ro.kcenters()
           
            q_cost_ro.append(ro.cost)
            q_diff_ro.append(compare_clustering(base_clusters, ro.clusters, poison))
            
            # apply greedy clustering to data with random poison
            kcgs = GreedyCenters(X.tolist()  + poison, k, min_r, poison)
            kcgs.kcenters()
            q_kcgs_cost.append(kcgs.cost)
            q_kcgs_diff.append(compare_clustering(base_clusters, kcgs.clusters, poison))


        t_cost.append(np.average(q_cost))
        t_cost_ro.append(np.average(q_cost_ro))
        t_kcgs_cost.append(np.average(q_kcgs_cost))
        t_diff.append(np.average(q_diff))
        t_diff_ro.append(np.average(q_diff_ro))
        t_kcgs_diff.append(np.average(q_kcgs_diff))
        
    results_cost[0, s] = np.average(t_cost)
    results_cost[1, s] = np.average(t_cost_ro)
    results_cost[2, s] = np.average(t_kcgs_cost)
    results_cost_std[0, s] = np.std(t_cost)
    results_cost_std[1, s] = np.std(t_cost_ro)
    results_cost_std[2, s] = np.std(t_kcgs_cost)
    
    results_diff[0, s] = np.average(t_diff)
    results_diff[1, s] = np.average(t_diff_ro)
    results_diff[2, s] = np.average(t_kcgs_diff)
    results_diff_std[0, s] = np.std(t_diff)
    results_diff_std[1, s] = np.std(t_diff_ro)
    results_diff_std[2, s] = np.std(t_kcgs_diff)

    # anti-gonzalez poisoning
    # bb=corner coordinates of hypercube
    bb=list(product([0,1], repeat=len(X[0])))
    AG = KCPoison(m, X.tolist(), bb)
    AG.kcenter_poisoning()
    ag_poison = AG.get_poison()

    q_cost = []
    q_diff = [] 
    for q in range(Q):
        kc2 = Gonzalez(X.tolist() + ag_poison, k)
        kc2.kcenters()
        
        q_cost.append(kc2.cost)
        q_diff.append(compare_clustering(base_clusters, kc2.clusters, ag_poison))
            
    results_cost[3, s] = np.average(q_cost)
    results_diff[3, s] = np.average(q_diff)
    results_cost_std[3, s] = np.std(q_cost)
    results_diff_std[3, s] = np.std(q_diff)
            
    # cluster data with ag poison using robust clustering
    # r = find_r(X, ag_poison, k, m)
    ro2 = CentersWOutliers(X.tolist() + ag_poison, k, r, m)
   
    q_cost = []
    q_diff = []
    q_kcgs_cost = []
    q_kcgs_diff = []
    for q in range(Q):
        ro2.kcenters()
        
        q_cost.append(ro2.cost)
        q_diff.append(compare_clustering(base_clusters, ro2.clusters, ag_poison))

        # apply greedy clustering to data with random poison
        kcgs2 = GreedyCenters(X.tolist() + ag_poison, k, min_r, ag_poison)
        kcgs2.kcenters()
        q_kcgs_cost.append(kcgs2.cost)
        q_kcgs_diff.append(compare_clustering(base_clusters, kcgs2.clusters, ag_poison))
    
    results_cost[4, s] = np.average(q_cost)
    results_cost[5, s] = np.average(q_kcgs_cost)
    results_diff[4, s] = np.average(q_diff)
    results_diff[5, s] = np.average(q_kcgs_diff)

    results_cost_std[4, s] = np.std(q_cost)
    results_cost_std[5, s] = np.std(q_kcgs_cost)
    results_diff_std[4, s] = np.std(q_diff)
    results_diff_std[5, s] = np.std(q_kcgs_diff)    
        # if kc.get_cost() > w_cost:
        #     w_cost = kc.get_cost()
        #     w_poison = poison

        # if kc.get_cost() < b_cost:
        #     b_cost = kc.get_cost()
        #     b_poison = poison

# print('worst')
# kc = KCenter(X.tolist() + w_poison, 3, w_poison)
# kc.d_gonzalez()
# kc.show()

# print('best')
# kc = KCenter(X.tolist() + b_poison, 3, b_poison)
# kc.d_gonzalez()
# kc.show()
    
print('mean cost no poison', results_cost)
print('mean diff no poison', results_diff)
print('std cost no poison', results_cost_std)
print('std diff no poison', results_diff_std)
#%%
import pandas as pd

# anti-gonzalez vs random as dimensions decrease

#IRIS
da = datasets.load_iris()
X0 = da.data
Y = da.target
k = 3

bankdata=False
#MNIST
if False:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    X0 = train_X[np.random.randint(len(train_X), size=500)]
    X0 = X0.reshape(len(X0),-1) #Flatten images
    X0 = PCA(n_components=10).fit_transform(X0)

else:
    da = datasets.load_diabetes()
    
    d = len(da.data[0])
    
    X0 = da.data
    Y = da.target
    
if bankdata:
    X0 = get_bankdata()
    X0 = X0[np.random.randint(len(train_X[0]), size=500)]
    X0 = PCA(n_components=10).fit_transform(X0)
    d = len(X0[0])
    
bc = datasets.load_breast_cancer()
X0 = PCA(n_components=10).fit_transform(bc.data)
Y = bc.target
d = len(X0[0])


# Vehicle
#%%
import pandas as pd

df = pd.read_csv('vehicles.csv',header=None,sep=" ")
x=df.loc[:,df.columns!=18]
X0 = PCA(n_components=10).fit_transform(x)
d = len(X0[0])

#%%
d = len(X0[0])
k = 3

# fraction of poison to add
f = 0.1

T=10
Q=10

results_cost = np.zeros((2, d-1))
results_diff = np.zeros((2, d-1))
results_cost_std = np.zeros((2, d-1))
results_diff_std = np.zeros((2, d-1))

# X0 = da.data[np.random.randint(len(da.data), 1000)]
# X0 = da.data
m = math.floor(f * len(X0))

for i in range(0, d - 1, 1):
    X = PCA(n_components=d-i).fit_transform(X0)

    dim = i #len(X[0])
    # Normalize to fit in unit-square
    for i in range(len(X[0])):
        X[:,i] += abs(np.min(X[:,i]))
        
    q = np.max(X)
    X /= q

    ba = Gonzalez(X.tolist(), k)
    ba.kcenters(randseed)
    base_clusters = ba.clusters

    # random poison results
    t_cost = []
    # t_cost_ro = 0
    # t_kcgs_cost = 0
    t_diff = []
    # t_diff_ro = 0
    # t_kcgs_diff = 0
    
    for t in range(T):
        print('.', end='')
        
        if m>0:
            poison = np.random.rand(m, len(X[0])).tolist()
        else:
            poison = []
            
        # r = find_r(X, poison, k, m)
        # r = min_r
        q_cost = []
        # q_cost_ro = 0
        # q_kcgs_cost = 0
        q_diff = []
        # q_diff_ro = 0
        # q_kcgs_diff = 0

        for q in range(Q):
            print('+', end='')

            # apply gonzalez to data with random poison
            kc = Gonzalez(X.tolist() + poison, k)
            kc.kcenters()
            
            q_cost.append(kc.cost)
            if k < 3:
                q_diff.append(compare_clustering(base_clusters, kc.clusters, poison))

        t_cost.append(np.average(q_cost))
        # t_cost_ro += q_cost_ro / Q
        # t_kcgs_cost += q_kcgs_cost / Q
        t_diff.append(np.average(q_diff))
        # t_diff_ro += q_diff_ro / Q
        # t_kcgs_diff += q_kcgs_diff / Q
        
    results_cost[0, dim] = np.average(t_cost)
    results_diff[0, dim] = np.average(t_diff)
    results_cost_std[0, dim] = np.std(t_cost)
    results_diff_std[0, dim] = np.std(t_diff)

    # anti-gonzalez poisoning
    # bb=corner coordinates of hypercube
    if len(X[0])<10:
        bb=list(product([0,1], repeat=len(X[0])))
    else:
        bb = []
    print('a')
    AG = KCPoison(m, X.tolist(), bb)
    print('b')
    AG.kcenter_poisoning()
    print('c')
    ag_poison = AG.get_poison()

    q_cost = []
    q_diff = []
    for q in range(Q):
        print('$', end='')
        kc2 = Gonzalez(X.tolist() + ag_poison, k)
        kc2.kcenters()
        
        q_cost.append(kc2.cost)
        if k<3:
            q_diff.append(compare_clustering(base_clusters, kc2.clusters, ag_poison))
            
    results_cost[1, dim] = np.average(q_cost)
    results_diff[1, dim] = np.average(q_diff)
    
    results_cost_std[1, dim] = np.std(q_cost)
    results_diff_std[1, dim] = np.std(q_diff)
            
print(results_cost)
print(results_diff)
print(results_cost_std)
print(results_diff_std)
#%%
# DIABETES DATA SET
from sklearn import datasets
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target



#%%
datasets.load_

#%%
c1 = 0
c2 = 0
for i in range(100):
    kc = KCenter(X.tolist() + poison, 3, poison)
    kc.d_gonzalez()
    c1 += kc.get_cost()
    # kc.show()
    # print(kc.get_cost())
    kc = KCenter(X.tolist() + ag_poison, 3, ag_poison)
    kc.d_gonzalez()
    c2 += kc.get_cost()
    # kc.show()
    # print(kc.get_cost())
    
    
print('poison', c1/100, 'antig', c2/100)


#%%

r = 0.0124
k = 3
m = 5

AG = Poison(m, X.tolist(), [[0,0], [0,1], [1,0], [1,1]])
AG.anti_gonzalaz()
ag_poison = AG.get_poison()
print(ag_poison)
ro = Robust(X.tolist() + ag_poison, r, k, m)
print(ro.robust_clustering())

kc = KCenter(ro.covered, k, ag_poison)
kc.d_gonzalez()
print(kc.get_cost())

#%%
mi = math.inf
for i in range(10000):
    kc = KCenter(X.tolist(), k, [])
    kc.d_gonzalez()
    if kc.get_cost() < mi:
        mi = kc.get_cost()
        print('.', end='')

print(mi)

    
#%%
A = 3
B = 4
for i in range(A):
    j = 0 - i
    while j < B:
        print(i, j)
        
        j+=1

#%%                 
X0 = X[:10].tolist()
X1 = X[:10].tolist()
k=3
for f in range(1):
    eps = 0
    for j in range(eps):
        i = np.random.randint(len(X1))
        X1[i] = list(np.random.rand(1,2)[0]) #[q * 2 for q in X1[i]]
    
    
    z = 0
    T = 1
    for i in range(T):
        kc = KCenter(X0, k, [])
        kc.d_gonzalez()
        
        # print(kc.get_cost())
        
        kc2 = KCenter(X1, k, [])
        kc2.d_gonzalez()
        # print(kc2.get_cost())
    
        # for c in kc.get_clusters():
        #     print(len(c.c_points))
            
        # for c in kc2.get_clusters():
        #     print(len(c.c_points))
        # kc.show()
        # kc2.show()
        l = compare_clustering(kc.get_clusters(), kc2.get_clusters(), [])
        # print(l)
        z += l
        
    print(z/T)
        
#%%
print(compare_clustering(kc.get_clusters(), kc2.get_clusters()))
        

#%%
def get_bankdata(): 
    # Import libraries
    ## Basic libs
    import pandas as pd
    import numpy as np
    import warnings
    ## Data Visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Configure libraries
    warnings.filterwarnings('ignore')
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.style.use('seaborn')
    
    
    # Load dataset
    df_bank = pd.read_csv('https://raw.githubusercontent.com/rafiag/DTI2020/main/data/bank.csv')
    
    # Drop 'duration' column
    df_bank = df_bank.drop('duration', axis=1)
    
    # print(df_bank.info())
    print('Shape of dataframe:', df_bank.shape)
    df_bank.head()
    
    
    from sklearn.preprocessing import StandardScaler
    
    # Copying original dataframe
    df_bank_ready = df_bank.copy()
    
    scaler = StandardScaler()
    num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
    df_bank_ready[num_cols] = scaler.fit_transform(df_bank_ready[num_cols])
    
    df_bank_ready.head()
    
    from sklearn.preprocessing import OneHotEncoder
    
    encoder = OneHotEncoder(sparse=False)
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    # Encode Categorical Data
    df_encoded = pd.DataFrame(encoder.fit_transform(df_bank_ready[cat_cols]))
    df_encoded.columns = encoder.get_feature_names(cat_cols)
    
    # Replace Categotical Data with Encoded Data
    df_bank_ready = df_bank_ready.drop(cat_cols ,axis=1)
    df_bank_ready = pd.concat([df_encoded, df_bank_ready], axis=1)
    
    # Encode target value
    df_bank_ready['deposit'] = df_bank_ready['deposit'].apply(lambda x: 1 if x == 'yes' else 0)
    
    print('Shape of dataframe:', df_bank_ready.shape)
    df_bank_ready.head()
    
    
    # Select Features
    feature = df_bank_ready.drop('deposit', axis=1)
    
    # Select Target
    target = df_bank_ready['deposit']
    
    # Set Training and Testing Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(feature , target, 
                                                        shuffle = True, 
                                                        test_size=0.2, 
                                                        random_state=1)
    
    return X_train.values