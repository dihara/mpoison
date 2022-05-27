# -*- coding: utf-8 -*-

from itertools import permutations
from anti_gonzalez import Poison
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from greedy_kcenter import GreedyCenters
from kcenter import Gonzalez, GreedyCenters, CentersWOutliers

from sklearn import datasets
da = datasets.load_iris()

#%%

X = np.empty(shape=(0,2))
for i in range(3):
    center = np.random.rand(2)
    print(center)
    xi = center + np.random.randn(50, 2) * 0.02
    X = np.concatenate((X, xi))
    
#%%    
plt.scatter(X[:,0], X[:,1])


#%%



X = PCA(n_components=2).fit_transform(da.data)
Y = da.target

# Normalize
X[:,0] += abs(X[:,0].min())
X[:,1] += abs(X[:,1].min())
q = max(X[:,0].max(), X[:,1].max())
X[:,0] /= q
X[:,1] /= q

scale = True
# Scale so poisoning can be added "far"
if scale:
    X[:,0] /= 100
    X[:,1] /= 100
    
    X[:,0] += 0.5 - X[:,0].max() / 2
    X[:,1] += 0.5 - X[:,1].max() / 2

#plot
plt.scatter(X[:,0], X[:,1], c=Y)

if scale:
    plt.ylim([0.48,0.52])
    plt.xlim([0.48,0.52])


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
    gkc = GreedyCenters(X, k, 0, 0)
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

        gkc = GreedyCenters(X.tolist(), k, r, 0)

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
    if Robust(X.tolist() + p, r, k, m):
        best = r
        
    
    while (low + 1) < high:
        q = (low + high) // 2
        r = z[q] + eps
        ro = Robust(X.tolist() + p, r, k, m)
        if not ro.robust_clustering():
            low = q
        else:
            high = q
            best = r
        
    # print(best, z[-1] + eps)    
    return best        

#%%
T = 100
Q = 100
# steps = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
steps = np.arange(11) * 0.01
results_cost = np.zeros((6, len(steps)))
results_diff = np.zeros((6, len(steps)))

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
print(min_r)
#%%
gkc = GreedyCenters(X, k, min_r, 0)
print(gkc.kcenters())
ro = CentersWOutliers(X.tolist(), k, min_r, 0)
print(ro.kcenters())

#%%
# kcg = k-center Gonzalez
# kcr = k-center Robust
# kcgs = k-center Greedy Sampling

measure_cost = True

for s, f in enumerate(steps):
    m = math.floor(f * len(X))

    ba = Gonzalez(X.tolist(), k)
    ba.kcenters(randseed)
    base_clusters = ba.clusters

    # random poison results
    t_cost = 0
    t_cost_ro = 0
    t_kcgs_cost = 0
    t_diff = 0
    t_diff_ro = 0
    t_kcgs_diff = 0
    
    for t in range(T):
        print('.', end='')
        poison = []
        while len(poison) < m:
            poison.append([np.random.rand(), np.random.rand()])   

        # r = find_r(X, poison, k, m)
        r = min_r
        q_cost = 0
        q_cost_ro = 0
        q_kcgs_cost = 0
        q_diff = 0
        q_diff_ro = 0
        q_kcgs_diff = 0

        for q in range(Q):
            # apply gonzalez to data with random poison
            kc = Gonzalez(X.tolist() + poison, k)
            kc.kcenters(randseed)
            
            q_cost += kc.cost
            q_diff += compare_clustering(base_clusters, kc.clusters, poison)

            # apply robust clustering to data with random poison
            ro = CentersWOutliers(X.tolist() + poison, k, r, m)
            ro.kcenters()
           
            q_cost_ro += ro.cost
            q_diff_ro += compare_clustering(base_clusters, kc.clusters, poison)

            # apply greedy clustering to data with random poison
            kcgs = GreedyCenters(X.tolist()  + poison, k, min_r, m)
            kcgs.kcenters()
            q_kcgs_cost += kcgs.cost
            q_kcgs_diff += compare_clustering(base_clusters, kcgs.clusters, poison)


        t_cost += q_cost / Q
        t_cost_ro += q_cost_ro / Q
        t_kcgs_cost += q_kcgs_cost / Q
        t_diff += q_diff / Q
        t_diff_ro += q_diff_ro / Q
        t_kcgs_diff += q_kcgs_diff / Q
        
    results_cost[0, s] = t_cost / T
    results_cost[1, s] = t_cost_ro / T
    results_cost[2, s] = t_kcgs_cost / T
    results_diff[0, s] = t_diff / T
    results_diff[1, s] = t_diff_ro / T
    results_diff[2, s] = t_kcgs_diff / T

    # anti-gonzalez poisoning
    AG = Poison(m, X.tolist(), [[0,0], [0,1], [1,0], [1,1]])
    AG.anti_gonzalaz()
    ag_poison = AG.get_poison()

    q_cost = 0
    q_diff = 0    
    for q in range(Q):
        kc = Gonzalez(X.tolist() + ag_poison, k)
        kc.kcenters(randseed)
        
        q_cost += kc.cost
        q_diff += compare_clustering(base_clusters, kc.clusters, ag_poison)
            
    results_cost[3, s] = q_cost / Q
    results_diff[3, s] = q_diff / Q
            
    # cluster data with ag poison using robust clustering
    # r = find_r(X, ag_poison, k, m)
    ro = CentersWOutliers(X.tolist() + ag_poison, k, r, m)

   
    q_cost = 0
    q_diff = 0
    for q in range(Q):
        ro.kcenters(randseed)
        
        q_cost += ro.cost
        q_diff += compare_clustering(base_clusters, ro.clusters, ag_poison)

        # apply greedy clustering to data with random poison
        kcgs = GreedyCenters(X.tolist() + ag_poison, k, min_r, m)
        kcgs.kcenters()
        q_kcgs_cost += kcgs.cost
        q_kcgs_diff += compare_clustering(base_clusters, kcgs.clusters, poison)
    
    results_cost[4, s] = q_cost / Q    
    results_cost[5, s] = q_kcgs_cost / Q    
    results_diff[4, s] = q_diff / Q    
    results_diff[5, s] = q_kcgs_diff / Q        
    
    
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
        