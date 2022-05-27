# -*- coding: utf-8 -*-
import numpy as np
import math
import sys

class Cluster:
    def __init__(self, center):
        self.center = center
        self.points = []
        self.radius = 0
           
    def add_point(self, p, d):
        self.points.append(p)
        if d > self.radius:
            self.radius = d       

    def get_radius(self):
        return self.radius

    def get_center(self):
        return self.center


class KCenter():
    def __init__(self, X, k):
        self.points = X
        self.poison = []
        self.n = len(X)
        self.k = k

        self.centers = []
        self.clusters = []
        self.distances = np.zeros(shape=(self.n, self.n))
        self.cost = math.inf
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                try:
                    self.distances[i, j] = math.dist(self.points[i], self.points[j])
                except:
                    print('error en', i, j)
                    print(self.points[i])
                    print(self.points[j])
                    raise


    def assign_clusters(self):
        """ assign points to nearest cluster center """
        
        self.cost = 0
        
        for i in self.centers:
            self.clusters.append(Cluster(self.points[i]))
        
        for i in range(self.n):
            # if i in self.centers:
            #     continue
            
            if i in self.poison:
                continue
            
            min_cluster_dist = sys.maxsize
            
            # find closest center
            for j in range(len(self.centers)):
                curr_dist = self.distances[i, self.centers[j]]
                if curr_dist < min_cluster_dist:
                    min_cluster_dist = curr_dist
                    index = j
                    
            # add point
            self.clusters[index].add_point(self.points[i], self.distances[self.centers[index], i])

            if self.cost < min_cluster_dist:
                self.cost = min_cluster_dist


#Gonzalez algorithm for k-center clustering
class Gonzalez(KCenter):
    def choose_furthest_point(self):
        max_dist = 0
        
        for i in range(len(self.points)):
            if i in self.centers:
                continue
            
            min_dist = math.inf
            for j in self.centers:
                if self.distances[i, j] < min_dist:
                    min_dist = self.distances[i, j]
                    
            if min_dist > max_dist:
                max_dist = min_dist
                furthest_point = i
        
        self.centers.append(furthest_point)


    def kcenters(self, seed=-1):
        if (self.k < 1):
            return
        
        if seed>=0:
            np.random.seed(seed)
            
        p = np.random.randint(len(self.points))
        self.centers.append(p)
       
        while len(self.centers) < self.k:
            self.choose_furthest_point()
            
        self.assign_clusters()
        
        return self.cost


#Greedy Sampling for Approximate Clustering in the Presence of Outliers
class GreedyCenters(KCenter):
    def __init__(self, X, k, r, P):
        KCenter.__init__(self, X, k)
        for i in range(self.n):
            if self.points[i] in P:
                self.poison.append(i)
        
        self.z = len(P)
        self.r = r
        

    def reset(self):
        self.centers = []
        self.clusters = []
        self.cost = math.inf
        

    def center_candidates(self):
        """ return all points at distance at least 2*r from current centers or furthest point """
        # max_dist = 0
        # furthest_point = -1

        if len(self.centers) == 0:
            return [np.random.randint(self.n)]

        candidate_points = []

        for i in range(self.n):
            if i in self.centers:
                continue
            
            d_closest_center = math.inf
            for j in self.centers:
                d = self.distances[i, j]
                if d < d_closest_center:
                    d_closest_center = d
            
            if d_closest_center > (2 * self.r):
                candidate_points.append(i)
 
            # if min_dist > max_dist:
            #     max_dist = min_dist
            #     furthest_point = i

        while len(candidate_points) == 0:
            c = np.random.randint(self.n)
            if not c in self.centers:
                candidate_points.append(c)       
        
        return candidate_points
               
    
    # def kcenters(self):
    #     T = 100
    #     # best_cost = math.inf
        
    #     while True:
    #         for t in range(self.k):
    #             c = self.center_candidates()
    #             if len(c) > 0:        
    #                 self.centers.append(c[np.random.randint(len(c))])
                
    #         if len(self.centers) == self.k:
    #             self.assign_clusters()
                
    #         if self.cost < best_cost:
    #             best_cost = self.cost
    #             best_centers = list(self.centers)
    #             self.reset()
                
    #         if 0 < T:
    #             T -= 1
    #         else:
    #             break
        
    #     if best_cost < math.inf:
    #         self.centers = best_centers
    #         self.assign_clusters()
        
    #     return self.cost
    
    def kcenters(self):
        T = 1000
        w = 5
        lowest_cost = math.inf
        best_centers = []
        while True:
            while w > 0:
                for t in range(self.k):
                    c = self.center_candidates()
                    if len(c) > 0:        
                        self.centers.append(c[np.random.randint(len(c))])
                    
                if len(self.centers) == self.k:
                    self.assign_clusters()
                    if self.cost < lowest_cost:
                        best_centers = self.centers
                        
                    w-=1
                else:
                    self.reset()

                if 0 < T:
                    T -= 1
                else:
                    return math.inf
            
            self.reset()
            self.centers = best_centers
            self.assign_clusters()
            return self.cost          


class CentersWOutliers(KCenter):
    def __init__(self, X, k, r, m):
        KCenter.__init__(self, X, k)
        self.r = r
        self.m = m
        self.regular_disks = []
        self.expanded_disks = []
        self.covered = []


    def construct_disks(self):
        """ function to construct disk of radius r """
        for i in range(self.n):
            temp_regular = []
            temp_expanded = []
            for j in range(self.n):
                if self.distances[i,j] <= self.r:
                    temp_regular.append(j)

                if self.distances[i,j] <= (self.r * 3):
                    temp_expanded.append(j)
                    
            # if len(temp) > 0:
            self.regular_disks.append(temp_regular)
            self.expanded_disks.append(temp_expanded)


    def cover_points(self):
        """ remove the disk with the most points. mark points as covered """
        for i in range(self.k):
            heaviest = -1
            for i in range(len(self.regular_disks)):
                if len(self.regular_disks[i]) > heaviest:
                    heaviest = len(self.regular_disks[i])
                    index = i
                    
            for p in self.expanded_disks[index]:
                for i in range(len(self.expanded_disks)):
                    if (i != index):
                        if (p in self.expanded_disks[i]):
                            self.expanded_disks[i].remove(p)
                        
                        if (p in self.regular_disks[i]):
                            self.regular_disks[i].remove(p)
                    
                if not p in self.covered:
                    self.covered.append(p)
            
            self.expanded_disks[index] = []
            self.regular_disks[index] = []


        for disk in self.expanded_disks:
            for point in disk:
                self.poison.append(point)                


    def find_outliers(self):
        """ 
        Algorithm for robust clustering 

        • Construct all disks and corresponding expanded disks.
        • Repeat the following k times:
            – Let Gj be the heaviest disk, i.e. contains the most uncovered points.
            – Mark as covered all points in the correspond- ing expanded disk Ej .
            – Update all the disks and expanded disks, i.e., remove from them all covered points.
        • If at least p points of V are marked as covered, then answer YES, else answer NO.
        """
        self.regular_disks = []
        self.expanded_disks = []
        self.covered = []

        self.construct_disks()
        self.cover_points()

        if len(self.covered) >= (len(self.points) - self.m):
            return True
        else:
            return False


    def furthest_covered_point(self):
        max_dist = 0
        
        for i in self.covered:
            if i in self.centers:
                continue
            
            min_dist = math.inf
            for j in self.centers:
                if self.distances[i, j] < min_dist:
                    min_dist = self.distances[i, j]
                    
            if min_dist > max_dist:
                max_dist = min_dist
                furthest_point = i
        
        self.centers.append(furthest_point)


    def kcenters(self, seed=-1):
        if (self.k < 1):
            return
        
        if self.find_outliers():       
            if seed>=0:
                np.random.seed(seed)
                
            p = self.covered[np.random.randint(len(self.covered))]
            self.centers.append(p)
           
            while len(self.centers) < self.k:
                self.furthest_covered_point()
                
            self.assign_clusters()
            
            return True
        else:
            print('outliers error')
            return False