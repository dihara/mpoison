import sys, math, numpy as np 
import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi, voronoi_plot_2d
from voronoi import voronoi

class KCPoison():
    """ class for poison """
    def __init__(self, m, points, corners):
        self.m = m
        self.points = points        # my goal is to poison these points
        self.vor = None             # we use voronoi to find furtheset point
        self.vertices = []
        self.corners = corners
        self.poison_points = []

    def show_points(self):
        for p in self.points:
            print(p)

    def get_poison(self):
        return self.poison_points
        
    def show_poison_points(self):
        for p in self.poison_points:
            print(p)

    def get_points(self):
        return self.points

    def create_voronoi(self):
        """ creates and returns a voronoi diagram """
        v_points = np.array(self.points)
        bnd = [[0,0], [1, 0], [0,1], [1,1]]

        _, self.vor = voronoi(v_points, bnd)
        # pnts,vorn = voronoi(pins,bnd)

       
    def show_voronoi(self):
        # voronoi_plot_2d(self.vor)
        plt.show()

    def set_vertices(self):
        """ exclude the vertices with negative x,y values """
        # vertices_n = self.vor.vertices.tolist()
        
        self.vertices  = []
        for l in self.vor:
            for v in l:
                if not v in self.vertices:
                    self.vertices.append(v)
                    
        if (len(self.corners) > 0):
            self.vertices = self.vertices + self.corners # when looking for poison position, consider the vertices and corners of space
                    
            
    def random_sample(self):
        t = 1000
        print('random sampling in ', len(self.points[0]), 'dimensions')
        self.vertices = np.random.uniform(size=(t, len(self.points[0]))).tolist()


    def find_candidate_points(self):
        if len(self.points[0]) > 2:
            self.random_sample()
        else:
            self.create_voronoi()
            self.set_vertices()
                    
            
    def kcenter_poisoning(self):
        for i in range(self.m):
            self.find_candidate_points()
            self.place_poison()
            

    def remove_corner(self):
        for i in range(len(self.corners)):
            if self.corners[i] in self.points:
                self.corners.pop(i)

    def place_poison(self):
        min_dis_v = []
        
        """ find the min distance from each poison-candidate to all points """
        for i in range(len(self.vertices)):
            min_d = sys.maxsize
            for j in range(len(self.points)):
                dist_p_v = math.dist(self.points[j], self.vertices[i])
                if  dist_p_v < min_d:
                    min_d = dist_p_v
            min_dis_v.append(min_d)

        """ get the max between all the min distances """
        max_dist = -1
        index = -1
        for i in range(len(min_dis_v)):
            if min_dis_v[i] > max_dist:
                max_dist = min_dis_v[i]
                index = i

        self.points.append(self.vertices[index])
        self.poison_points.append(self.vertices[index])
        #self.remove_corner()