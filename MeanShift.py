import numpy as np
from math import sqrt

class MeanShift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step
        
    def fit(self, X):
        if self.radius == None:
            all_data_centroid = np.average(X, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step
        
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]
            
        weights = [i for i in range(self.radius_norm_step)][::-1]
        
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in X:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.000000000000000001
                    
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    
                    to_add = (weight[weight_index]**2) * [featureset]
                    in_bandwidth += to_add
                    
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))
            
            uniques = sorted(list(set(new_centroids)))
            to_pop = []
            for i in uniques:
                if i in to_pop:
                    pass
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius and ii not in to_pop:
                        to_pop.append(ii)
                        
            for i in to_pop:
                uniques.remove(i)
            
            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
                
            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            
            if optimized:
                break
        
        self.centroids = centroids
        self.classifications = {}
        
        for i in range(len(self.centroids)):
            self.classifications[i] = []
            
        for featureset in X:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)
    
    def predict(self, X):
        distances = [sqrt(np.linalg.norm(featureset-self.centroids[centroid])) for centroid in self.centroids]
        classification = distances.index(min(distances)) 
        return classification