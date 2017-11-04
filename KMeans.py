import numpy as np
from math import sqrt

class KMeans:
    def __init__(self, k=2, tol=0.001, max_iteration=300):
        self.k = k
        self.tol = tol
        self.max_iteration = max_iteration
    
    def fit(self, X):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
        
        for i in range(self.max_iteration):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
                
            for featureset in X:
                distances = [sqrt(np.linalg.norm(data - self.centroids[centrold])) for centroid in self.centrolds]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            previous_centroid = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/ original_centroid*100.0) > self.tol:
                    optimized = False
            
            if optimized:
                break
    
    def predict(self, data):
        distances = [sqrt(np.linalg.norm(data - self.centroids[centrold])) for centroid in self.centrolds]
        classification = distances.index(min(distances))