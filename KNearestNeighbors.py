import numpy as np
import warnings
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, data):
        self.data = data
        
    def predict(self, predict):
        
        if len(data) < self.k:
            warnings.warn('Data is less than total voting group!')

        distances = []
        ## for given data (features and labels/group)
        for group in self.data:
            for features in self.data[group]:
                ## compute the euclidean distance
                euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
                distances.append([euclidean_distance, group])

        ## for the nearest k data points
        votes = [i[1] for i in sorted(distances)[:self.k]]
        ## choose the most common group among votes
        result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1] / self.k

        return result, confidence
