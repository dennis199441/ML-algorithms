import numpy as np

class SVM:
    def fit(self, data):
        ## data = {class1: np.array([x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]),
        ##         class2: np.array([x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn])}
        self.data = data
        ## {||W||: [W, b]}
        opt_dict = {}
        transforms = [[1,1], [-1,1],[-1,-1],[1,-1]]
        
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
                
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        
        ## support vectors yi((Xi dot W) + b) = 1
        step_size = [self.max_feature_value * 0.1,
                    self.max_feature_value * 0.01,
                    self.max_feature_value * 0.001]
        ## step size is the point of expense
        
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10
        
        for step in step_sizes:
            W = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                  self.max_feature_value * b_range_multiple,
                                  step * b_range_multiple):
                    for transformation in transforms:
                        W_t = W * transformation
                        found_option = True
                        ## Weakest link in the SVM fundamentally
                        ## SMO attempts to fix this a bit
                        ## yi((Xi dot W) + b) >= 1
                        for i in self.data:
                            for Xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(W_t, Xi) + b) >= 1:
                                    found_option = False
                        
                        if found_option:
                            opt_dict[np.linalg.norm(W_t)] = [W_t, b]
                            
                if W[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    W = W - step
                    
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.W = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2
        
    def predict(self, data):
        ## sign((X dot W) + b)
        classification = np.sign(np.dot(np.array(data), self.W) + self.b)
        return classification
