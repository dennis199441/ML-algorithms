from DecisionTree import DecisionTree, Question, Leaf, Decision_Node

class RandomForest:
    def __init__(self, n_subsets=1, n_replacement=5):
        self.innermodel = DecisionTree()
        self.n_subsets = n_subsets
        self.n_replacement = n_replacement

    def generate_subset(self, group_number, data):
        subset = []
        i = 0
        while i < len(data):
            if i % self.n_subsets == group_number:
                subset.append(data[i])
            i += 1

        return subset

    def fit(self, data):
        ## save the original data
        self.dataset = data
        forest = [] ## use to save the tree of each subset

        i = 0
        while i < self.n_replacement: 
            random.shuffle(data)		
            ## split data into subsets and save to a list
            subsets = []
            j = 0
            while j < self.n_subsets:
                subset = self.generate_subset(j, data)
                subsets.append(subset)
                j += 1

            for subset in subsets:
                tree = self.innermodel.fit(subset)
                forest.append(tree)	
            i += 1

        return forest

    def classify(self, row, forest):
        results = []
        confidence = 0
        for node in forest:
            result = self.innermodel.classify(row, node)
            for key in result:
                if len(result) == 1:
                    results.append(key)
                else:
                    results.append(max(result, key=result.get))
                    break

        classification = max(set(results), key=results.count)
        confidence = round(results.count(max(set(results), key=results.count)) / len(results)*100, 2)
        return [classification, confidence]