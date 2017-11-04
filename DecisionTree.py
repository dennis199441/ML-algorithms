class DecisionTree:

    ## Find the unique values for a column in dataset
    def unique_vals(self, rows, col):
        return set([row[col] for row in rows])

    ## Count the number of each type of example in dataset
    def class_counts(self, rows):
        counts = {}
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0

            counts[label] += 1

        return counts

    ## Test if value is numeric
    def isNumeric(self, value):
        return isinstance(value, int) or isinstance(value, float)

    def partition(self, rows, question):
        ## Partition the dataset.
        ## For each row in the dataset, check if it matches the question. 
        ## If yes, add it to 'true rows', otherwise, add to 'false rows' 
        true_rows = []
        false_rows = []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)

        return true_rows, false_rows

    def gini(self, rows):
        ## Calculate the Gini Inpurity for a list of rows.
        counts = self.class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2

        return impurity

    def info_gain(self, left, right, current_uncertainty):
        ## Information Gain.
        ## The uncertainty of the starting node, minus the weighted impurity of
        ## two child nodes.
        p = float(len(left)) / (len(left) + len(right))

        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

    def find_best_split(self, rows):
        ## Find the best question to ask by iterating over every feature / value
        ## and calculating the Information Gain
        best_gain = 0
        best_question = None
        current_uncertainty = self.gini(rows)
        n_features = len(rows[0]) - 1

        for col in range(n_features):
            values = set([row[col] for row in rows])
            for val in values:
                question = Question(col, val)
                true_rows, false_rows = self.partition(rows, question)

                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                gain = self.info_gain(true_rows, false_rows, current_uncertainty)
                if gain >= best_gain:
                    best_gain = gain
                    best_question = question

        return best_gain, best_question

    def fit(self, rows):
        ## Build the tree
        ## Rules of recursion:
        ## 1) Believe that it works.
        ## 2) Start by checking for the base case (no further information gain).
        ## 3) Prepare for giant stack traces.
        gain, question = self.find_best_split(rows)

        if gain == 0:
            return Leaf(rows)

        true_rows, false_rows = self.partition(rows, question)
        true_branch = self.fit(true_rows)
        false_branch = self.fit(false_rows)

        return Decision_Node(question, true_branch, false_branch)

    def print_tree(self, node, spacing=""):
        if isinstance(node, Leaf):
            print(spacing + "Predict", node.predictions)
            return

        print(spacing + str(node.question))
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

    def classify(self, row, node):
        if isinstance(node, Leaf):
            return node.predictions

        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    def print_leaf(self, counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"

        return probs

class Question(DecisionTree):
    ## This class is used to partition a dataset.
    ## This class just records a column number such as 0 for color and
    ## a column value such as green. The match method is used to compare 
    ## the feature value in an example to the feature value stored in the question.
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if self.isNumeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        ## Print question in a readable format
        condition = "=="
        if self.isNumeric(self.value):
            condition = ">="

        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))

class Leaf(DecisionTree):
    ## Leaf node classifies data.
    ## This holds a dictionary of class such as apple -> number of times
    ## It appears in the rows from the training data that reach this leaf.
    def __init__(self, rows):
        self.predictions = self.class_counts(rows)

class Decision_Node(DecisionTree):
    ## Decision nodes ask a question
    ## This holds a reference to the question, and to the two child nodes.
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch