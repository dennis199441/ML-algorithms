import sys, operator
from collections import defaultdict
from itertools import chain, combinations

# transaction_data = [['milk', 'bananas', 'chocolate'],
#                     ['milk', 'chocolate'],
#                     ['milk', 'bananas'],
#                     ['chocolate'],
#                     ['chocolate'],
#                     ['milk', 'chocolate']]

class AprioriAlgorithm:
    def __init__(self, minSupport=0.15, minConfidence=0.6):
        self.minSupport = minSupport
        self.minConfidence = minConfidence

    def generate_itemSet_transactionList(self, dataset):
        itemSet = set()
        transactionList = list()
        for data in dataset:
            transaction = frozenset(data)
            transactionList.append(transaction)
            for item in transaction:
                itemSet.add(frozenset([item]))

        return itemSet, transactionList

    def getItemsWithMinSupport(self, itemSet, transactionList, freqSet):
        resultSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
            for transaction in transactionList:
                if item.issubset(transaction):
                    self.freqSet[item] += 1
                    localSet[item] += 1

        for item, count in localSet.items():
            support = float(count)/len(transactionList)
            if support >= self.minSupport:
                resultSet.add(item)

        return resultSet

    def joinSet(self, itemSet, length):
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

    def subsets(self, arr):
        return chain(*[combinations(arr, i+1) for i, a in enumerate(arr)])

    def fit(self, data):
        self.itemSet, self.transactionList = self.generate_itemSet_transactionList(data)
        self.freqSet = defaultdict(int)
        largeSet = dict() ## stores (keys = n - itemSets, value = support)
        assoRules = dict()

        itemWithMinSupport = self.getItemsWithMinSupport(self.itemSet, self.transactionList, self.freqSet)

        currentLSet = itemWithMinSupport
        k = 2
        while(currentLSet != set([])):
            largeSet[k-1] = currentLSet
            currentLSet = self.joinSet(currentLSet, k)
            currentCSet = self.getItemsWithMinSupport(currentLSet, self.transactionList, self.freqSet)
            k += 1

        RetItems = []
        for key, value in largeSet.items():
            RetItems.extend([(tuple(item), self.getSupport(item)) for item in value])

        RetRules = []
        for key, value in largeSet.items():
            for item in value:
                _subsets = map(frozenset, [x for x in self.subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        confidence = self.getSupport(item) / self.getSupport(element)
                        if confidence >= self.minConfidence:
                            RetRules.append(((tuple(element), tuple(remain)), confidence))

        return RetItems, RetRules

    def getSupport(self, item):
        return float(self.freqSet[item]) / len(self.transactionList)

    def printResults(self, items, rules):
        for item, support in sorted(items, key=operator.itemgetter(1)):
            print("item: %s , %.3f" % (str(item), support))

        print("\n------------------ RULES: ")
        for rule, confidence in sorted(rules, key=operator.itemgetter(1)):
            pre, post = rule
            print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))