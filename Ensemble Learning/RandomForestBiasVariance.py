import copy
import numpy as np
import pandas as pd
import statistics as st
from collections import Counter
import random
import matplotlib.pyplot as plt
'''
features and its values
used while predicting
'''
attributes = {
    'age':[0,1],
    'job':["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"],
    'marital':["married","divorced","single"],
    'education':["unknown","secondary","primary","tertiary"],
    'default':["yes","no"],
    'balance':[0,1],
    'housing':["yes","no"],
    'loan':["yes","no"],
    'contact':["unknown","telephone","cellular"],
    'day':[0,1],
    'month':["jan", "feb", "mar","apr","may","jun","jul","aug","sep","oct", "nov", "dec"],
    'campaign':[0,1],
    'pdays':[0,1],
    'previous':[0,1],
    'poutcome':["unknown","other","failure","success"],
    'duration':[0,1]
}

'''
Class for Node of the tree having attributes
feature - name of the tree node(representing feature in our data)
children - dictionary saving all the childs for this feature
depth - depth at which this feature is present in the tree
isleaf - boolean value, showing is the node leaf node or not
label - only used for leaf node, showing final classification output
'''
class TreeNode:
    def __init__(self, feature, depth, isleaf, label):
        self.feature = feature
        self.children = {}
        self.depth = depth
        self.isleaf = isleaf
        self.label = label


'''
ID3 class 
Contains tree construction function and prediction function and other utility functions
'''
class ID3:
    def __init__(self, maxdepth,infogain):
        self.root = None
        self.maxdepth = maxdepth
        self.infogainmethod = infogain

    @staticmethod
    def entropy(labels):  ##entropy calculation
        label, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        logs = np.log2(probabilities)
        entropyattr = 0
        for i in range(len(probabilities)):
            entropyattr -= probabilities[i] * logs[i]
        return entropyattr

    @staticmethod
    def gini(labels):
        label, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        probabilitysquare = [probability*probability for probability in probabilities]
        return 1 - sum(probabilitysquare)

    @staticmethod
    def majorityerror(labels):
        label,counts = np.unique(labels, return_counts=True)
        if len(label) == 1:
            return 0
        probabilities = counts / len(labels)
        return min(probabilities)

    def totalEntropy(self, data, attribute):
        totalentropy = 0
        func = None
        for value in attributes[attribute]:
            value_label = data[data[attribute] == value]['y']
            totalentropy += (len(value_label) / len(data)) * (ID3.entropy(value_label))
        return totalentropy

    def informationgain(self,data,attribute):
        infogain = 0
        for value in attributes[attribute]:
            value_label = data[data[attribute] == value]['y']
            if self.infogainmethod == 0:
                infogain += (len(value_label) / len(data)) * (ID3.entropy(value_label))
            elif self.infogainmethod == 1:
                infogain += (len(value_label) / len(data)) * (ID3.gini(value_label))
            else:
                if len(value_label) == 0:
                    return 0
                infogain += (len(value_label) / len(data)) * (ID3.majorityerror(value_label))
        return infogain


    def fit(self,data, attributeList, depth):
        minentropy = 999
        lowest_entropy_attr = None
        labels,counts = np.unique(data['y'],return_counts=True)
        if len(labels) == 1:            # First base case - only unique labels present in dataset - create leaf node
            node = TreeNode(None, depth, True, labels[0])
            return node
        if len(attributeList) == 0:   # second base case - no attribute to further divide the tree - create leaf node
            node = TreeNode(None,depth,True,None)
            if len(labels) != 0:
                node.label = labels[counts == counts.max()][0]
            return node
        if depth == self.maxdepth:  # third base case - if maximum depth is reached - create leaf node
            node = TreeNode(None, depth, True, None)
            if len(labels) != 0:
                node.label = labels[counts == counts.max()][0]
            return node
        if len(data) == 0:  # fourth base case - No data present - create leaf node with maximum labels as output
            return TreeNode(None,depth,True,None)
        for a in attributeList:  # selecting best attribute to divide the tree
            entropy_attr = self.informationgain(data,a)
            if minentropy > entropy_attr:
                minentropy = entropy_attr
                lowest_entropy_attr = a

        node = TreeNode(lowest_entropy_attr, depth, False, None)  # creating node for the selected attribute
        if self.root is None:
            self.root = node
        attributeList.remove(lowest_entropy_attr) # removing attribute for the attribute list
        for value in attributes[lowest_entropy_attr]:  # constructing tree for all branches of that node
            new_data = data[data[lowest_entropy_attr] == value]
            attr = copy.deepcopy(attributeList)
            child = self.fit(new_data,attr,depth+1)  # recursively calling the function to create tree for the branch
            if child.isleaf is True and child.label is None:
                child.label = labels[counts == counts.max()][0]
            node.children[value] = child
        return node

    @staticmethod
    def predict(data,node):  # predict the output for a particular set of data
        if node.isleaf:
            return node.label
        feature_value = data[node.feature]
        child = node.children[feature_value]
        return ID3.predict(data,child)

    @staticmethod
    def calculateErrors(predictedlabels, targetlabels, label_values):  # calculate error using the predicted label and actual labels
        y_actual = pd.Series(targetlabels,name = 'Actual')
        y_predicted = pd.Series(predictedlabels,name = 'Predicted')
        confusion_matrix = pd.crosstab(y_actual,y_predicted,margins=True)
        labels = np.unique(predictedlabels)
        correctly_predicted = 0
        for label in labels:
            correctly_predicted += confusion_matrix.loc[label,label]
        return 1-correctly_predicted/len(predictedlabels)

    def testdataset(self,testdataset): # calling predict function for each set of data in our dataset
        predictedlabels = []
        for i in range(len(testdataset)):
            predictedlabels.append(self.predict(testdataset.iloc[i], self.root))
        return predictedlabels


class RandomForest:
    def __init__(self,no_of_trees,feature_size):
        self.trees = no_of_trees    #no of tree in random forest
        self.models = []
        self.features_size = feature_size

    def fit(self,dataset):
        attributeList = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                         'day', 'month',
                         'duration', 'campaign', 'pdays', 'previous', 'poutcome']
        for i in range(self.trees):
            arr = np.random.uniform(0,dataset.shape[0],int(0.6*dataset.shape[0]))  #bootstrapped sampling
            indexes = [int(j) for j in arr]
            new_dataset = dataset.iloc[indexes]
            tree = ID3(None,0)
            tree.fit(new_dataset,copy.deepcopy(attributeList),0,self.features_size) #fitting data on bootstraped samples
            self.models.append(tree)

    @staticmethod
    def predict(data,root):
        return ID3.predict(data,root)

    def my_mode(self,sample):
        c = Counter(sample)
        return [k for k, v in c.items() if v == c.most_common(1)[0][1]][0]

    def testdataset(self,testdataset):
        predicted_models = []
        final_predictions = []
        for i in range(testdataset.shape[0]):
            for j in range(len(self.models)):
                predicted_models.append(RandomForest.predict(testdataset.iloc[i],self.models[j].root))
            final_predictions.append(self.my_mode(predicted_models))
            predicted_models = []
        return final_predictions

    @staticmethod
    def calculateErrors(predictedlabels, targetlabels):
        # return ID3.calculateErrors(predictedlabels,targetlabels)
        correctlypredicted = 0
        for predicted, actual in zip(predictedlabels, targetlabels):
            if predicted == actual:
                correctlypredicted += 1
        return 1 - correctlypredicted / len(predictedlabels)


if __name__ == '__main__':
    colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    dataset = pd.read_csv("Bank Dataset/train.csv", names=colnames)
    testdataset = pd.read_csv("Bank Dataset/test.csv", names=colnames)
    for i in dataset.columns[dataset.dtypes == 'int64']:
        dataset[i] = dataset[i].apply(lambda x: 1 if x > dataset[i].median() else 0)

    for i in testdataset.columns[testdataset.dtypes == 'int64']:
        testdataset[i] = testdataset[i].apply(lambda x: 1 if x > testdataset[i].median() else 0)
    bagged_predictors = []
    first_trees = []
    for i in range(100):
        random_indexes = random.sample(range(dataset.shape[0]),1000)
        new_dataset = dataset.iloc[random_indexes]
        bg = RandomForest(500)
        bg.fit(new_dataset)
        bagged_predictors.append(bg)
        first_trees.append(bg.models[0])
    predictions = []
    biasdt = []
    variancedt = []
    targetlabels = [1 if lable == "yes" else -1 for lable in testdataset['y']]

    ## Bias and Variance for single decision tree learners
    for i in range(len(testdataset)):
        bias = 0
        variance = 0
        for j in range(len(first_trees)):  #for every sample in testdata, predict with every single decision tree
            predictions.append(ID3.predict(testdataset.iloc[i],first_trees[j].root))
        pred = [1 if lable == "yes" else -1 for lable in predictions]
        target = 1 if testdataset.iloc[i]['y'] == "yes" else -1
        bias = np.square(np.mean(pred) - target)   #calculating bias
        variance = sum(np.square(np.mean(targetlabels) - pred))/(len(pred)) #calculating variance
        predictions = []
        biasdt.append(bias)
        variancedt.append(variance)
    print("bias for single decision tree learner: ",np.mean(biasdt))
    print("variance for single decision tree learner: ", np.mean(variancedt))
    print("General squared error: ",np.mean(biasdt)+np.mean(variancedt))

    ## Bias and Variance for Bagged Predictors
    predictions_bg = {}
    biasbt = []
    variancebt = []
    for i in range(len(bagged_predictors)):
        predictions_bg[i] = bagged_predictors[i].testdataset(testdataset)
    for i in range(len(dataset)):
        bias = 0
        variance = 0
        pred = [1 if predictions_bg[lable][i] == "yes" else -1 for lable in predictions_bg]
        target = 1 if testdataset.iloc[i]['y'] == "yes" else -1
        bias = np.square(np.mean(pred) - target)
        variance = sum(np.square(np.mean(targetlabels) - pred))/(len(pred))
        biasbt.append(bias)
        variancebt.append(variance)
    print("bias for bagged tree predictor: ", np.mean(biasdt))
    print("variance for bagged tree learner: ", np.mean(variancedt))
    print("General squared error: ", np.mean(biasdt) + np.mean(variancedt))