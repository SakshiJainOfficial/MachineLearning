import copy
import numpy as np
import pandas as pd

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
    def weighted_gini(labels,weights):
        probability = {}
        uniquelabels = np.unique(labels)
        for i in uniquelabels:
            probability[i] = 0
        for label, weight in zip(labels, weights):
            probability[label] += weight
        probabilitysquare = [probability[label]*probability[label] for label in probability]
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

    @staticmethod
    def weightedentropy(data,labels,weights):
        probability = {}
        uniquelabels = np.unique(labels)
        s = sum(weights)
        for i in uniquelabels:
            probability[i] = 0
        for label,weight in zip(labels,weights):
            probability[label] += weight
        for i in uniquelabels:
            probability[i] = probability[i]/s
        attribute_entropy = 0
        for i in np.unique(labels):
            attribute_entropy -= probability[i] * np.log(probability[i])
        return attribute_entropy

    def weightedinformationgain(self,data,attribute,weights):
        information_gain = 0
        for value in attributes[attribute]:
            indexes = data.index[data[attribute] == value]
            attr_weights = [weights[i] for i in indexes]
            #attr_weights = weights.iloc[[data.index[data[attribute] == value]]]
            value_label = data[data[attribute] == value]['y']
            information_gain += (sum(attr_weights)/sum(weights)) * ID3.weightedentropy(data,value_label,attr_weights)
            #information_gain += (len(value_label) / len(data)) * ID3.weighted_gini(value_label, weights)
            #information_gain += sum(attr_weights) * ID3.weighted_gini(value_label, weights)
        return information_gain

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


    def fit(self,data, attributeList, depth, weights):
        minentropy = 999
        lowest_entropy_attr = None
        labels,counts = np.unique(data['y'],return_counts=True)
        if len(labels) == 1:   # First base case - only unique labels present in dataset - create leaf node
            node = TreeNode(None, depth, True, labels[0])
            return node
        if len(attributeList) == 0:  # second base case - no attribute to further divide the tree - create leaf node
            node = TreeNode(None,depth,True,None)
            if len(labels) != 0:
                maxcount = -1
                for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        node.label = leaflabel
            return node
        if depth == self.maxdepth: # third base case - if maximum depth is reached - create leaf node
            node = TreeNode(None, depth, True, None)
            if len(labels) != 0:
                maxcount = -1
                maxsum = 0
                for i in labels:
                    index = data.index[data['y'] == i]
                    weigths_label = [weights[i] for i in index]
                    label_weightage = sum(weigths_label)
                    if maxsum < label_weightage:
                        node.label = i
                        maxsum = label_weightage
                '''for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        node.label = leaflabel'''
            return node
        if len(data) == 0: # fourth base case - No data present - create leaf node with maximum labels as output
            return TreeNode(None,depth,True,None)
        for a in attributeList: # selecting best attribute to divide the tree
            entropy_attr = self.weightedinformationgain(data,a,weights)
            if minentropy > entropy_attr:
                minentropy = entropy_attr
                lowest_entropy_attr = a

        node = TreeNode(lowest_entropy_attr, depth, False, None) # creating node for the selected attribute
        if self.root is None:
            self.root = node
        attributeList.remove(lowest_entropy_attr) # removing attribute for the attribute list
        for value in attributes[lowest_entropy_attr]: # constructing tree for all branches of that node
            new_data = data[data[lowest_entropy_attr] == value]
            attr = copy.deepcopy(attributeList)
            child = self.fit(new_data,attr,depth+1,weights) # recursively calling the function to create tree for the branch
            if child.isleaf is True and child.label is None:
                maxcount = -1
                maxsum = 0
                for i in labels:
                    index = data.index[data['y'] == i]
                    weigths_label = weights[index]
                    label_weightage = sum(weigths_label)
                    if maxsum < label_weightage:
                        child.label = i
                        maxsum = label_weightage
                '''for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        child.label = leaflabel'''
            node.children[value] = child
        return node

    def predict(self,data,node):  # predict the output for a particular set of data
        if node.isleaf:
            return node.label
        feature_value = data[node.feature]
        child = node.children[feature_value]
        return self.predict(data,child)

    @staticmethod
    def calculateErrors(predictedlabels, targetlabels):  # calculate error using the predicted label and actual labels
        y_actual = pd.Series(targetlabels,name = 'Actual')
        y_predicted = pd.Series(predictedlabels,name = 'Predicted')
        confusion_matrix = pd.crosstab(y_actual,y_predicted,margins=True)
        labels = np.unique(predictedlabels)
        correctly_predicted = 0
        for label in labels:
            correctly_predicted += confusion_matrix.loc[label,label]
        return 1-correctly_predicted/len(predictedlabels)

    def testdataset(self,testdataset):  # calling predict function for each set of data in our dataset
        predictedlabels = []
        for i in range(len(testdataset)):
            predictedlabels.append(self.predict(testdataset.iloc[i], self.root))
        return predictedlabels
