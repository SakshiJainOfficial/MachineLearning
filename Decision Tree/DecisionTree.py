import copy
import numpy as np
import pandas as pd
'''
features and its values
used while predicting
'''
attributes = {
    'buying':['vhigh', 'high', 'med', 'low'],
    'maint':['vhigh', 'high', 'med', 'low'],
    'doors':['2', '3', '4', '5more'],
    'persons':['2','4','more'],
    'lug_boot':['small', 'med', 'big'],
    'safety':['low', 'med', 'high']
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
            value_label = data[data[attribute] == value]['label']
            totalentropy += (len(value_label) / len(data)) * (ID3.entropy(value_label))
        return totalentropy

    def informationgain(self,data,attribute):
        infogain = 0
        for value in attributes[attribute]:
            value_label = data[data[attribute] == value]['label']
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
        labels,counts = np.unique(data['label'],return_counts=True)
        if len(labels) == 1:            # First base case - only unique labels present in dataset - create leaf node
            node = TreeNode(None, depth, True, labels[0])
            return node
        if len(attributeList) == 0:   # second base case - no attribute to further divide the tree - create leaf node
            node = TreeNode(None,depth,True,None)
            if len(labels) != 0:
                maxcount = -1
                for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        node.label = leaflabel
            return node
        if depth == self.maxdepth:  # third base case - if maximum depth is reached - create leaf node
            node = TreeNode(None, depth, True, None)
            if len(labels) != 0:
                maxcount = -1
                for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        node.label = leaflabel
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
                maxcount = -1
                for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        child.label = leaflabel
            node.children[value] = child
        return node

    def predict(self,data,node):  # predict the output for a particular set of data
        if node.isleaf:
            return node.label
        feature_value = data[node.feature]
        child = node.children[feature_value]
        return self.predict(data,child)

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


colnames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
attribute = ['buying','maint','doors','persons','lug_boot','safety']
label_values = ['unacc', 'acc', 'good', 'vgood']
dataset = pd.read_csv("train.csv",names=colnames)
testdataset = pd.read_csv("test.csv",names = colnames)

#decision tree depth - 1 to 6
max_depth = int(input("Enter the max depth of the tree: "))
print("Train Errors:")
print("             Gini Index     Entropy     Majority Error    Average Error")
for i in range(1,max_depth+1):
    decisiontree_entropy = ID3(i, 0)
    decisiontree_gini = ID3(i, 1)
    decisiontree_me = ID3(i, 2)
    decisiontree_entropy.fit(dataset,copy.deepcopy(attribute),0)
    decisiontree_gini.fit(dataset, copy.deepcopy(attribute), 0)
    decisiontree_me.fit(dataset, copy.deepcopy(attribute), 0)
    predictedlabels_train_me = decisiontree_me.testdataset(dataset)
    predictedlabels_train_gini = decisiontree_gini.testdataset(dataset)
    predictedlabels_train_ent = decisiontree_entropy.testdataset(dataset)
    ginierror = ID3.calculateErrors(predictedlabels_train_gini, dataset['label'], label_values)
    entropyerror = ID3.calculateErrors(predictedlabels_train_ent, dataset['label'], label_values)
    majorityerror = ID3.calculateErrors(predictedlabels_train_me, dataset['label'], label_values)
    averageerror = (ginierror + entropyerror + majorityerror)/3
    print("Depth ", i,"     {:.4f}".format(ginierror),"        {:.4f}".format(entropyerror),"         {:.4f}".format(majorityerror),"       {:.4f}".format(averageerror))  # training error

print("Test Errors:")
print("             Gini Index     Entropy     Majority Error    Average Error")
for i in range(1, max_depth + 1):
    decisiontree_entropy_test = ID3(i, 0)
    decisiontree_gini_test = ID3(i, 1)
    decisiontree_me_test = ID3(i, 2)
    decisiontree_entropy_test.fit(dataset, copy.deepcopy(attribute), 0)
    decisiontree_gini_test.fit(dataset, copy.deepcopy(attribute), 0)
    decisiontree_me_test.fit(dataset, copy.deepcopy(attribute), 0)
    predictedlabels_test_me = decisiontree_me_test.testdataset(testdataset)
    predictedlabels_test_gini = decisiontree_gini_test.testdataset(testdataset)
    predictedlabels_test_ent = decisiontree_entropy_test.testdataset(testdataset)

    ginierror = ID3.calculateErrors(predictedlabels_test_gini, testdataset['label'], label_values)
    entropyerror = ID3.calculateErrors(predictedlabels_test_ent, testdataset['label'], label_values)
    majorityerror = ID3.calculateErrors(predictedlabels_test_me, testdataset['label'], label_values)
    averageerror = (ginierror + entropyerror + majorityerror) / 3
    print("Depth ", i, "     {:.4f}".format(ginierror), "        {:.4f}".format(entropyerror),"         {:.4f}".format(majorityerror),"         {:.4f}".format(majorityerror))  # training error
