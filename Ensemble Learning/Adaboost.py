import numpy as np
import pandas as pd
import copy
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
                node.label = labels[counts == counts.max()][0]
            return node
        if depth == self.maxdepth: # third base case - if maximum depth is reached - create leaf node
            node = TreeNode(None, depth, True, None)
            if len(labels) != 0:
                #maxcount = -1
                maxsum = 0
                for i in labels:
                    index = data.index[data['y'] == i]
                    weigths_label = [weights[i] for i in index]
                    label_weightage = sum(weigths_label)
                    if maxsum < label_weightage:
                        node.label = i
                        maxsum = label_weightage
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
                maxsum = 0
                for i in labels:
                    index = data.index[data['y'] == i]
                    weigths_label = weights[index]
                    label_weightage = sum(weigths_label)
                    if maxsum < label_weightage:
                        child.label = i
                        maxsum = label_weightage
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


class Adaboost:
    def __init__(self,iterations):
        self.modelweights = []
        self.decisionstumps = []
        self.training_iterations = iterations
        self.decisionstumps_errors = []
        self.decisionstumps_testerror = []

    def fit(self, data):
        weights = [1/data.shape[0] for i in range(data.shape[0])]
        attributeList = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing','loan','contact','day','month',
            'duration','campaign','pdays','previous','poutcome']
        predictedlabels = []
        for i in range(self.training_iterations):
            decisionStump = ID3(1,0)
            decisionStump.fit(data,copy.deepcopy(attributeList),0,weights)
            self.decisionstumps.append(decisionStump)
            for j in range(len(data)):
                predictedlabels.append(decisionStump.predict(data.iloc[j],self.decisionstumps[i].root))
            predictedint = [1 if lable == "yes" else -1 for lable in predictedlabels]
            targetlabels = [1 if lable == "yes" else -1 for lable in data['y']]
            self.decisionstumps_errors.append(self.calculateErrors(predictedint,targetlabels))
            error = sum(weights * (np.not_equal(targetlabels,predictedint)).astype(int))
            self.modelweights.append(1/2 * np.log((1-error)/error))
            weights = [weights[x] * np.exp(-self.modelweights[i]*predictedint[x]*targetlabels[x]) for x in range(len(weights))]
            weights = weights/np.sum(weights)
            predictedlabels = []

    def predict(self,data,root):
        id3obj = ID3(1,0)
        return id3obj.predict(data,root)

    def testdataset(self,testdataset):
        predictedlabels = []
        predictedyn = []
        predictions = []

        for i in range(len(testdataset)):
            for j in range(len(self.decisionstumps)):
                predictions.append(self.predict(testdataset.iloc[i], self.decisionstumps[j].root))
            predicted = [1 if label == "yes" else -1 for label in predictions]
            sum_predicted = np.dot(predicted,self.modelweights)
            predictedlabels.append(np.sign(sum_predicted))
            if sum_predicted>0:
                predictedyn.append("yes")
            else:
                predictedyn.append("no")
            predictions = []
            #predicted = []
        return predictedyn

    @staticmethod
    def calculateErrors(predictedlabels, targetlabels):
        correctlypredicted = 0
        for predicted,actual in zip(predictedlabels,targetlabels):
            if predicted == actual:
                correctlypredicted += 1
        return 1 - correctlypredicted/len(predictedlabels)

if __name__ == '__main__':
    colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                'month','duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    dataset = pd.read_csv("Bank Dataset/train.csv", names=colnames)
    testdataset = pd.read_csv("Bank Dataset/test.csv", names=colnames)
    for i in dataset.columns[dataset.dtypes == 'int64']:
        dataset[i] = dataset[i].apply(lambda x: 1 if x > dataset[i].median() else 0)

    for i in testdataset.columns[testdataset.dtypes == 'int64']:
        testdataset[i] = testdataset[i].apply(lambda x: 1 if x > testdataset[i].median() else 0)
    print("T             training errors           test errors")
    trainingerror = []
    testerrors = []
    T = [x for x in range(1,501)]
    obj = None
    for i in range(1, 501):
        ad = Adaboost(i)
        ad.fit(dataset)
        predictedlabels_train = ad.testdataset(dataset)
        predictedlabels_test = ad.testdataset(testdataset)
        trainingerror.append(Adaboost.calculateErrors(predictedlabels_train, dataset['y']))
        testerrors.append(Adaboost.calculateErrors(predictedlabels_test, testdataset['y']))
        print(i,"           ","%.4f"%trainingerror[i-1],"                   ","%.4f"%testerrors[i-1])
        obj = ad
    predictedlabels_ds = []
    for i in range(obj.training_iterations):
        for j in range(len(testdataset)):
            predictedlabels_ds.append(obj.decisionstumps[i].predict(testdataset.iloc[j], obj.decisionstumps[i].root))
        predictedint = [1 if lable == "yes" else -1 for lable in predictedlabels_ds]
        targetlabels = [1 if lable == "yes" else -1 for lable in testdataset['y']]
        obj.decisionstumps_testerror.append(Adaboost.calculateErrors(predictedint, targetlabels))
    print(obj.decisionstumps_testerror)
    print(obj.decisionstumps_errors)
    print(trainingerror)
    print(testerrors)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.plot(T, trainingerror, color='blue', label="Training Errors")
    ax1.plot(T, testerrors, color='red', label="Test Error")
    ax1.set_xlabel('No. of decision stumps (T)')
    ax1.set_ylabel('Errors')
    ax1.set_title("T vs Training/Test error")
    ax1.legend()
    ax2.plot(T,obj.decisionstumps_errors, color='blue', label="Training Errors(For Individual Decision Stumps)")
    ax2.plot(T,obj.decisionstumps_testerror,color= 'red',label ="Test Errors(For Individual Decision Stumps)")
    ax2.set_xlabel('decision stump #')
    ax2.set_ylabel('Errors')
    ax2.set_title("Error of individual decision stump")
    ax2.legend()
    plt.tight_layout()
    plt.show()

