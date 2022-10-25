# CS-6350-Machine-Learning-UoU
This is a machine learning library developed by Harsh Mahajan for CS5350/6350 in University of Utah

### How to use decision tree?
DecisionTree.py file contains ID3 class. To create a decision tree, create an object of ID3 class with following parameters - 
 - Max Depth -> the maximum depth required for decision tree.(Default Value - None)
 - Information gain -> How you want to calculate information gain? (Pass 0 - entropy, 1 - gini value, 2 - majority error)

Methods in decision tree -
 - fit(data,attributeList,depth)
    data: dataset to train decision tree
    attributeList: different features(columns) present in our data
    depth: current depth of decision tree. (Default value = 0)
     
 - testdataset(testdata)
    testdata: test dataset to test our model
 
 - calculateErrors(predictedlabels, targetlabels)
    predictedlabels: predicted labels from our trained model.
    targetlabels: actual labels from dataset
    
 To train model- 
 tree = ID3(None, 0)
 tree.fit(data,attributes)
 
 To test dataset- 
 predicted = tree.testdataset(testdata)
 
 To calculate errors- 
 tree.calculateErrors(predicted,targetlabels)
