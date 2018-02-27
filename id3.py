import pandas as pd 
import numpy as np
import math
import random
import copy
import sys

def calculateEntropy(currNode):
	posCount = currNode.posClassCount
	negCount = currNode.negClassCount
	if ((posCount == 0 and negCount != 0) or ( negCount == 0 and posCount != 0 )):
		return 0
	if (posCount == negCount):
		return 1
	p = posCount / (posCount+negCount)
	q = negCount / (posCount+negCount)
	return -(p * math.log(p,2)) - (q * math.log(q,2))

def calculateIG(currNode, leftChild, rightChild):
	return currNode.entropy - ( 
		((leftChild.posClassCount+leftChild.negClassCount)/(currNode.posClassCount+currNode.negClassCount) * leftChild.entropy)
		+ ((rightChild.posClassCount+rightChild.negClassCount)/(currNode.posClassCount+currNode.negClassCount) * rightChild.entropy) )


def calculatePosNegClassCounts(currNode):
	classValues = currNode.dataRows.Class # TO:DO - dont use Class, use the last column 
	classZero = 0
	classOne = 0
	for x in classValues:
		if x == 0:
			classZero = classZero + 1
		elif x == 1:
			classOne = classOne + 1	
	return classOne,classZero								

def printTree(root,s):
	parentAttr = root.attribute 
	if(not(root.left) and not(root.right)):
		if(root.negClassCount > root.posClassCount):
			print(" 0")
		else:
			print(" 1")
	else:
		print("\n",end="")
		if(root.left):
			op = s + parentAttr + "=" + "0" + ":"
			#op = op + "\n"
			s = s + "|  "
			print(op,end="")
			printTree(root.left , s)
		if(root.right):
			s = s[:-3]
			op = s + parentAttr + "=" + "1" + ":"
			#op = op + "\n"
			s = s + "|  "
			print(op,end="")
			printTree(root.right , s)


def createID3DecisionTree(currNode, allAttributes):
	if(len(allAttributes) == 0 ):
		return

	global NodeId
	bestIG = 0
	bestAttr = ""
	#print (allAttributes)
	for attr in allAttributes:
		##print (attr)
		leftChild = Node(columnNames=currNode.dataRows.columns) # TO: DO - optimise
		rightChild = Node(columnNames=currNode.dataRows.columns)
		mynparrayleft = leftChild.dataRows.values
		mynparrayright = rightChild.dataRows.values
		for i,x in currNode.dataRows.iterrows():
			if x[attr] == 0:
				mynparrayleft = np.vstack((mynparrayleft,x)) 
				#leftChild.dataRows.append(x, ignore_index=True)
			elif x[attr] == 1:
				mynparrayright = np.vstack((mynparrayright,x)) 
				#rightChild.dataRows.append(x, ignore_index=True)
		leftChild.dataRows = pd.DataFrame(mynparrayleft, columns=currNode.dataRows.columns)
		rightChild.dataRows = pd.DataFrame(mynparrayright, columns=currNode.dataRows.columns)
		##print (leftChild.dataRows)
		leftChild.posClassCount, leftChild.negClassCount = calculatePosNegClassCounts(leftChild)
		##print (attr ," : ", leftChild.posClassCount, " : ", leftChild.negClassCount) 
		leftChild.entropy = calculateEntropy(leftChild)
		rightChild.posClassCount, rightChild.negClassCount = calculatePosNegClassCounts(rightChild)
		rightChild.entropy = calculateEntropy(rightChild)
		currIG = calculateIG(currNode, leftChild, rightChild)
		##print (attr, " : " , currIG)
		if currIG >= bestIG: # do it atleast once 
			currNode.left = leftChild
			currNode.right = rightChild
			currNode.attribute = attr
			bestIG = currIG
			currNode.infoGain = currIG
	NodeId = NodeId + 1
	currNode.left.nodeId = NodeId
	NodeId = NodeId + 1
	currNode.right.nodeId = NodeId
	##print(currNode.attribute , " : " , currNode.infoGain)
	##print ("final-left:" , currNode.attribute ," : ", currNode.left.posClassCount, " : ", currNode.left.negClassCount)
	##print ("final-right:" , currNode.attribute ," : ", currNode.right.posClassCount, " : ", currNode.right.negClassCount)
	#allAttributes.remove(currNode.attribute)
	##print (allAttributes)
	if(currNode.left.posClassCount!=0 and currNode.left.negClassCount!=0):
		##print ("Went Left")
		createID3DecisionTree(currNode.left, [att for att in allAttributes if att != currNode.attribute])
		##print("back from left")
	if(currNode.right.posClassCount!=0 and currNode.right.negClassCount!=0):
		##print ("Went right")
		#allAttributes.append(currNode.attribute)
		createID3DecisionTree(currNode.right, [att for att in allAttributes if att != currNode.attribute])
		##print("back from right")
		#allAttributes.append(currNode.attribute)
	##print(currNode.posClassCount , currNode.negClassCount)

def testModel(model, data):
	if(model.left or model.right):
		##print(type(model.attribute))
		length = len(data)-1
		a = model.attribute
		if(data[a] == 0):
			if(model.left):
				return testModel(model.left, data)
		elif(data[a] == 1):
			if(model.right):
				return testModel(model.right, data)
	else:
		##print(data["Class"] , " : pos : " , model.posClassCount , " : neg : ", model.negClassCount)
		if(model.posClassCount > model.negClassCount):
			if(data["Class"] == 1):
				#print ("Here1")
				return 1
			else:
				#print ("Here2")
				return 0
		else:
			if(data["Class"] == 0):
				#print ("Here3")
				return 1
			else:
				#print ("Here4")
				return 0



class Node(object):
    def __init__(self, left = None, right = None, attribute = None, nodeId = None, infoGain = None, entropy = None, posClassCount = None, negClassCount = None, columnNames=None):
        self.left = left
        self.right = right
        self.attribute = attribute
        self.id = nodeId
        self.infoGain = infoGain
        self.entropy = entropy
        self.posClassCount = posClassCount
        self.negClassCount = negClassCount 
        self.dataRows = pd.DataFrame(columns=columnNames)


def totalNodeCount(node):
	if node is None:
		return 0
	if node.left is None and node.right is None:
		return 1
	else: 
		return 1 + totalNodeCount(node.left) +  totalNodeCount(node.right)

def leafNodeCount(node):
	if node is None:
		return 0
	if node.left is None and node.right is None:
		return 1
	else: 
		return leafNodeCount(node.left) +  leafNodeCount(node.right)

def printDecisionTree(currNode):
	if (currNode):
		print ("node id : ", currNode.nodeId ,"attribute : ", currNode.attribute, "Positive Count : ",currNode.posClassCount, 
			"Negative Count : ",currNode.negClassCount)
		print("left")
		printDecisionTree(currNode.left)
		print("Right")
		printDecisionTree(currNode.right)

def chooseRandomNodes(totalNodeCount, pruningFactor):
	return random.sample(range(4, totalNodeCount+1), int(totalNodeCount*pruningFactor)) # ignore the root node during pruning


def pruneTreeRecur(node , nodeId):
	if(node.nodeId == nodeId):
		node.left = None
		node.right = None
		return
	if(node.left): 
		pruneTreeRecur(node.left, nodeId)
	if(node.right): 
		pruneTreeRecur(node.right, nodeId)

def findAccuracy(root, testDataDict):
	correctData=0
	for x in testDataDict:
		result = testModel(root, x)
		if ( result ):
			correctData = correctData + 1
	return correctData / (len(testDataDict))

def pruneTree(root,pruningFactor,prePrunedAccuracy, validationDataDict, maxPrunes, pruneCount):
	tempTree = copy.deepcopy(root) # create a copy of the decision tree
	nodesToBeDeleted = chooseRandomNodes(totalNodeCount(tempTree), pruningFactor)
	#print ("Number of nodes to be pruned : ", len(nodesToBeDeleted), "from node count : ", totalNodeCount(tempTree))
	#if (pruneCount > maxPrunes):
	#	print("No improvement acheived while pruning in ", maxPrunes , "attempts. Thus, not pruning. Try running ID3 again!!")
	#	return tempTree
	for x in nodesToBeDeleted:
		#print("Pruning node : ", x)
		pruneTreeRecur(tempTree, x)
	newAccuracy = findAccuracy(tempTree, validationDataDict)
	#print(totalNodeCount(tempTree), newAccuracy)
	print("Pruning attempt ", pruneCount ,".....")#, " Accuracy : ", newAccuracy, " Node count : ", totalNodeCount(tempTree))
	while(not (newAccuracy >= prePrunedAccuracy) and (pruneCount < maxPrunes)):
		return pruneTree(root,pruningFactor,prePrunedAccuracy, validationDataDict, maxPrunes, pruneCount+1)
	return tempTree


#------------------------------------------------------------------------------------


trainingInputFile = sys.argv[1]
validationInputFile = sys.argv[2]
testInputFile = sys.argv[3]
pruningFactor = sys.argv[4]

trainingDataset = pd.read_csv(trainingInputFile)#("data_sets1/training_set.csv")
allAttributes = trainingDataset.loc[:, ~trainingDataset.columns.isin(["Class"])].columns.tolist()
traningDataDict = trainingDataset.to_dict('records')
##print (allAttributes) # fetches all column headers except "Class"
##print (df1.iloc[1]) #fetch row 2
##print (df1.ix[:,1]) # fetch column 2

validationDataset = pd.read_csv(validationInputFile)#("data_sets1/validation_set.csv")
validationDataDict = validationDataset.to_dict('records')

testDataset = pd.read_csv(testInputFile)#("data_sets1/test_set.csv")
testDataDict = testDataset.to_dict('records')

NodeId=1
root = Node()
root.dataRows = trainingDataset
root.posClassCount, root.negClassCount = calculatePosNegClassCounts(root)
root.entropy = calculateEntropy(root)
root.nodeId = NodeId
createID3DecisionTree(root, allAttributes )
#printDecisionTree(root)

printTree(root,"")

print("\n")
print("Pre-Pruned Accuracy")
print("--------------------------------")
print("Number of training instances = ", trainingDataset.shape[0])
print("Number of training attributes = ", trainingDataset.shape[1])
print("Total Number of nodes in the tree = ", totalNodeCount(root))
print("Number of leaf nodes in the tree = ", leafNodeCount(root))
prePrunedAccuracyTrain= findAccuracy(root, traningDataDict)
print ("Accuracy of the model on the training dataset : " , prePrunedAccuracyTrain*100, "%")
print("\n")
prePrunedAccuracyVal= findAccuracy(root, validationDataDict)
print("Number of validation instances = ", validationDataset.shape[0])
print("Number of validation attributes = ", validationDataset.shape[1])
print ("Accuracy of the model on the validation	dataset	before pruning : " , prePrunedAccuracyVal*100, "%")
print("\n")
prePrunedAccuracyTest= findAccuracy(root, testDataDict)
print("Number of test instances = ", testDataset.shape[0])
print("Number of test attributes = ", testDataset.shape[1])
print ("Accuracy of the model on the testing dataset : " , prePrunedAccuracyTest*100 , "%")
print("\n")

prunedTree = pruneTree(root,float(pruningFactor),prePrunedAccuracyVal,validationDataDict,3,1)


print("\n")
print("Pruned Accuracy")
print("--------------------------------")
print("Number of training instances = ", trainingDataset.shape[0])
print("Number of training attributes = ", trainingDataset.shape[1])
print("Total Number of nodes in the tree = ", totalNodeCount(prunedTree))
print("Number of leaf nodes in the tree = ", leafNodeCount(prunedTree))
postPrunedAccuracy= findAccuracy(prunedTree, traningDataDict)
print ("Accuracy of the model on the training dataset : " , postPrunedAccuracy*100 , "%")
print("\n")
postPrunedAccuracy= findAccuracy(prunedTree, validationDataDict)
print("Number of validation instances = ", validationDataset.shape[0])
print("Number of validation attributes = ", validationDataset.shape[1])
print ("Accuracy of the model on the validation	dataset	after pruning : " , postPrunedAccuracy*100, "%")
print("\n")
postPrunedAccuracy= findAccuracy(prunedTree, testDataDict)
print("Number of test instances = ", testDataset.shape[0])
print("Number of test attributes = ", testDataset.shape[1])
print ("Accuracy of the model on the testing dataset : " , postPrunedAccuracy*100, "%")

#printDecisionTree(root)
##print(root.attribute , root.posClassCount, root.negClassCount)
##print(root.left.attribute, root.left.posClassCount, root.left.negClassCount, root.right.attribute, root.right.posClassCount, root.right.negClassCount)
##print (root.left.left.posClassCount,root.left.left.negClassCount , root.right.posClassCount,root.right.negClassCount)







