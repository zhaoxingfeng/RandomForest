# coding:utf-8
"""
作者：zhaoxingfeng	日期：2017.06.12
功能：随机森林，Random Forest（RF），wine数据集[1,2]二分类
版本：1.0
"""
from __future__ import division
import pandas as pd
import copy
import random
import math

# 最后一个属性还不能将样本完全分开，此时数量最多的label被选为最终类别
def majorClass(classList):
    classDict = {}
    for cls in classList:
        classDict[cls] = classDict.get(cls, 0) + 1
    sortClass = sorted(classDict.items(), key=lambda item: item[1])
    return sortClass[-1][0]

# 计算基尼系数
def calcGini(dataSet):
    labelCounts = {}
    # 给所有可能分类创建字典
    for dt in dataSet:
        currentLabel = dt[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    Gini = 1
    for key in labelCounts:
        prob = labelCounts[key] / len(dataSet)
        Gini -= prob * prob
    return Gini

# 对连续变量划分数据集
def splitDataSet(dataSet, featIndex, value):
    leftData, rightData = [], []
    for dt in dataSet:
        if dt[featIndex] <= value:
            leftData.append(dt)
        else:
            rightData.append(dt)
    return leftData, rightData

# 选择最好的数据集划分方式
def chooseBestFeature(dataSet):
    bestGini = 1
    bestFeatureIndex = -1
    bestSplitValue = None
    # 第i个特征
    for i in range(len(dataSet[0]) - 1):
        featList = [dt[i] for dt in dataSet]
        # 产生候选划分点
        sortfeatList = sorted(list(set(featList)))
        splitList = []
        for j in range(len(sortfeatList) - 1):
            splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2)

        # 第j个候选划分点，记录最佳划分点
        for splitValue in splitList:
            newGini = 0
            subDataSet0, subDataSet1 = splitDataSet(dataSet, i, splitValue)
            newGini += len(subDataSet0) / len(dataSet) * calcGini(subDataSet0)
            newGini += len(subDataSet1) / len(dataSet) * calcGini(subDataSet1)
            if newGini < bestGini:
                bestGini = newGini
                bestFeatureIndex = i
                bestSplitValue = splitValue
    return bestFeatureIndex, bestSplitValue

# 去掉第i个属性，生成新的数据集
def splitData(dataSet, featIndex, features, value):
    newFeatures = copy.deepcopy(features)
    newFeatures.remove(features[featIndex])
    leftData, rightData = [], []
    for dt in dataSet:
        temp = []
        temp.extend(dt[:featIndex])
        temp.extend(dt[featIndex + 1:])
        if dt[featIndex] <= value:
            leftData.append(temp)
        else:
            rightData.append(temp)
    return newFeatures, leftData, rightData

# 建立决策树
def createTree(dataSet, features):
    classList = [dt[-1] for dt in dataSet]
    # label一样，全部分到一边
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 最后一个特征还不能把所有样本分到一边，则选数量最多的label
    if len(features) == 1:
        return majorClass(classList)
    bestFeatureIndex, bestSplitValue = chooseBestFeature(dataSet)
    bestFeature = features[bestFeatureIndex]
    # 生成新的去掉bestFeature特征的数据集
    newFeatures, leftData, rightData = splitData(dataSet, bestFeatureIndex, features, bestSplitValue)
    # 左右两颗子树，左边小于等于最佳划分点，右边大于最佳划分点
    myTree = {bestFeature: {'<' + str(bestSplitValue): {}, '>' + str(bestSplitValue): {}}}
    myTree[bestFeature]['<' + str(bestSplitValue)] = createTree(leftData, newFeatures)
    myTree[bestFeature]['>' + str(bestSplitValue)] = createTree(rightData, newFeatures)
    return myTree

# 用生成的决策树对测试样本进行分类
def treeClassify(decisionTree, featureLabel, testDataSet):
    firstFeature = decisionTree.keys()[0]
    secondFeatDict = decisionTree[firstFeature]
    splitValue = float(secondFeatDict.keys()[0][1:])
    featureIndex = featureLabel.index(firstFeature)
    if testDataSet[featureIndex] <= splitValue:
        valueOfFeat = secondFeatDict['<' + str(splitValue)]
    else:
        valueOfFeat = secondFeatDict['>' + str(splitValue)]
    if isinstance(valueOfFeat, dict):
        pred_label = treeClassify(valueOfFeat, featureLabel, testDataSet)
    else:
        pred_label = valueOfFeat
    return pred_label

# 随机抽取样本，样本数量与原训练样本集一样，维度为sqrt(m-1)
def baggingDataSet(dataSet):
    n, m = dataSet.shape
    features = random.sample(dataSet.columns.values[:-1], int(math.sqrt(m - 1)))
    features.append(dataSet.columns.values[-1])
    rows = [random.randint(0, n-1) for _ in range(n)]
    trainData = dataSet.iloc[rows][features]
    return trainData.values.tolist(), features

def testWine():
    df = pd.read_csv('wine.txt', header=None)
    labels = df.columns.values.tolist()
    df = df[df[labels[-1]] != 3]
    # 生成多棵决策树，放到一个list里边
    treeCounts = 10
    treeList = []
    for i in range(treeCounts):
        baggingData, bagginglabels = baggingDataSet(df)
        decisionTree = createTree(baggingData, bagginglabels)
        treeList.append(decisionTree)
    print treeList
    # 对测试样本分类
    labelPred = []
    for tree in treeList:
        testData = [12, 0.92, 2, 19, 86, 2.42, 2.26, 0.3, 1.43, 2.5, 1.38, 3.12, 278]
        label = treeClassify(tree, labels[:-1], testData)
        labelPred.append(label)
    # 投票选择最终类别
    labelDict = {}
    for label in labelPred:
        labelDict[label] = labelDict.get(label, 0) + 1
    sortClass = sorted(labelDict.items(), key=lambda item: item[1])
    print "The predicted label is: {}".format(sortClass[-1][0])
testWine()
