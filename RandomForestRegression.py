# coding:utf-8
"""
作者：zhaoxingfeng	日期：2017.06.13
功能：随机森林，Random Forest（RF），housing数据集回归
版本：1.0
"""
from __future__ import division
import pandas as pd
import numpy as np
import copy
import random
import math

# 对连续变量划分数据集，返回数据只包括最后一列
def splitDataSet(dataSet, featIndex, value):
    leftData, rightData = [], []
    for dt in dataSet:
        if dt[featIndex] <= value:
            leftData.append(dt[-1])
        else:
            rightData.append(dt[-1])
    return leftData, rightData

# 选择最好的数据集划分方式，使得误差平方和最小
def chooseBestFeature(dataSet):
    bestR2 = float('inf')
    bestFeatureIndex = -1
    bestSplitValue = None
    # 第i个特征
    for i in range(len(dataSet[0]) - 1):
        featList = [dt[i] for dt in dataSet]
        # 产生候选划分点
        sortfeatList = sorted(list(set(featList)))
        splitList = []
        # 如果值相同，不存在候选划分点
        if len(sortfeatList) == 1:
            splitList.append(sortfeatList[0])
        else:
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2)
        # 第j个候选划分点，记录最佳划分点
        for splitValue in splitList:
            subDataSet0, subDataSet1 = splitDataSet(dataSet, i, splitValue)
            lenLeft, lenRight = len(subDataSet0), len(subDataSet1)
            # 防止数据集为空，mean不能计算
            if lenLeft == 0 and lenRight != 0:
                rightMean = np.mean(subDataSet1)
                R2 = sum([(x - rightMean)**2 for x in subDataSet1])
            elif lenLeft != 0 and lenRight == 0:
                leftMean = np.mean(subDataSet0)
                R2 = sum([(x - leftMean) ** 2 for x in subDataSet0])
            else:
                leftMean, rightMean = np.mean(subDataSet0), np.mean(subDataSet1)
                leftR2 = sum([(x - leftMean)**2 for x in subDataSet0])
                rightR2 = sum([(x - rightMean)**2 for x in subDataSet1])
                R2 = leftR2 + rightR2
            if R2 < bestR2:
                bestR2 = R2
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
def regressionTree(dataSet, features):
    classList = [dt[-1] for dt in dataSet]
    # label一样，全部分到一边
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 最后一个特征还不能把所有样本分到一边，则划分到平均值
    if len(features) == 1:
        return np.mean(classList)
    bestFeatureIndex, bestSplitValue = chooseBestFeature(dataSet)
    bestFeature = features[bestFeatureIndex]
    # 删除root特征，生成新的去掉root特征的数据集
    newFeatures, leftData, rightData = splitData(dataSet, bestFeatureIndex, features, bestSplitValue)

    # 左右子树有一个为空，则返回该节点下样本均值
    if len(leftData) == 0 or len(rightData) == 0:
        return np.mean([dt[-1] for dt in leftData] + [dt[-1] for dt in rightData])
    else:
        # 左右子树不为空，则继续分裂
        myTree = {bestFeature: {'<' + str(bestSplitValue): {}, '>' + str(bestSplitValue): {}}}
        myTree[bestFeature]['<' + str(bestSplitValue)] = regressionTree(leftData, newFeatures)
        myTree[bestFeature]['>' + str(bestSplitValue)] = regressionTree(rightData, newFeatures)
    return myTree

# 用生成的回归树对测试样本进行测试
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

def testHousing():
    df = pd.read_csv('housing.txt')
    labels = df.columns.values.tolist()
    # 生成多棵回归树，放到一个list里边
    treeCounts = 10
    treeList = []
    for i in range(treeCounts):
        baggingData, bagginglabels = baggingDataSet(df)
        decisionTree = regressionTree(baggingData, bagginglabels)
        treeList.append(decisionTree)
    print treeList
    # 对测试样本求预测值
    labelPred = []
    for tree in treeList:
        testData = [0.38735,0,25.65,0,0.581,5.613,95.6,1.7572,2,188,19.1,359.29,27.26]
        label = treeClassify(tree, labels[:-1], testData)
        labelPred.append(label)
    print "The predicted value is: {}".format(np.mean(labelPred))
testHousing()
