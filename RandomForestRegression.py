# -*- coding: utf-8 -*-
"""
@Env:Python2.7
@Time: 2019/10/24 14:22
@Author: zhaoxingfeng
@Function：Random Forest（RF），随机森林回归
@Version: V1.1
参考文献：
[1] UCI. housing[DB/OL].https://archive.ics.uci.edu/ml/machine-learning-databases/housing.
"""
import pandas as pd
import numpy as np
import random
import math
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


# 定义一棵决策树
class Tree(object):
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    # 通过递归决策树找到样本所属叶子节点
    def calc_predict_value(self, dataset):
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    # 以json形式打印决策树，方便查看树结构
    def describe_tree(self):
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForestRegression(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree="sqrt", subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree  # 列采样
        self.subsample = subsample  # 行采样
        self.random_state = random_state
        self.trees = dict()
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        targets = targets.to_frame(name='label')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # 两种列采样方式
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        for stage in range(self.n_estimators):
            print(("iter: "+str(stage+1)).center(80, '='))

            # bagging方式随机选择样本和特征
            random.seed(random_state_stages[stage])
            subset_index = random.sample(range(len(dataset)), int(self.subsample * len(dataset)))
            subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
            dataset_copy = dataset.loc[subset_index, subcol_index].reset_index(drop=True)
            targets_copy = targets.loc[subset_index, :].reset_index(drop=True)

            tree = self._fit(dataset_copy, targets_copy, depth=0)
            self.trees[stage] = tree
            print(tree.describe_tree())

    # 递归建立决策树
    def _fit(self, dataset, targets, depth):
        # 如果该节点的类别全都一样/样本小于分裂所需最小样本数量，则选取出现次数最多的类别。终止分裂
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # 如果分裂的时候用到该特征，则该特征的importance加1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._fit(left_dataset, left_targets, depth+1)
                tree.tree_right = self._fit(right_dataset, right_targets, depth+1)
                return tree
        # 如果树的深度超过预设值，则终止分裂
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    # 选择最好的数据集划分方式，找到最优分裂特征、分裂阈值、分裂增益
    def choose_best_feature(self, dataset, targets):
        best_split_gain = float("inf")
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # 对可能的分裂阈值求分裂增益，选取增益最大的阈值
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_r2(left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    # 选择所有样本的均值作为叶子节点取值
    @staticmethod
    def calc_leaf_value(targets):
        return targets.mean()

    # 回归树采用平方误差来选择最优分裂点
    @staticmethod
    def calc_r2(left_targets, right_targets):
        r2 = 0
        for targets in [left_targets, right_targets]:
            mean = targets.mean()
            for dt in targets:
                r2 += (dt - mean) ** 2
        return r2

    # 根据特征和阈值将样本划分成左右两份，左边小于等于阈值，右边大于阈值
    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    # 输入样本，得到预测值
    def predict(self, dataset):
        res = []
        for index, row in dataset.iterrows():
            pred_list = []
            # 统计每棵树的预测结果，再求平均作为最终预测值
            for stage, tree in self.trees.items():
                pred_list.append(tree.calc_predict_value(row))
            res.append(sum(pred_list) * 1.0 / len(pred_list))
        return np.array(res)


if __name__ == '__main__':
    df = pd.read_csv("source/housing.txt").fillna(-1)
    df = df.rename(columns={'MEDV': 'label'})
    clf = RandomForestRegression(n_estimators=5,
                                 max_depth=-1,
                                 min_samples_split=20,
                                 min_samples_leaf=10,
                                 colsample_bytree="sqrt",
                                 subsample=0.6,
                                 random_state=66)
    train_count = int(0.7 * len(df))
    clf.fit(df.ix[:train_count, :-1], df.ix[:train_count, 'label'])

    from sklearn import metrics
    print(metrics.mean_squared_error(df.ix[:train_count, 'label'], clf.predict(df.ix[:train_count, :-1])))
    print(metrics.mean_squared_error(df.ix[train_count:, 'label'], clf.predict(df.ix[train_count:, :-1])))
