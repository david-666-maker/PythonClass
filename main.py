from math import log, sqrt
import operator
import re
import matplotlib.pyplot as plt
import pickle  # 利用pickle模块存储决策树


class C45Tree_make():

    def __init__(self):
        print('C45类')

    def createDataSet(self):
        """
        导入数据集
        """
        fr = open(r'PlayData.txt')
        Alldata = [inst.strip().split('\t') for inst in fr.readlines()]
        dataSet = Alldata[1: -1]
        labels = Alldata[0]  # 两个特征
        return dataSet, labels

    def classCount(self, dataSet):
        """
        类别统计函数
        :param dataSet 数据集
        :return  labelCount: 不同的类别数
        """
        label_Cnt = {}
        for one in dataSet:
            if one[-1] not in label_Cnt.keys():
                label_Cnt[one[-1]] = 0
            label_Cnt[one[-1]] += 1
        return label_Cnt

    def getShannonEntropy(self, dataSet):
        """
        通过公式计算熵值
        :param dataSet: 训练数据集
        :return: Entropy: 熵
        """
        label_Cnt = self.classCount(dataSet)
        Entries_num = len(dataSet)
        Entropy = 0.0
        Entropy_cal = lambda x: x * log(x, 2)   # 熵值计算
        for i in label_Cnt:
            probility = float(label_Cnt[i]) / Entries_num
            Entropy -= Entropy_cal(probility)
        return Entropy

    def getClass_most(self, dataSet):
        """
        返回出现次数最多的类别（通过排序）
        :param   dataSet 训练数据集
        :return: 出现次数最多的类别
        """
        label_Cnt = self.classCount(dataSet)
        sortedLabelCnt = sorted(label_Cnt.items(), key=operator.itemgetter(1), reverse=True)
        return sortedLabelCnt[0][0]

    # 划分数据集
    def splitDataSet(self, dataSet, i, value):
        """
        划分数据集
        :param   dataSet: 数据集
        :param   i： 开始进行划分的下标
        :param   value: 划分特征的值
        :return  sub_DataSet: 划分出的数据集
        """
        sub_DataSet = []
        for one in dataSet:
            if one[i] == value:
                reduce_Data = one[:i]
                reduce_Data.extend(one[i + 1:])
                sub_DataSet.append(reduce_Data)
        return sub_DataSet

    def split_Continuous(self, dataSet, i, value, direction):
        """
        对连续数据集进行划分
        :param dataSet: 数据集
        :param i: 将要划分的特征序号
        :param value: 划分特征的值
        :param direction: 划分取向
        :return  sub_DataSet: 该情况下划分出的的数据集dataSet的子集
        """
        sub_DataSet = []
        # 对每条数据遍历
        for one in dataSet:
            # 选择大的进行划分
            if direction == 0:
                if one[i] > value:
                    reduce_Data = one[:i]
                    reduce_Data.extend(one[i + 1:])
                    sub_DataSet.append(reduce_Data)
            # 选择小的进行划分
            if direction == 1:
                if one[i] <= value:
                    reduce_Data = one[:i]
                    reduce_Data.extend(one[i + 1:])
                    sub_DataSet.append(reduce_Data)

        return sub_DataSet

    def chooseFeature_best(self, dataSet, labels):
        """
        选择最佳数据集划分方式
        :param dataSet: 数据集
        :param labels: 所有属性
        :return  最佳划分属性的索引和连续属性的最佳划分值
        """
        # 计算整个数据集的熵并初始化
        base_Entropy = self.getShannonEntropy(dataSet)
        Feature_best = 0
        gainRate_base = -1
        Features_num = len(dataSet[0]) - 1  # 特征数
        SplitDic_best = {}
        InInfoGain_cal = lambda x: x * log(x, 2)  # 熵值计算
        i = 0

        # 对每个特征循环
        for i in range(Features_num):

            featureList = [item[i] for item in dataSet]  # 特征向量
            # 对于连续属性
            if type(featureList[0]).__name__ == 'float' or type(featureList[0]).__name__ == 'int':
                j = 0
                sortedfeatureList = sorted(featureList)
                splitList = []
                # 对每个特征循环，逐个计算每两个之间的平均值
                for j in range(len(featureList) - 1):
                    splitList.append((sortedfeatureList[j] + sortedfeatureList[j + 1]) / 2.0)
                # 对每一个潜在的划分点计算其增益率，得到最佳划分
                for j in range(len(splitList)):
                    Entropy_new = 0.0  # 新的熵
                    gainRate = 0.0  # 增益率
                    split_InfoGain = 0.0  # 信息增益
                    value = splitList[j]  # 划分点的值
                    # 划分数据集
                    sub_DataSet_0 = self.split_Continuous(dataSet, i, value, 0)
                    sub_DataSet_1 = self.split_Continuous(dataSet, i, value, 1)
                    # 计算新划分下的熵
                    probility0 = float(len(sub_DataSet_0)) / len(dataSet)
                    Entropy_new -= probility0 * self.getShannonEntropy(sub_DataSet_0)
                    probility1 = float(len(sub_DataSet_1)) / len(dataSet)
                    Entropy_new -= probility1 * self.getShannonEntropy(sub_DataSet_1)
                    # 使用lambda计算增益率
                    split_InfoGain -= InInfoGain_cal(probility0)
                    split_InfoGain -= InInfoGain_cal(probility1)
                    gainRate = float(base_Entropy - Entropy_new) / split_InfoGain
                    print('IVa ' + str(j) + ':' + str(split_InfoGain))
                    # 根据增益率判断划分
                    if gainRate > gainRate_base:
                        gainRate_base = gainRate
                        Split_best = j
                        Feature_best = i
                SplitDic_best[labels[i]] = splitList[Split_best]

            # 对于离散属性
            else:
                uniquefeatureList = set(featureList)  # 提取不重复的属性
                gainRate = 0.0  # 新的熵
                split_InfoGain = 0.0  # 增益率
                Entropy_new = 0.0  # 信息增益
                # 对属性遍历并计算新的熵
                for value in uniquefeatureList:
                    sub_DataSet = self.splitDataSet(dataSet, i, value)
                    prob = float(len(sub_DataSet)) / len(dataSet)
                    split_InfoGain -= InInfoGain_cal(prob)
                    Entropy_new -= prob * self.getShannonEntropy(sub_DataSet)
                # 计算新信息增益
                gainRate = float(base_Entropy - Entropy_new) / split_InfoGain
                if gainRate > gainRate_base:
                    Feature_best = i
                    gainRate_base = gainRate

        # 对于连续属性，得到最佳划分值
        if type(dataSet[0][Feature_best]).__name__ == 'float' or type(dataSet[0][Feature_best]).__name__ == 'int':
            Feature_bestValue = SplitDic_best[labels[Feature_best]]
        # 对于离散属性，得到最佳划分值
        if type(dataSet[0][Feature_best]).__name__ == 'str':
            Feature_bestValue = labels[Feature_best]

        return Feature_best, Feature_bestValue

    # 创建决策树
    def C45Tree(self, dataSet, labels):
        """
        创建决策树
        :param dataSet: 数据集
        :param labels: 属性集
        :return Tree: 决策树
        """
        # 找出类别向量
        classList = [item[-1] for item in dataSet]
        # 如果样本全部属于同一类
        if len(set(classList)) == 1:
            return classList[0][0]
        # 如果样本在该属性上的取值都相同
        if len(dataSet[0]) == 1:
            return self.getClass_most(dataSet)
        Entropy = self.getShannonEntropy(dataSet)
        # 获取最优划分属性
        Feature_best, Feature_bestLabel = self.chooseFeature_best(dataSet, labels)
        # 将最优的划分属性进行记录
        Tree = {labels[Feature_best]: {}}
        sub_Labels = labels[:Feature_best]
        sub_Labels.extend(labels[Feature_best + 1:])
        # 对于离散型变量
        if type(dataSet[0][Feature_best]).__name__ == 'str':

            # 找出该特征包含的所有值
            featureList = [item[Feature_best] for item in dataSet]
            unique_featureList = set(featureList)
            # 对这些特征递归调用该函数
            for value in unique_featureList:
                reduceDataSet = self.splitDataSet(dataSet, Feature_best, value)
                Tree[labels[Feature_best]][value] = self.C45Tree(reduceDataSet, sub_Labels)

        # 对于连续型变量
        if type(dataSet[0][Feature_best]).__name__ == 'int' or type(dataSet[0][Feature_best]).__name__ == 'float':
            value = Feature_bestLabel
            # 将数据进行划分
            DataSet_great = self.split_Continuous(dataSet, Feature_best, value, 0)
            DataSet_small = self.split_Continuous(dataSet, Feature_best, value, 1)
            # 对这些特征递归调用该函数，根据增益率选择最高的进行划分
            Tree[labels[Feature_best]]['>' + str(value)] = self.C45Tree(DataSet_great, sub_Labels)
            Tree[labels[Feature_best]]['<=' + str(value)] = self.C45Tree(DataSet_small, sub_Labels)

        return Tree

    # 以下函数为绘制决策树所用
    def plotNode(self, node_text, centerPt, parent_position, nodeType):
        """
        绘制决策树节点
        :param node_text: 该节点的标签
        :param centerPt: 文本位置
        :param nodeType: 节点类型
        """
        newTxt = node_text
        if type(node_text).__name__ == 'tuple':
            newTxt = node_text[0] + '\n'
            for strI in node_text[1:-1]:
                newTxt += str(strI) + ','
            newTxt += str(node_text[-1])
        # 设置决策节点和叶节点以及箭头的形状
        arrow_args = dict(arrowstyle="<-", connectionstyle="arc3", shrinkA=0,
                          shrinkB=16)
        self.createPlot.ax1.annotate(newTxt, xy=parent_position,
                                     xycoords='axes fraction',
                                     xytext=centerPt, textcoords='axes fraction',
                                     va="top", ha="center", bbox=nodeType,
                                     arrowprops=arrow_args)

    def getLeafs_num(self, Tree):
        """
        获取叶子节点数目
        :param Tree: 决策树
        :return Leafs_num: 叶子节点数目
        """
        Leafs_num = 0
        # 第一个节点
        first_node_Str = list(Tree.keys())[0]
        # 第一个节点所对应的内容
        secondDict = Tree[first_node_Str]

        # 遍历子节点
        for key in secondDict.keys():
            # 如果有子节点，则递归调用
            if type(secondDict[key]).__name__ == 'dict':
                Leafs_num += self.getLeafs_num(secondDict[key])
            # 没有子节点，为叶子节点
            else:
                Leafs_num += 1
        return Leafs_num

    def getTree_Depth(self, Tree):
        """
        获取树的深度
        :param Tree: 决策树
        :return maxDepth: 最大深度
        """
        maxDepth = 0  # 初始化最大层数
        first_node_Str = list(Tree.keys())[0]  # 使用List转为列表进行操作
        secondDict = Tree[first_node_Str]  # 得到下表对应的节点

        # 遍历整个数并判断节点，直到判断为叶子节点结束，并进行递归调用
        for key in secondDict.keys():
            # 如果有子节点，则递归调用
            if type(secondDict[key]).__name__ == 'dict':
                thisDepth = 1 + self.getTree_Depth(secondDict[key])
            # 没有子节点，为叶子节点
            else:
                thisDepth = 1
            # 获得最大层数
            if thisDepth > maxDepth:
                maxDepth = thisDepth

        return maxDepth

    def createTree_retrieve(self, i):
        """
        创建一个初始化树
        """
        listOfTrees = [
            {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
            {'no surfacing': {0: 'no', 1: {
                'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
        ]
        return listOfTrees[i]

    def plotText_Mid(self, child_position, parent_position, textString):
        """
        填写父子节点中的标签信息
        :param child_position: 子节点位置
        :param parent_position: 父节点位置
        :param textString: 填充的内容
        """
        # 计算x方向中点
        x_position_Mid = (parent_position[0] - child_position[0]) / 2.0 + child_position[0]
        # 计算y方向中点
        y_position_Mid = (parent_position[1] - child_position[1]) / 2.0 + child_position[1]
        # 绘制
        self.createPlot.ax1.text(x_position_Mid, y_position_Mid, textString)

    def plotTree(self, Tree, parent_position, node_text):
        """
        计算决策树宽高
        :param Tree: 决策树
        :param parent_position: 父节点的位置
        :param node_text: 节点的内容
        """
        # 设置决策节点和叶节点的边框形状、边距和透明度
        decisionNode = dict(boxstyle="square,pad=0.5", fc="0.9")
        # 叶子节点数目
        leafNode = dict(boxstyle="round4, pad=0.5", fc="0.9")
        Leafs_num = self.getLeafs_num(Tree)
        # 树的深度
        depth = self.getTree_Depth(Tree)
        # 获取根节点内容
        first_node_Str = list(Tree.keys())[0]

        # 计算当前的根节点在其子节点中的中间位置
        child_position = (C45Tree_make.plotTree.xOff + (1 + float(Leafs_num)) / 2.0 / C45Tree_make.plotTree.totalW,
                          C45Tree_make.plotTree.yOff)
        # 绘制该节点和其父节点的联系
        C45Tree_make.plotText_Mid(self, child_position, parent_position, node_text)
        # 绘制该节点
        C45Tree_make.plotNode(self, first_node_Str, child_position, parent_position, decisionNode)
        # 获得当前节点的子节点构成的树
        secondDict = Tree[first_node_Str]
        # 计算y方向偏移量
        C45Tree_make.plotTree.yOff = C45Tree_make.plotTree.yOff - 1.0 / C45Tree_make.plotTree.totalD

        # 遍历所有的节点索引
        for key in secondDict.keys():
            # 如果非叶子节点则递归调用
            if type(secondDict[key]).__name__ == 'dict':
                C45Tree_make.plotTree(self, secondDict[key], child_position, str(key))
            else:
                # 计算x方向偏移量
                C45Tree_make.plotTree.xOff = C45Tree_make.plotTree.xOff + 1.0 / C45Tree_make.plotTree.totalW
                # 绘制叶子节点
                C45Tree_make.plotNode(self, secondDict[key], (C45Tree_make.plotTree.xOff, C45Tree_make.plotTree.yOff),
                                      child_position, leafNode)
                # 绘制叶子节点和父节点的联系
                C45Tree_make.plotText_Mid(self, (C45Tree_make.plotTree.xOff, C45Tree_make.plotTree.yOff),
                                          child_position, str(key))

        # 返回绘制上一层的y方向内容
        C45Tree_make.plotTree.yOff = C45Tree_make.plotTree.yOff + 1.0 / C45Tree_make.plotTree.totalD

    def createPlot(self, inTree):
        """
        绘制整个决策树
        :param inTree: 决策树
        """
        # 创建一个图像
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        C45Tree_make.createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

        # 计算要绘制的树的宽度与深度
        C45Tree_make.plotTree.totalW = float(self.getLeafs_num(inTree))
        C45Tree_make.plotTree.totalD = float(self.getTree_Depth(inTree)) + 0.5

        # 初始化x与y方向偏移量
        C45Tree_make.plotTree.xOff = -0.5 / C45Tree_make.plotTree.totalW
        C45Tree_make.plotTree.yOff = 1.0

        # 画图并显示
        C45Tree_make.plotTree(self, inTree, (0.5, 1.0), '')
        plt.show()



if __name__ == '__main__':
    tree1 = C45Tree_make()
    dataSet, labels = tree1.createDataSet()  # 获取数据集
    tree = tree1.C45Tree(dataSet, labels)  # 建立树
    print('打印决策树：')  # 打印显示结果
    print(tree)
    tree1.createPlot(tree)  # 画出决策树的树图



