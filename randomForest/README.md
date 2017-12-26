# 随机森林之JAVA实现

 最近在学习randomForest算法和JAVA，于是乎把两者结合在一起作为练手项目。其中randomForest的理论部分主要来自周志华老师的《机器学习》，本博文主要包括以下几个部分。
- **Bagging与RandomForest的主要区别**
- **RandomForest Algorithm**
- **Main Feature**
- **About DATA**
-------------------


## Bagging与RandomForest的主要区别


Bagging作为集成学习方法的一种，其主要的特征在于在对原始数据采样中加入了**数据扰动**的部分，具体来说，主要是基于自助采样法 (bootstrap sampling)，给定包含 m 个样本的数据集，我们先随机取出一个样本放入采样集中，再把该样本放回原始数据集，这样，经过 m次随机采样操作，我们得到含 m 个样本的采样集，根据概率统计知识可知，原始数据集中约有 63.2%的样本出现在采样集中，其余36.8%的数据可用于袋外估计(out-of-bag estimate)。

假定基学习器的计算复杂度为 O(m) ，O(s)是投票/平均过程中的计算复杂度，T是基学习器数量， 则 Bagging 的计算复杂度大致为T (O(m) + O(s)) ，由于 T通常是一个不太大的常数，O(s)常常可忽略，因此，训练一个 Bagging 集成与直接使用基学习算法同阶，这说明 Bagging 是一个很高效的集成学习算法。

而randomForest算法与Bagging算法的主要区别在于，randomForest在Bagging的基础上增加了**属性扰动**部分，具体来说，以决策树作为基学习器为例，传统决策树在选择划分属性时，是在当前结点的属性集合(假定有 d 个属性)中选择一个最优属性，而在RF 中，对决策树的每个结点，先从该结点的属性集合中随机选择一个包含 k 个属性的子集（一般来说，k=log2(d)），然后再从这k个属性中选择一个最优属性用于划分。
一般来说，RF在训练过程中加入了**数据扰动**和**属性扰动**，所以不需要剪枝就能达到很好的泛化效果。

## RandomForest Algorithm

在采用以决策树作为基学习器的RF算法中，决策树的建立与递归划分的过程主要采用西瓜书中的内容，如下所示：

![决策树](http://img.blog.csdn.net/20171225214106036)

RF算法的训练过程如下所示：

![RF](http://img.blog.csdn.net/20171225214337677)

## Main Feature

1. 对程序的测试的数据采用了公开数据集[**NSL-KDD**](http://www.unb.ca/cic/datasets/nsl.html)

2. 属性数据采用的是连续属性（Continuous Attributes），在划分决策树节点，分成两个分叉

3. 在代码的后面，增加了计算[**Variable Importance**](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr)的内容，理论知识可参考伯克利关于RF的Introduction


## About DATA

Data.txt是训练数据集， Text.txt是测试数据集，经线下测试，在使用上述train data建立100棵树的RF model，Test集的forest-wide Error可达到90%，同时发现Attr16和Attr17的Variable Importance较高

