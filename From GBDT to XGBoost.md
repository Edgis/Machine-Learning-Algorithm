## From GBDT to XGBoost
  - GBDT Algorithm
  - XGBoost 
  - GBDT和XGBoost的异同

### **1. GBDT Algorithm**
GBDT(Gradient Boosting Decision Tree )是boosting Tree的一种，是以分类与回归树(CART)为基学习器，采用加法模型(Additive function Model)和前向分布算法(Forward stagewise algorithm),是Freidman在1999年提出来的，其关键是利用损失函数的负梯度在当前模型的值来拟合基学习器模型。
算法的过程参考wepon的slides：
![这里写图片描述](http://img.blog.csdn.net/20180208144027403?)
下面，我将详细解释算法的过程：
**Input**:输入训练数据集 $(x_{i},y_{i})$ ，迭代次数*T*， 以及损失函数*L* 
**1.** 这里的初始化过程，参考了Freidman原文，目的是找到一个常数，使得初始经验损失最低。
![这里写图片描述](http://img.blog.csdn.net/20180208144454767?)
**2.** 对于每一次迭代*t*
**2.1.** 计算损失函数在当前点的负梯度，*N* 是训练集的数目
**2.2** 计算第*t* 棵树的参数，这里 $h_{t}$ 表示第*t* 棵回归树，$\omega$  表示第*t* 棵回归树的参数，也就是回归树的切分变量(splitting variables)，切分点(splitting point)和叶子节点的值。至于为什么是平方和的形式，我参考了一篇 [paper](https://www.frontiersin.org/articles/10.3389/fnbot.2013.00021/full)  
> Instead of looking for the general solution for the boost increment in the function space, one can simply choose the new function increment to be the most correlated with −gt(x). This permits the replacement of a potentially very hard optimization task with the classic least-squares minimization one.                    —《Gradient boosting machines, a tutorial》


我是这样理解的，求解负梯度的通用具体形式太过复杂，可以近似建立一棵回归树来近似表示负梯度的值，如何近似呢，最简单的就是平方和最小，所以优化问题就转变成了最小二乘优化问题。

>Freidman原文：This permits the replacement of the  **difficult function minimization** problem  by **least-squares function minimization**(11), followed by **only a single parameter optimization** based on the original criterion (12)                      
>      ![这里写图片描述](http://img.blog.csdn.net/20180208191758903?)
>      ![这里写图片描述](http://img.blog.csdn.net/20180208191823717?)           
>    

                                                                                      

**2.3** 线性遍历求步长 $\rho$
**2.4** 更新模型，对于回归树，其本身就是一个加法模型(Additive function Model)，
> ![15](http://img.blog.csdn.net/201802082025384?) 
>  式中 $J$ 是叶子节点个数，$\{R_{j}\}_{1}^{J}$ 是切分变量，  $\{b_{j}\}_{1}^{J}$ 是切分点     

该过程可描述如下：
> ![16](http://img.blog.csdn.net/20180208202236820?)


至于 $\beta_{m}$ 的求解，在Freidman的paper中，在基学习器是回归树，损失函数(loss function)是LDA(绝对值)的情况下， 上式中的 $\alpha_{m}$ 与  $\beta_{m}$ 是一起求解的。 上式(16)可以写成以下形式：
![这里写图片描述](http://img.blog.csdn.net/20180208215030653?)
其中  $\gamma_{j m}$ = $\alpha_{m}$ * $\beta_{m}$ , 因此问题就变成了求解  $\gamma_{j m}$ ，限制(constriction)为
![这里写图片描述](http://img.blog.csdn.net/20180208215635342?)
由于 $R_{jm}$ 在一棵回归树上上是互斥的，故上式可表示为：
![这里写图片描述](http://img.blog.csdn.net/20180208215852189?)

$\gamma_{j m}$ = $argmin_{\gamma}$  $\sum_{x_{i}->R_{j m}}$ $|y_{i} - F_{m-1}(x_{i}) - \gamma |$

由于该条件(我也不太懂)
![这里写图片描述](http://img.blog.csdn.net/20180208221929984?)
因此最终结果为：
![这里写图片描述](http://img.blog.csdn.net/20180208220910891?)
其中，median即中值。
综上所述， $\gamma_{j m}$ 为在第*m*次迭代过程中的残差(当前值—上一模型的预测值)得中值，故LDA下的 GBDT 算法如下：
![这里写图片描述](http://img.blog.csdn.net/20180208222612475?)

**3.**  输出模型


***
### **2. XGBoost** 
#### **2.1  正则化(Regularization)**
XGBoost是GBDT的加强版，这部分主要参考了陈天奇同学的 [paper](https://dl.acm.org/citation.cfm?id=2939785) 和[slide](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf) 。同样使用加性模型(Addative model)和前向分布算法(Forward stagewise Algorithm)。个人感觉XGBoost与GBDT最大的区别在于目标函数的选择，XGBoost在经验损失函数的基础上增加了正则项，使得学习出来的模型更加不容易过拟合
![xgb_obj](http://img.blog.csdn.net/20180208224337282?)
![这里写图片描述](http://img.blog.csdn.net/20180208224720115?)
其中，正则化的部分包括叶子节点个数 *T*， 叶节点分数(leaf scores)  *w* 
#### **2.2  梯度提升(gradient boosting)**
上面的原始优化目标函数肯定是不好优化的，考虑到XGBoost采用的是加法模型，在 *t* 次迭代过程中，可以写成
![这里写图片描述](http://img.blog.csdn.net/20180209105651227?)
其中：![这里写图片描述](http://img.blog.csdn.net/20180209105828719?)
接下来，我们将 *t* 次迭代过程中的目标函数进行二阶泰勒展开
首先回忆一下二阶泰勒展开的公式：
![这里写图片描述](http://img.blog.csdn.net/20180209110530478?)
所以， *t* 次迭代过程中的目标函数的泰勒展开为
![这里写图片描述](http://img.blog.csdn.net/20180209112412794?)
其中：$g_{i}$ , $h_{i}$ 分别表示损失函数对当前预测模型的一次，二次导数
![这里写图片描述](http://img.blog.csdn.net/20180209112844731?)       ![这里写图片描述](http://img.blog.csdn.net/20180209112858346?)
对于中括号内的第一项 $\ell(y_{i} - \hat y ^{t-1})$ 是确定值，即为一常数，故去除第一项得：
![这里写图片描述](http://img.blog.csdn.net/20180209130841242?)
我们定义： $I_{j}$ = { $i$ |$q_{(x_{i})}$ = $j$ } 为在叶子节点 $j$ 上的样本集合，同时展开正则项，得到关于叶子节点 $T$ 的目标函数，其中 $w_{j}$ 是叶子节点 $j$ 的叶子得分(leaf scores)，我们知道对于一棵确定的回归树，则对于每一个输入，都会有确定的输出（叶子得分 leaf scores ），即 $f_{t}(x_{i})$ = $w_{j}$ 
![这里写图片描述](http://img.blog.csdn.net/20180209132524987?)
这是一个二次项的求和问题，所以对于一个确定的叶子节点 $j$ ，得到最佳叶子得分(leaf score) $w_{j}^*$
![这里写图片描述](http://img.blog.csdn.net/20180209132824437?)
将  $w_{j}^*$ 带入(4)，得到在回归树 $q_{(x)}$ 的 $j$ 叶子节点的最优目标函数
![这里写图片描述](http://img.blog.csdn.net/20180209133132610?)
其中，(6）式的物理意义类似于决策树中的熵值(Entropy)或者基尼指数(Gini Index)，是用来度量该树的结构，该值越小，树的结构就越好。问题进行到这里，如何确定树结构的切分变量(splitting variable)和切分点(splitting point)是下一步的工作的关键，舍弃了树结构遍历(enumerate)的方式，XGBoost采用了一中贪婪算法(Greedy Algorithm)，我们定义 $I$ 为该节点上所有样本的索引集，  $I_{L}$ ， $I_{R}$ 分裂后左右子节点的样本的索引集，即 $I$ = $I_{L}$$\cup$$I_{R}$ ，

$L_{I}$=-$1\over2$  $\frac{(\sum_{i\in I} g_{i})^2} {\sum_{i\in I} h_{i}  + \lambda}$ +$\gamma$ 
$L_{L}$=-$1\over2$  $\frac{(\sum_{i\in I_{L}} g_{i})^2} {\sum_{i\in I_{L}} h_{i}  + \lambda}$ +$\gamma$
$L_{R}$=-$1\over2$  $\frac{(\sum_{i\in I_{R}} g_{i})^2} {\sum_{i\in I_{R}} h_{i}  + \lambda}$ +$\gamma$

所以切分后损失减少值为(loss reduction)为
$GAIN$ = $L_{split}$ = $L_{I}$ - $L_{L}$ - $L_{R}$
![这里写图片描述](http://img.blog.csdn.net/20180209143433291?)
这个公式类似于ID3中的信息增益或者基尼增益，XGBoost 就是利用这个公式计算出的值作为分裂条件， 且该值越大越好。在每一个节点的分裂中寻找最优切分变量(splitting variable)和切分点(splitting point)。
##### **2.2.1 分裂点搜索算法(splitting finding Algorithm)**
如(7)所示，优化的下一步问题在于寻找最优切分变量(splitting variable)和切分点(splitting point)。陈天奇的文章中，介绍了两种搜索算法，一种是遍历搜索算法(原文称做 Exact Greedy Algorithm)和一种近似的估计算法(原文称做 Approximate Algorithm)。
##### **2.2.1.1 遍历搜索算法(Exact Greedy Algorithm)**

> ![这里写图片描述](http://img.blog.csdn.net/20180209185048247?)
> 这里，$k$ 代表第 $k$ 个特征(k-th feature) $j$ 代表的是第 $j$ 个数据的下表(j-th single data)，$x_{jk}$ 代表 $k$ 个特征的第 $j$ 个值 。


该算法首先会对第 $k$ 个特征的所有值进行排序，然后所有可能的分割点， 计算 $GAIN$ (或者 $L_{split}$ ) ， 选取值最大的(feature,  value)去分割。当然，此方法比较耗时，而且容易过拟合(over-fitting)。


##### **2.2.1.2 近似估计算法(Approximate Algorithm)**


与第一种方法的遍历不同，近似估计算法首先根据特征值的分布，确定 $l$ 个分位数 $S_{k}$ = $\{s_{k1}, s_{k2},...,s_{kl} \}$，利用这 $l$ 个分位数将特征 $k$ 的取值区间截断为 $l+1$ 个子区间。然后分别在每个子区中将该区间中求各个样本的一阶和二阶导数，并求和，得到该区间的一阶和二阶导数的统计量。后面在寻找最优分裂点时就只搜索这 $l$ 个分位数点，并从这 $l$ 各分位数点中找到 $GAIN$ 最大的值，作为该特征上最优分裂点。
对于分位数的使用，XGBoost有两种形式，一种是全局(global)形式，一种是局部(local)形式。全局(global)形式只在算法的初试阶段只计算一次，在后面的计算（树的各层构造中）仍然采用相同的分位数，全局形式(global)适合大部分的数的结构。局部形式(local)在每次分裂都会重新计算一次分位数，这种适合比较深的树的结构。
至于如何确定分位数，这是一个比较难的问题，在陈天奇的paper中，给了一个supplement链接。不像大家平常见到的数都是等权重的，这里不同特征值有不同的权重。至于为什么不同的特征值有不同的权重，我们可以把(2)式改写。
![这里写图片描述](http://img.blog.csdn.net/2018020921192553?)
我们发现，这是关于 $\frac{g_{i}}{h_{i}}$ 的平方损失函数，其权重就是 $h_{i}$ ，所以每一个特征值的权重为损失函数关于各个特征值的二阶导数。求不等权重的分位点的算法，陈天奇给了一个 [supplement](http://homes.cs.washington.edu/~tqchen/pdf/xgboost-supp.pdf)
由于这篇文章太过复杂，在本博客中就不解释了，打算过段时间再专门写一篇文章。
当求解分位点的这个最大的问题解决之后，我们才可以说**了解** XGBoost算法。
算法的具体步骤如下：
![这里写图片描述](http://img.blog.csdn.net/20180209212614132?)

#### **2.3  缩减(Shrinkage) 和列采样(Column Subsampling)**
缩减(Shrinkage) 就是在boosting的过程中，对于每一步生成的模型的前面乘以一个缩放因子 $\eta$ ，这就相当于，对于每一个生成的模型（回归树）的影响都乘以一个小于1的 $\eta$ ，也就是为了以后的优化留了一定的空间。
列采样(Column Subsampling)在随机森林用的比较多，主要是通过属性扰动，来避免过拟合。具体步骤是，对回归树的每个结点，先从该结点 $d$ 个属性中随机选择一个包含 $k$ 个属性的子集，然后再从这个子集中选择一个最优属性用于划分，一般来说， $k=log_{2}d$ 。
####  **2.4  对稀疏值的处理(Sparsity-aware Split Finding)**
稀疏值的处理基本思想如下：在非叶子节点进行分裂时，增加一个default分支（branch），是的对于 $k$ 特征中，缺失值被分到这个分支（branch）。同时为了进一步提高Accuracy，缺失值可以有两个分配方向，而最佳分配方向就从2个当中选取一个使得 $GAIN$ 最大的方向。更重要的是，通过这样的处理，我们使得处理稀疏值的计算量消耗(computation complexity)与稀疏值的数量成正比。
具体算法如下：
![这里写图片描述](http://img.blog.csdn.net/20180210141550379?)

####  **2.5  并行计算(Column Block for Parallel Learning)**
训练过程中最耗时的步骤是排序，XGBoost实现了一种Column Block技术，首先在建树之前，把每个特征（feature）的所有数值，按照特征大小顺序排序，并且计算出梯度信息，压缩（CSC）存储在内存中，这样的话，在以后的迭代中就可以使用索引来访问这些梯度信息。
![这里写图片描述](http://img.blog.csdn.net/20180210154417123?)

####  **2.6  Cache-aware Access**
column block按特征大小顺序存储，相应的样本的梯度信息是分散的，造成内存的不连续访问，或者梯度信息没有缓存到Cache中，降低了学习速度。
对于遍历算法(exact greedy algorithm)，我们为每一个Thread一个buffer中 ， 再统计梯度信息。
对于近似算法(approximate Algorithms)，我们通过手动选择合适的Block size来解决这个问题。Block size 我们定义为在包含在一个block中最大的样本数量。Block size太小，则影响并行计算，过大则会导致内存空间的浪费。作者是通过实验来确定block size的大小。

####  **2.7 Blocks for Out-of-core Computation**
利用Block来进行核外计算(Out-of-core Computation)的核心思想是计算器一边读取数据(由独立线程完成)，一边计算，为此，作者采用了两个策略：
1. 块压缩(Block Compression)
block是逐特征压缩(by feature)，并且解压工作也是由独立线程完成。实际操作中，我们使用 block beginning Index -row Index得到一个偏移量，这个偏移量存储在一个16 bits Integer ，也就是一个block存储256条数据。
2. 块分片(Block Sharding)
块分片就是把数据分割到多个磁盘中，然后每一个磁盘分配一个读取线程把数据读取到buffer，最后训练数据的线程再逐个从buffer读取数据。



