# -*- coding: utf-8 -*-


'''
RANSAC 
Random Sample Consensus随机抽样一致
基本假设是：
（1）数据由“局内点”组成
（2）“局外点”是不能适应该模型的数据,比如噪声的极值；错误的测量方法；对数据的错误假设。

adv:能鲁棒的估计模型参数
disadv:只有一定的概率得到可信的模型，概率与迭代次数成正比
            要求设置跟问题相关的阀值,迭代次数无上限
应用：图像拼接 ...
#Pseudo

Given:
    data – a set of observed data points
    model – a model that can be fitted to data points
    n – the minimum number of data values required to fit the model
    k – the maximum number of iterations allowed in the algorithm
    t – a threshold value for determining when a data point fits a model
    d – the number of close data values required to assert that a model fits well to data

Return:
    bestfit – model parameters which best fit the data (or nul if no good model is found)

iterations = 0
bestfit = nul
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
        thiserr = a measure of how well model fits these points

        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
'''
## Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

def ransac(data,model,n,k,t,d,debug=False,return_all=False):
    """fit model parameters to data using the RANSAC algorithm

This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

Given:
    data - a set of observed data points # 可观测数据点集
    model - a model that can be fitted to data points #
    n - the minimum number of data values required to fit the model # 拟合模型所需的最小数据点数目
    k - the maximum number of iterations allowed in the algorithm # 最大允许迭代次数
    t - a threshold value for determining when a data point fits a model #确认某一数据点是否符合模型的阈值
    d - the number of close data values required to assert that a model fits well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good model is found)
"""
    import numpy
    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error( test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]

        if len(alsoinliers) > d:
            betterdata = numpy.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = numpy.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        print("did not meet fit acceptance criteria")
        return 
    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}
    else:
        return bestfit


'''
1:  K 的设定
    w = 局内点数量与总数据集的比，估计值
    n :  用n个独立点选择来估计模型
    w^n： n 个点都是局内点的概率
    1 - w^n： 至少有一个点是局外点的概率
     故： k = log(1-p)/log(1-w^n)
     (因为  1-p 即（1-w^n）^k  表示选择的点每次都有局内点)
'''
