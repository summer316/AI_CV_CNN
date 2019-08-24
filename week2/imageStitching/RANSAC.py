# -*- coding: utf-8 -*-
import numpy as np
import random
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
def random_partition(n, m):
    return n, m
## Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.
def ransac_v1(data,model,n,k,t,d,debug=False,return_all=False):
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

    iterations = 0
    bestfit = None
    besterr = np.inf
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
            betterdata = np.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = np.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) )
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

def dlt(f, t, num_points=4):
	"""	
    Returns Homography matrix in which 'f' points are mapped to 't' points
	using Direct Linear Transform algorithm.
	t = Hf
	"""
	assert f.shape == t.shape
	num_points = f.shape[0]
	A = np.zeros((2*num_points, 9))
	for p in range(num_points):
		fh = np.array([f[p,0], f[p,1], 1])		# Homogenous coordinate of point p
		A[2*p] = np.concatenate(([0, 0, 0], -fh, t[p,1]*fh))		# [0' -wX' yX']
		A[2*p + 1] = np.concatenate((fh, [0, 0, 0], -t[p,0]*fh))	# [wX' 0' -xX']
	U, D, V = np.linalg.svd(A)
	H = V[8].reshape(3, 3)
	return H / H[-1,-1]

"""
processes of ransac:
    1随机选取一组匹配点
    计算仿射变换矩阵
    根据给定的阈值计算在正常范围内的匹配对的数目
    不断重复，保留最大正常匹配对的数目
    对保留的最大整整匹配对进行最小二乘法的估计，得到图2到图1的仿射变换矩阵。
"""
def ransac_v2(ptsA, ptsB):
    assert len(ptsA) == len(ptsB)
    status = np.zeros((len(ptsA)), dtype=np.int32)
    threshold_inliers = 0
    threshold_distance = 0.5

    iterations = 0  
    while iterations < 1000:
        indexes = random.sample(range(len(ptsA)), 4)
        fp = np.array([ptsA[pt] for pt in indexes])
        tp = np.array([ptsB[pt] for pt in indexes])
        homography = dlt(fp, tp)        # tp = H*fp
        src_pts = np.insert(ptsA, 2, 1, axis=1).T # Add column of 1 at the end (Homogenous coordinates)
        dst_pts = np.insert(ptsB, 2, 1, axis=1).T # Add column of 1 at the end (Homogenous coordinates)
        projected_pts = np.dot(homography, src_pts)
        error = np.sqrt(np.sum(np.square(dst_pts - (projected_pts/projected_pts[-1])), axis=0))
        
        # 如果内点数增多，就认为找到一个比较好的模型，进行更新
        if np.count_nonzero(error < threshold_distance) > threshold_inliers:
            # src_inliers = src_pts[:, np.argwhere(error < threshold_distance).flatten()][:-1].T
            # dst_inliers = dst_pts[:, np.argwhere(error < threshold_distance).flatten()][:-1].T
            best_H = homography
            threshold_inliers = np.count_nonzero(error < threshold_distance)
        
        iterations += 1

    if best_H is not None:
        status = [1 for i in np.argwhere(error < threshold_distance)]
        return (best_H, status)
    else:
        raise ValueError("homography is None")
        return (None, None)