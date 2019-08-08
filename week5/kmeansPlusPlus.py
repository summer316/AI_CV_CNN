#  -*- encoding:utf-8 -*-

import numpy as np
from  sklearn.datasets import make_moons
import matplotlib.pyplot  as plt

"""
process of kmeans
1: init K centers 
2: Repeat untill no longer update of centers:
    1: compute distance between every sample and K centers
    2: decide which cluster this point is belong to
    3: computer new centers, update centers
---------------------------------------
如何用改进的方法得到初始K个聚类中心。
1：假定数据点集合X有n个数据点，依次用X(1)、X(2)、……、X(n)表示，
依次计算每个数据点与最近的种子点（聚类中心）的距离，依次得到D(1)、D(2)、……、D(n)构成的集合D。
在D中，为了避免噪声，不能直接选取值最大的元素，应该选择值较大的元素，然后将其对应的数据点作为种子点。
2：把集合D中的每个元素D(x)想象为一根线L(x)，线的长度就是元素的值。
将这些线依次按照L(1)、L(2)、……、L(n)的顺序连接起来，组成长线L。L(1)、L(2)、……、L(n)称为L的子线
3：根据概率的相关知识，如果我们在L上随机选择一个点，那么这个点所在的子线很有可能是比较长的子线，
而这个子线对应的数据点就可以作为种子点。
"""
class kmeansPlusPlus():
    """
    Paras: 
    """
    
    def __init__(self, n_samples, K):
        """Constructor"""
        self.K = K
        self.X, self.y = self.genData(n_samples)
        self.initD = 1e100
        self.threshold = 1e-8
        
        
    def genData(self,  n_samples):
        """"""
        X, y = make_moons(n_samples=n_samples, noise=0.2)
        # fig = plt.figure( figsize=(8, 6))
        # plt.subplot(111)
        # plt.title("make_moons datasets")
        # plt.scatter(X[:, 0],  X[:, 1], marker="o", c=y)
        # plt.show()
        
        return X, y
    
    def nearest(self, sample, cents):
        if len(cents) == 1:
            dist = np.sum((sample - cents)**2)
            return dist
        else:
            dist = np.sum((sample - cents)**2, axis=1)
            idx = np.argwhere(dist == np.min(dist))
            #??:the shape of dist[idx] is [[1.11..]], why not is [1.11...]
            return dist[idx].reshape(-1)

    def initCenters(self, plus):
        """"""
        #whether to use advanced methor to generate centers
        centers = [ ]
        if plus:
            # step 1: random choise a point as center from samples
            idx = np.random.randint(0, len(self.X))
            centers.append(self.X[idx, :])
            for i in range(1, self.K):
                distances = []
                for j in self.X:
                    #compoute the nearest distance between every sample and existed centers
                    nearestCenter = self.nearest(j, centers)
                    #compute the distance between this sample and existed centers, D(X**2)
                    distances.append(nearestCenter)

                # http://blog.sciencenet.cn/blog-324394-292355.html (蒙特卡罗)
                # multiply a random number for simulate Monte Carlo method(蒙特卡罗)
                temp = np.sum(distances) * np.random.rand()
                for j in range(len(self.X)):
                    temp -= distances[j]

                    #The farther the distance, the higher the probability 
                    if temp < 0: 
                        centers = np.append(centers, [self.X[j]], axis=0)
                        break  #only exited from current for loop 

        else:
            #random choise K centers from samples
            idx = [np.random.randint(len(self.X)) for i in range(self.K)]
            centers = [self.X[val, :] for idx,val in enumerate(idx)]

        return centers
    

    def assignment(self, centers):
        idxClusters = [[ ] for i in range(self.K)]
        for idx, val in enumerate(self.X):
            distance = [np.sqrt(np.sum((i - val)**2)) for i in centers]
            #idx present the cluster this point belong to
            idxCluster = distance.index(np.min(distance))
            idxClusters[idxCluster].append(idx)
        
        #return cluster result with idx
        return idxClusters 

    def update(self, idxClusters, centers):
        for idx, val in enumerate(idxClusters):    
            centers[idx] = np.mean([self.X[i, :] for i in val], axis=0)
        
        return np.array(centers,dtype=np.float32)


    def  kmeans(self):
        """ """
        import copy
        #inintiate K centers
        #initCenters true: user kmeans++, false: kmeans
        centers = self.initCenters(True)
        
        #get finally centers
        oldCenters = np.array([[0, 0] for i in range(self.K)])
        i = 0
        while np.sum((centers-oldCenters)**2) > self.threshold:
            oldCenters = copy.deepcopy(centers)
            idxClusters = self.assignment(centers)     
            #update centers
            centers = self.update(idxClusters, centers)
            if i % 5 == 0:
                fig = plt.figure( figsize=(8, 6))
                plt.subplot(111)
                plt.title("make_moons datasets")
                plt.scatter(self.X[:, 0],  self.X[:, 1], marker="o", c=self.y, linewidths=0.5)
                plt.scatter(centers[:,0], centers[:, 1], marker="v",color="r", linewidths=3)
                plt.draw()
                plt.pause(1)
                plt.close()

            i += 1
            
        print("After {:} steps, kmeans++ cluster centers is {:}".format(i, centers))

        
        
         
if __name__ == "__main__":
    n_samples = 1000 #numbers of samples
    k = 2 #numbers of  kmeans centers
    model = kmeansPlusPlus(n_samples=n_samples, K=k)
    model.kmeans()
    
