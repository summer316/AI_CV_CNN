##### RANSAC.py
    pseudo
 
##### week2_median_blur.py: 中值滤波，
    中值滤波有很多种实现方式，也有关于提高滤波的效率的研究。
    为此，中值滤波的方法有： SM(standart Median filter), MFM(mean_based fast median filters), 
    CTMF(Median Filtering in Constant Time)等  思路有 提高获取中值的效率，提高更新中值的效率。

    CTMF是基于直方图的中值滤波，论文
    [http://files.cnblogs.com/Imageshop/AFastTwo-DimensionalMedianFilteringAlgorithm.pdf]
    (http://files.cnblogs.com/Imageshop/AFastTwo-DimensionalMedianFilteringAlgorithm.pdf)
    此次实现的中值滤波是参考CTMF。
    支持不同大小的kernel和两种padding方式，用快速排序查找中值。
