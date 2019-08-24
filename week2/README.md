
##### medianBlur/medianBlur.py: 中值滤波，
    中值滤波有很多种实现方式，也有关于提高滤波的效率的研究。
    为此，中值滤波的方法有： SM(standart Median filter), MFM(mean_based fast median filters), 
    CTMF(Median Filtering in Constant Time)等  思路有 提高获取中值的效率，提高更新中值的效率。

    CTMF是基于直方图的中值滤波，[http://files.cnblogs.com/Imageshop/AFastTwo-DimensionalMedianFilteringAlgorithm.pdf](CTMF 论文)
    此次实现的中值滤波是参考CTMF。
    支持不同大小的kernel和两种padding方式，用快速排序查找中值。
    
    '''
    pseudo of CTMF
    
    Input: image X of size m*n, kernel radius r.
    
    output: image Y as X.
    for i = r to m - r do  #遍历每一行，从第一个卷积中心开始，到最后一个卷积中心结束
    
      每进入一个新行时 初始化一个直方图;  n = 0，用来计数比中值小的数的多少
      求得第一次卷积时的中值，用快速排序算法
      更新直方图
      
      for j = r to n - r do  遍历每一列，从第一个卷积中心开始，到最后一个卷积中心结束 
        如果是从左到右的第一个卷积，跳过

        #更新直方图的值 
        for a = i-r to i+r  遍历卷积对应的位置的每一行
          for b = j-r to j+r  遍历卷积对应的位置的每一旬
            卷积核左边的负一列的值对应的 直方图及 n 值都 减 1，
            卷积核最右边的值对应 直方图 和 n 都加 1 
          end 
        end
        
        从得到的直方图里的值获取 中值
        用中值对应此时的卷积核中心
        
      end
      
    end
    '''
##### imageStitching/
###### image_stitching.py
   * Pipeline of image stitching
   * 1: find features points in each image by ransac 
   * 2: use KNN to find keypoint matches
   * 3: use RANSAC to find homography matrix to get transferring info
   * 4: merge two image
 
###### RANSAC.py
    * ransac_v1: pseudo code of ransac
    * ransac_v2: find homography matrix for two images stitching
    
###### harrisCorner.py
    * process of harris corner detection algorithm
  
###### 
    * progress of HOG, undo
    
     
