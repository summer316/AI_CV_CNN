# _*_ coding:utf-8 _*_
import cv2
import numpy as np
import time


def salt(img, n):  
    for k in range(n):  
        i = int(np.random.random() * img.shape[1]);  
        j = int(np.random.random() * img.shape[0]);  
        if img.ndim == 2:   
            img[j,i] = 255  
        elif img.ndim == 3:   
            img[j,i,0]= 255  
            img[j,i,1]= 255  
            img[j,i,2]= 255  
    return img

#quick median blur:https://blog.csdn.net/rocketeerLi/article/details/88017306
class medianBlur():
    def __init__(self, kernel_size):
        super(medianBlur, self).__init__()
        self.size = kernel_size
        self.skip =  kernel_size // 2
    
    #  padding_way:REPLICA  or  ZERO
    def  padding(self,  img,  padding_way):
        print('origin shape of image:  ',  np.shape(img))
        if padding_way == 1:
            img =  np.pad(img, ((self.skip,self.skip),(self.skip,self.skip)), 'edge')
        else:  #padding zero by default
            img =  np.pad(img,((self.skip,self.skip),(self.skip,self.skip)), 'constant')
        print('after shape of image:  ',  np.shape(img))
        return img

    #return inputs median value
    def  getMedian(self,  input,  start,  end):
        #判断low是否小于high,如果为false,直接返回
        if start < end:
            i, j = start, end
            base = input[i]  #设置基准数
            while i < j:
                #如果列表后边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
                while (i < j) and (input[j] >= base):
                    j = j - 1
                #如找到,则把第j个元素赋值给第个元素i,此时表中i,j个元素相等
                input[i] = input[j]
    
                #同样的方式比较前半区
                while (i < j) and (input[i] <= base):
                    i = i + 1
                input[j] = input[i]
            #做完第一轮比较之后,列表被分成了两个半区,并且i=j,需要将这个数设置回base
            input[i] = base
    
            #递归前后半区
            self.getMedian(input, start, i - 1)
            self.getMedian(input, j + 1, end)
        return input[len(input)//2]
                
            
    def blur(self,  oriImg,  padding_way):
        #padding
        img =  self.padding(oriImg,  padding_way)  
        
        #blur, stride is 1 by default
        h,  w =  img.shape
        for row in range(self.skip,  h-self.size):
            H = np.zeros(256, dtype=int)    # 直方图
            input =  img[row-self.skip:row+self.skip+1,  0:self.size].reshape(-1)
            med =  np.uint8(self.getMedian(input,  0,  len(input) - 1) )  #get median value
            n = 0            

            for i in range(-self.skip, self.skip+1) :
                for j in range(0, self.size) :
                    H[img[row+i][j]] += 1
                    if img[row+i][j] <= med :
                        n = n + 1  
            for col in range(self.skip,  w-self.size) :
                if col == self.skip:
                    continue
                else:
                    for i in range(-self.skip, self.skip+1) :  # 更新直方图 并计算 n 的值
                        # 对左列元素 值减一 
                        H[img[row+i][col - (self.skip + 1)]] -= 1
                        if img[row+i][col - (self.skip + 1)] <= med :
                            n -= 1
                        # 对右列元素 值加一
                        H[img[row+i][col + self.skip]] += 1
                        if img[row+i][col + self.skip] <= med :
                            n +=  1
                    # 重新计算中值
                    threshold =  int(np.ceil(self.size**2 / 2))
                    if n > threshold :
                        while n > threshold :
                            if med == 0 :
                                break
                            n = n - H[med]
                            med = med - 1
                    elif n < threshold :
                        while n < threshold :
                            med = med + 1
                            n = n + H[med]

                # 更新中值后的直方图
                H[img[row][col]] = H[img[row][col]] - 1
                if med < img[row][col] :
                    n += 1
                img[row][col] = med
                H[med] += 1
    
        return img


if __name__ == "__main__":
    #add salt noise 
    img = cv2.imread('5.jpg')
    img =  img[:,  :,  0] #gray image
    img_salt = salt(img,  500)
    cv2.imshow('img_salt.jpg', img_salt)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
    
    start =  time.time()
    kernel_size =  3   #example values: 3, 5, 7, it must be odd integer
    padding_way = 0    #padding_way: 0(zero), 1(arounding)
    medianBlur = medianBlur(kernel_size)
    medianFeature =  medianBlur.blur(img_salt,  padding_way)
    #medianFeature =  cv2.medianBlur(img,  kernel_size)
    print('keep time: {:6f}'.format(time.time() -  start))
    #kernel_size：5，time:  13 second
    #kernel_size:3,  time: 9 seconds
    cv2.imshow('medianBlur.jpg', medianFeature)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()    
