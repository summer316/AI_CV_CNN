# _*_encoding:utf-8 _*_
import cv2
import numpy as np
import time

'''
pseudo of CTMF
Input: image X of size m*n, kernel radius r.
output: image Y as X.
for i = r to m - r do 
　　for j = r to n - r do
　　   initialize list A[]
　　　　for a = i-r to i+r
　　　　　　for b = j-r to j+r
　　　　　　　　add X(a, b) to  A[]
　　　　　　end 
　　　　end
　　　　sort A[] then Y(i ,j) = A[A.size/2]
　　end
end
'''

def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])  
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 3:   
            img[j, i, 0:3] = 255  
        else:  
            img[j, i] = 255 
            
    return img

#quick median blur:https://blog.csdn.net/rocketeerLi/article/details/88017306
class medianBlur():
    def __init__(self, kernel_size):
        self.size = kernel_size
        self.skip = kernel_size // 2
    
    # padding_way:REPLICA  or  ZERO
    def padding(self, img, padding_way):
        print('origin shape of image:  ',  np.shape(img))
        if padding_way == 1:
            img = np.pad(img, ((self.skip,self.skip),(self.skip,self.skip)), 'edge')
        #padding zero by default
        else:  
            img =  np.pad(img,((self.skip,self.skip),(self.skip,self.skip)), 'constant')
        print('after shape of image:  ',  np.shape(img))
        
        return img
    
    def getMedian(self, input, start, end):
        #判断low是否小于high,如果为false,直接返回
        if start < end:
            i, j = start, end
            base = input[i]  #设置基准数
            while i < j:
                #如果列表后边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
                while (i < j) and (input[j] >= base):
                    j -= 1

                #如找到,则把第j个元素赋值给第个元素i,此时表中i,j个元素相等
                input[i] = input[j]

                #同样的方式比较前半区
                while (i < j) and (input[i] <= base):
                    i += 1

                input[j] = input[i]

            #做完第一轮比较之后,列表被分成了两个半区,并且i=j,需要将这个数设置回base
            input[i] = base

            #递归前后半区
            self.getMedian(input, start, i - 1)
            self.getMedian(input, j + 1, end)

        return input[len(input)//2]


    def blur(self, img, paddingWay=0):
        # img = self.padding(img, paddingWay)
        h, w = img.shape

        for row in range(self.skip, h-self.skip):
            #init histogram
            n = 0 #reduce the step for compare
            H = np.zeros(256, dtype=int)
    
            #get median in the first step for every row
            kernerOutput = img[row-self.skip:row+self.skip+1, 0:self.size]
            kernerOutput = kernerOutput.reshape(-1)
            med = int(self.getMedian( kernerOutput, 0, len(kernerOutput)-1 ))

            #update hist in the first step for every row
            for i in range(-self.skip, self.skip + 1):
                for j in range(0, self.size):
                    H[img[row+i][j]] += 1
                    if img[row+i][j] <= med:
                        n += 1

            for col in range(self.skip, w-self.skip):
                if self.skip == col:
                    None
                
                else:
                    for k in range(-self.skip, self.skip+1):  # 更新直方图 并计算 n 的值
                        # 对左列元素 值减一 
                        H[img[row+k][col -(self.skip + 1)]] -= 1
                        if img[row+k][col -(self.skip + 1)] <= med:
                            n -= 1
                        # 对右列元素 值加一
                        H[img[row+k][col + self.skip]] += 1
                        if img[row+k][col + self.skip] <= med:
                            n += 1

                    # 重新计算中值
                    threshold = int(np.ceil(self.size**2 / 2))
                    medList = list(np.where(H != 0)[0])
                  
                    if n > threshold:
                        while n > threshold:
                            if (n - H[med] < threshold):
                                break           
                            n -= H[med]
                            med = medList[medList.index(med)-1]
                            
                        
                    elif n < threshold:
                        while n < threshold:
                            if med == medList[-1]:
                                break
                            med = medList[medList.index(med)+1]
                            n += H[med]
                    
                            

                H[img[row][col]] -= 1
                if med < img[row][col]:
                    n += 1
                
                # if np.median(img[row-self.skip:row+self.skip+1, col-self.skip:col+self.skip+1]) != med:
                #     print(img[row-self.skip:row+self.skip+1, col-self.skip:col+self.skip+1], med)
                img[row][col] = med
                H[med] += 1

        return img


if __name__ == "__main__":
    img = cv2.imread("./5.jpg", flags=0)
    #add salt noise, 
    imgSalt = salt(img, 500)
    cv2.imshow("imgSalt.jpg", imgSalt)
    key = cv2.waitKey()
    if 27 == key:
        cv2.destroyAllWindows()

    start = time.time()
    kernerSize = 3   #example values: 3, 5, 7, it must be odd integer
    paddingWay = 0    #padding_way: 0(zero), 1(arounding)
    medianBlur = medianBlur(kernerSize)
    medianFeature = medianBlur.blur(imgSalt, paddingWay)
    # medianFeature = cv2.medianBlur(imgSalt,  kernerSize)
    print('keep time: {:6f}'.format(time.time() - start))
    cv2.imshow('medianBlur.jpg', medianFeature)
    key = cv2.waitKey()
    if 27 == key:
        cv2.destroyAllWindows()