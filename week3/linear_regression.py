#_*_ encoding:utf-8 -^-
import numpy as np
import  time
import numpy.random as random

class simpleLinearRegression:
    def  __init__(self):
        self.w1 = random.randint(0, 10) + random.random()
        self.w2 = random.randint(0, 10) + random.random()
        self.b = random.randint(0, 5) + random.random()
        self.lambd = None  #0.2  #None is betten than add L2,???
    
    def  loss(self,  loss):
        #loss function   (true-predicy)**2/(2*len(X))  #除法比乘法慢
        if self.lambd:
            loss =  np.mean(np.square(loss) +  np.sum(self.w1**2+ self.w2**2)) * 0.5 #
        else:
            loss =  np.mean(np.square(loss)) * 0.5
        return loss
        

    def  gradientDescent(self,  loss,  X, alpha,  w1,  w2,  b, lambd):
        #update weight, bias
        delta_w1,  delta_w2 =  np.dot(X,  loss)  * 0.1  #0.1表示除以10
        delta_b =  np.mean(loss)
        
        #add l2 regularization
        #有正则化的时候，收敛变慢，因为weight的更新变慢，
        if lambd:
            #print('cost function with L2  regularization' )
            w1 =  w1 *  (1 -  alpha * lambd * 0.1)  -   (alpha * lambd * delta_w1)
            w1 =  w1 *  (1 -  alpha * lambd  * 0.1)  -   (alpha * lambd * delta_w2)
            b -=  alpha *  lambd *  delta_b            
        else:  
            w1 -= alpha * delta_w1
            w2 -=  alpha *  delta_w2
            b  -= alpha * delta_b
            
        return w1,  w2,  b     
        
    def  linear(self, X,  w1,  w2,  b):
        weight =  [w1,  w2]
        #linear regression function
        return  np.dot(weight,  X) +  b  

    def  train(self,  X, Y, epoch,  alpha):
        for i in range(epoch):
            yPred = self.linear(X,  self.w1,  self.w2,  self.b)
            self.w1,  self.w2,  self.b =  self.gradientDescent(yPred-Y,  X ,
                                                               alpha,  self.w1,  self.w2,  self.b,  self.lambd)

            print('in {:} epoch,  loss : {:.4f}'.format(i,  self.loss(yPred - Y)))        
    

def  normalization(X,  Y):
    #max min value  normalization
    xMin =  [np.min(i) for i in  X]
    xMax =  [np.max(i) for i in X ]
    yMin =  np.min(Y)
    yMax =  np.max(Y)
    update =  []
    for  i in range(len(X)):
        if (xMax[i] -  xMin[i]) !=  0:
            #下面一句，返回的值是整形的，即使类型转换也是整形的值，不知道为什么
            #X[i] =  (X[i] - xMin[i]) / (xMax[i] -  xMin[i])  
            update.append((X[i] - xMin[i]) / float(xMax[i] -  xMin[i]) )
        else:
            update.append(X)
            
    if   (yMax -  yMin) !=  0:
        Y =  (Y - yMin) /  (yMax -  yMin)
    
    return update, Y


def  generateData():
    X = [[2104,3],  [1600, 3], [2400, 3], [1416, 2], [3000, 4], [1985, 4], [1534, 3], [1427, 3], [1380, 3], [1494 , 3]]
    Y = [399900,  329900,  369000,  232000,  539900,  299900,  314900,  198999,  212000,  242500]
    return np.transpose(X,  (1,  0)),  Y


def  main( ):
    #generate data, x [10 ,2], y[10]
    X,  Y =  generateData()
    
    #normalization
    X,  Y =  normalization(X,  Y)
    
    #hyperparameter
    epoch =  1000
    alpha = 1e-1
    
    #train
    start =  time.time()
    model =  simpleLinearRegression()
    model.train(X,  Y, epoch,  alpha)
    print('time  {:.4f}'.format(time.time() -  start))
    

if __name__ == '__main__':	
    main()
