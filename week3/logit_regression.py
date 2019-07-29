#_*_ encoding:utf-8 -^-
import numpy as np
import  time
import numpy.random as random
from sklearn.datasets import make_multilabel_classification

class simpleLogisticRegression:
    def __init__(self):
        self.w1 = random.randint(0, 10) + random.random()
        self.w2 = random.randint(0, 10) + random.random()
        self.b = random.randint(0, 5) + random.random()
        self.lambd = None    #None is betten than L2 in this example
    
    def loss(self,  loss):
        #loss function   (true-predicy)**2/(2*len(X))  #除法比乘法慢
        if self.lambd:
            loss =  np.mean(np.square(loss) +  np.sum(self.w1**2+ self.w2**2)) * 0.5 #
        else:
            loss =  np.mean(np.square(loss)) * 0.5
        return loss
        

    def gradientDescent(self, loss, X, alpha, w1, w2, b, lambd):
        #update weight, bias
        delta_w1, delta_w2 = np.dot(X, loss) * 0.1  #0.1表示除以10
        delta_b = np.mean(loss)
        
        #add l2 regularization
        #有正则化的时候，收敛变慢，因为weight的更新变慢，
        if lambd:
            #print('cost function with L2  regularization' )
            w1 =  w1 * (1 - alpha * lambd * 0.1) - (alpha * lambd * delta_w1)
            w1 =  w1 * (1 - alpha * lambd * 0.1) - (alpha * lambd * delta_w2)
            b -=  alpha * lambd * delta_b            
        else:  
            w1 -= alpha * delta_w1
            w2 -= alpha * delta_w2
            b -= alpha * delta_b
            
        return w1, w2, b     
        
    def  linear(self, X, w1, w2, b):
        weight = np.reshape([w1, w2], (-1,  2))
        z = np.dot(weight,  X) +  b  
        #logistic regression function
        return 1.0/(1+np.exp(-z))

    def  train(self, X, Y, epoch, alpha):
        for i in range(epoch):
            yPred = self.linear(X, self.w1, self.w2, self.b)
            yPred = np.reshape(yPred, (-1,  1))
            self.w1, self.w2, self.b = self.gradientDescent(yPred-Y, X,alpha, 
                                                            self.w1, self.w2, self.b, self.lambd)

            print('in {:} epoch,  loss : {:.4f}'.format(i, self.loss(yPred - Y)))        
    

def normalization(X):
    #max min value  normalization
    xMin =  [np.min(X[:,  0]), np.min(X[:,  1])]
    xMax =  [np.max(X[:,  0]), np.max(X[:,  1])]
    update =  []
    for i in range(2):
        if (xMax[i] -  xMin[i]) != 0:
            #下面一句，返回的值是整形的，即使类型转换也是整形的值，不知道为什么
            #X[i] = (X[:,i] - xMin[i]) / (xMax[i] -  xMin[i])  
            update.append((X[:, i] - xMin[i]) / float(xMax[i] - xMin[i]) )
        else:
            update.append(X[:, i])
    
    return update


def main( ):
    #generate data, x [50,2], y[50, 1]
    X, Y = make_multilabel_classification(n_samples=50,n_features=2,n_classes=1,random_state=0)
    
    #normalization
    X = normalization(X) 
    
    #hyperparameter
    epoch = 1000
    alpha = 1e-1
    
    #train
    start = time.time()
    model = simpleLogisticRegression()
    model.train(X, Y, epoch, alpha)
    print('time {:.4f}'.format(time.time() - start))
    

if __name__ == '__main__':	
    main()
