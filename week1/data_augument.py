# _*_ encoding:utf-8 _*_
import numpy as np
import random
import time
import sys
import os
import cv2


def get_images(path):
    '''
    get all files index in "path" catalog, include subdirectory
     '''    
    #windows下分大小写，Linux下分
    ext=['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'bmp']
    listFiles = []
    if os.path.isfile(path):
        if path.rpartition('.')[2] in ext:
            listFiles.append(path)
        return listFiles
    pt = os.walk(path)
    for t in pt:
        if len(t[2]) > 0:
            listFiles.extend([os.path.join(t[0], fileName) for fileName in t[2] if fileName.rpartition('.')[2] in ext])
    return listFiles

class dataAugument():
    """ """
    def __init__(self):
        pass
    
    def interface(self, img, method=-1, type=-1):
        h, w, _ = img.shape
        ## 基于空间，例如裁剪缩放平移翻转旋转反射仿射 视觉变换
        if method == 0:  
            # 0:getRotationMatrix2D 1:getAffineTransform 2:getPerspectiveTransform 4:others
            if type == 0:
                angle = 120
                ratio = 1.0
                aug_img = self.random_warp(img, h, w, type=type, angle=angle, ratio=ratio)
            else:
                aug_img = self.random_warp(img, h, w, type=type)

        # 基于颜色，例如对比度亮度Gamma调整色彩PCA抖动 锐化浮雕 噪声 随机通道变换或擦除
        elif method == 1:  
            aug_img = self.random_light_color(img)
            gamma = 2.0
            aug_img = self.adjust_gamma(aug_img, gamma=gamma)

        # 基于空域 例如二值化 （非）线性模糊 RGB_to_HSV
        elif method == 2:  
            ## 灰度化的（R +B + G)/3 or  0.3*R + 0.3*G +0.4*B
            GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            minValue, maxValue = 127, 255      
            ret, aug_img = cv2.threshold(GrayImage, minValue, maxValue,cv2.THRESH_TOZERO)

        # 基于频域 低通高通滤波 FrequencyNoise90Alpha             
        elif method == 3:
            #seq = iaa.FrequencyNoiseAlpha(first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True))
            #aug_img = seq.augment_images(img)
            pass

        # others(多样本合成, gan, mixup, AutoAugment， 图像拼接等)
        else:
            aug_img = img

        return img


    # random_crop 处理基于空间变换时的黑色背景
    def removeBlack(self, image):
        return image

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0/gamma
        table = []
        for i in range(256):
            table.append(((i / 255.0) ** invGamma) * 255)
        table = np.array(table).astype("uint8")
        return cv2.LUT(image, table)

    def random_light_color(self, img):
        # brightness
        B, G, R = cv2.split(img)
        b_rand = random.randint(-50, 50)
        if b_rand == 0:
            pass
        elif b_rand > 0:
            lim = 255 - b_rand
            B[B > lim] = 255
            B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
        elif b_rand < 0:
            lim = 0 - b_rand
            B[B < lim] = 0
            B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

        g_rand = random.randint(-50, 50)
        if g_rand == 0:
            pass
        elif g_rand > 0:
            lim = 255 - g_rand
            G[G > lim] = 255
            G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
        elif g_rand < 0:
            lim = 0 - g_rand
            G[G < lim] = 0
            G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

        r_rand = random.randint(-50, 50)
        if r_rand == 0:
            pass
        elif r_rand > 0:
            lim = 255 - r_rand
            R[R > lim] = 255
            R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
        elif r_rand < 0:
            lim = 0 - r_rand
            R[R < lim] = 0
            R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

        img_merge = cv2.merge((B, G, R))
        return img_merge

    def random_warp(self, img, h=0, w=0, type=-1, **argv):
        if  type == 0:
            '''
            # 得到变换的矩阵，通过这个矩阵再利用warpAffine来进行变换
            :param:旋转中心，元组的形式，这里设置成图像中心
            :param:旋转的角度， 正值表示逆时针旋转
            :param 表示放缩的系数，1表示保持原图大小
            '''
            angle = argv["angle"]
            ratio = argv["ratio"]
            M = cv2.getRotationMatrix2D((w / 2, h/ 2), angle, ratio)

        elif type == 1:
            ## https://www.520mwx.com/view/7117
            #保持平行性，不保持角度的变换
            #dst(a00 * X + a01 * Y + b0, a10 * X, a11 * Y, b1) = src(X, Y)
            pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]]) 
            pts2 = np.float32([[w * 0.2, h * 0.1], [w * 0.9, h * 0.2], [w * 0.1, h * 0.9]])
            M = cv2.getAffineTransform(pts1, pts2)

        elif type == 2:
            ## https://scm_mos.gitlab.io/vision/homography-matrix/
            random_margin = 60
            x1 = random.randint(-random_margin, random_margin)
            y1 = random.randint(-random_margin, random_margin)
            x2 = random.randint(w - random_margin - 1, w - 1)
            y2 = random.randint(-random_margin, random_margin)
            x3 = random.randint(w - random_margin - 1, w - 1)
            y3 = random.randint(h - random_margin - 1, h - 1)
            x4 = random.randint(-random_margin, random_margin)
            y4 = random.randint(h - random_margin - 1, h - 1)
        
            dx1 = random.randint(-random_margin, random_margin)
            dy1 = random.randint(-random_margin, random_margin)
            dx2 = random.randint(w - random_margin - 1, w - 1)
            dy2 = random.randint(-random_margin, random_margin)
            dx3 = random.randint(w - random_margin - 1, w - 1)
            dy3 = random.randint(h - random_margin - 1, h - 1)
            dx4 = random.randint(-random_margin, random_margin)
            dy4 = random.randint(h - random_margin - 1, h - 1)
        
            pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
            # 先用四个点来确定一个3*3的变换矩阵
            M_warp = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M_warp, (w, h))
            return img
        else:
            tx = np.random.randint(50)
            ty = np.random.randint(100) 
            M = np.float32([[1,0,tx],[0,1,ty]]) #表示平移变换,移动的距离为（tx,ty）
           
        img = cv2.warpAffine(img, M, (h, w)) 
        return img



def main(imgPath, inputSize, outputPath):
    #获取图像输入输出路径
    assert imgPath, 'please input path of image'
    if not os.path.exists(imgPath):
        os.mkdir(imgPath)
        
    #获取图像随机列表
    image_list = np.array(get_images(imgPath))
    print('{} training images in {}'.format(image_list.shape[0], imgPath))
    index = np.arange(0, image_list.shape[0])
    np.random.shuffle(index)
    
    start = time.time()
    augu = dataAugument()
    for i in index:
        try:
            im_fn = image_list[i]
            im = cv2.imread(im_fn)

            # augument image in random method
            method = np.random.randint(4, size=1)
            type = np.random.randint(4, size=1)
            im = augu.interface(im, method, type)
            # 缩小使用INTER_AREA，放缩使用INTER_CUBIC(较慢)和INTER_LINEAR(较快效果也不错)。
            #默认情况下，所有的放缩都使用INTER_LINEAR。
            im = cv2.resize(im, dsize=(inputSize, inputSize), interpolation=cv2.INTER_AREA) 
            cv2.imwrite(outputPath + os.path.basename(im_fn).rpartition('\/')[-1], im)

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    print('Cost {:.3f} second for {:} images'.format(time.time()-start, len(index)))


if __name__ == "__main__":
    imgPath = "./test/"
    inputSize = 224
    outputPath = "./result/"
    main(imgPath, inputSize, outputPath)
