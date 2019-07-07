# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
import random
import time
import sys
import os
import cv2

flags = tf.app.flags
flags.DEFINE_integer('input_size', 224, 'images path')
flags.DEFINE_string('image_path', './test_img/', 'images path')
flags.DEFINE_string('output_path', './outputs/', 'images path after data augmentation')
FLAGS = flags.FLAGS

def get_images(path):
    '''
    获得path目录（文件）下的所有的文件序列，包括子目录
    '''    
    #windows下不分大小写，Linux下分
    ext=['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'bmp']
    listFiles = []
    if not os.path.exists(path):
        return listFiles
    if os.path.isfile(path):
        if path.rpartition('.')[2] in ext:
            listFiles.append(path)
        return listFiles
    pt = os.walk(path)
    for t in pt:
        if len(t[2]) > 0:
            listFiles.extend([os.path.join(t[0], fileName) for fileName in t[2] if fileName.rpartition('.')[2] in ext])
    return listFiles

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


def random_light_color(img):
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

def random_warp(img, h=0, w=0, c=0,  type=None):
    if  type == 'getRotationMatrix2D':
        '''
        # 得到变换的矩阵，通过这个矩阵再利用warpAffine来进行变换
        :param:旋转中心，元组的形式，这里设置成相片中心
        :param:旋转的角度
        :param 表示放缩的系数，1表示保持原图大小
        '''
        M = cv2.getRotationMatrix2D((w / 2, h/ 2), 120, 1)
        img = cv2.warpAffine(img, M, (w, h))
    elif type == 'getAffineTransform':
        #保持平行性，不保持角度的变换
        pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        pts2 = np.float32([[w * 0.2, h * 0.1], [w * 0.9, h * 0.2], [w * 0.1, h * 0.9]])
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (w, h))
    elif type == 'getPerspectiveTransform':
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
    else:
        tx = np.random.randint()
        ty = np.random.randint(100) 
        H = np.float32([[1,0,tx],[0,1,ty]]) #表示平移变换,移动的距离为（tx,ty）
        '''
        :para2:输出图像的大小
        :para3: 输出图像的大小
        '''
        res = cv2.warpAffine(img,H,(h,w)) 
    return img


def augmentation(img):
    '''
    :param img: .......
    :return img: augmentation image
    '''    
    #基于空间、多样本合成多为增加样本，及提升泛化能力，基于颜色部分是此功能 
    #后面多为凸显目标特征，提高图像质量，如去噪加噪，分割。
    method = [0, 1, 2, 3, 4, 5]
    random = np.random.randint(3, size=1)
    h,  w,  c =  img.shape
    random =  2
    if random == 0:  ##基于空间
        #type =  getRotationMatrix2D,getAffineTransform, getPerspectiveTransform
        aug_img = random_warp(img, h, w, c,  type='getPerspectiveTransform')
    elif random == 1:  #基于颜色
        aug_img =  random_light_color(img)
        aug_img =  adjust_gamma(aug_img, gamma=2.0)
    elif random == 2:  #基于空域
        GrayImage=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        ret, aug_img=cv2.threshold(GrayImage,127,255,cv2.THRESH_TOZERO)             
    elif random == 3: #基于频域
        #seq = iaa.FrequencyNoiseAlpha(first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True))
        #aug_img = seq.augment_images(img)
        pass
    elif random == 4: #多样本合成
        pass
    elif random == 5:  #others(gan, AutoAugment)
        pass    
    else:
        return img

    return aug_img



def main(argv=None):
    #获取图像输入输出路径
    assert FLAGS.image_path, 'please input image_path=path of image'
    if not tf.gfile.Exists(FLAGS.output_path):
        tf.gfile.MkDir(FLAGS.output_path)    
        
    #获取图像随机列表
    image_list = np.array(get_images(FLAGS.image_path))
    print('{} training images in {}'.format(image_list.shape[0], FLAGS.image_path))
    index = np.arange(0, image_list.shape[0])
    np.random.shuffle(index)
    
    start = time.time()
    start_1 = start
    for i in index:
        try:
            im_fn = image_list[i]
            im = cv2.imread(im_fn)
            im = augmentation(im)#
            # 缩小使用INTER_AREA，放缩使用INTER_CUBIC(较慢)和INTER_LINEAR(较快效果也不错)。
            #默认情况下，所有的放缩都使用INTER_LINEAR。
            im = cv2.resize(im, dsize=(FLAGS.input_size, FLAGS.input_size), interpolation=cv2.INTER_AREA) 
            cv2.imwrite(FLAGS.output_path + os.path.basename(im_fn).rpartition('\/')[-1], im)
            print('{:.3f} second/image'.format(time.time() -start))
            start = time.time()
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    print('Cost {:.3f} second for {:} images'.format(time.time()-start_1, len(index)))
        
            
            
if __name__ == '__main__':
    tf.app.run()