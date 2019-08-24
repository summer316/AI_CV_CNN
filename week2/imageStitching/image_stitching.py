# -*-encoding:utf-8 -*-

import os
import cv2
import imutils
import numpy as np 
import argparse
"""
Pipeline of image stitching
    1: find features points in each image 
    2: use RANSCA to find keypoint matches
    3: use homography matrix to get transferring info
    4: merge two image
"""
from RANSAC import ransac_v2


class Stitcher():
    def __init__(self, args, threshold=4.0, ratio=0.75, use_ransac=True):
        # user Ransac algothim find all matches
        self.RANSAC = use_ransac
        self.args = args
        self.threshold = threshold
        self.ratio = ratio
        curPath = os.path.abspath(__file__)
        self.fatherPath = os.path.abspath(os.path.dirname(curPath) + os.path.sep + ".")
        self.imagePath = self.getImageList()
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.DescriptorMatcher_create("BruteForce")

    def getImageList(self):
        path = os.path.join(self.fatherPath, self.args.image_path)
        if not os.path.exists(path):
            raise FileExistsError("cann't find specified imagePath")
        filePathList = [os.path.join(path, f) for f in os.listdir(path)]
        if len(filePathList) <= 1:
            assert FileExistsError("there are two images at least.")
        return filePathList
    
    def drawMatch(self, imgA, imgB, kpsA, kpsB, matches, status):
        hA, wA, _ = imgA.shape
        hB, wB, _ = imgB.shape
        vis = np.zeros((max(hA, hB), wA+wB, 3), dtype=np.uint8)
        vis[:, 0:wA] = imgA
        vis[:, wA:] = imgB
        
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0])+wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        
        return vis

    def multiStitch(self, saveMatch=False):
        imgs = []
        for path in self.imagePath:
            img = cv2.imread(path)
            imgs.append(img)

        for i in range(1, len(imgs)):
            # support multiply images which numbers is more than two stitching
            if i < 2:
                srcImg = imgs[i-1]
            else:
                srcImg = result
                
            result, kpsA, kpsB, matches, status = self.stitcher(srcImg, imgs[i])

            #save matches and stitching images
            if saveMatch:
                vis = self.drawMatch(srcImg, imgs[i], kpsA, kpsB, matches, status)
                output_path = os.path.join(self.fatherPath, "result\\")
                cv2.imwrite(output_path+"vis.jpg", vis)
                cv2.imwrite(output_path+"result.jpg", result)

                # cv2.imshow(output_path+"vis.jpg", vis)
                # cv2.imshow(output_path+"result.jpg", result)
                # key = cv2.waitKey()
                # if 27 == key:
                #     cv2.destroyAllWindows()

    
    def matchkps(self, kpsA, kpsB, feaA, feaB):
        """
        para1: rawMatches：
        queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），
                    同时也是描述符对应特征点的下标。
        trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。    
        """
        rawMatches = self.matcher.knnMatch(feaA, feaB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append( (m[0].trainIdx, m[0].queryIdx) )

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
           
            # (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.threshold)
            (H, status)= ransac_v2(ptsA, ptsB)

            return (matches, H, status)

        return None

    def enhancement(self, img):
        result = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

		# create two copies of the mask: one to serve as our actual
		# minimum rectangular region and another to serve as a counter
		# for how many pixels need to be removed to form the minimum
		# rectangular region
        minRect = thresh.copy()
        sub = thresh.copy()

		# keep looping until there are no non-zero pixels left in the
		# subtracted image
        while cv2.countNonZero(sub) > 0:
			# erode the minimum rectangular mask and then subtract
			# the thresholded image from the minimum rectangular mask
			# so we can count if there are any non-zero pixels left
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

		# find contours in the minimum rectangular mask and then
		# extract the bounding box (x, y)-coordinates
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)

		# use the bounding box coordinates to extract the final stitched image
        return result[y:y + h, x:x + w]

    def stitcher(self, imgA, imgB):
        """
        SIFT 的返回值 
        para1: kps: 关键点。所包含的信息有： 
        angle：角度，表示关键点的方向，通过Lowe大神的论文可以知道，为了保证方向不变形，
               SIFT算法通过对关键点周围邻域进行梯度运算，求得该点方向。-1为初值。
        class_id：当要对图片进行分类时，可用class_id对每个特征点进行区分，未设定时为-1，需要手动设定
        octave：代表是从金字塔哪一层提取的得到的数据。
        pt：关键点点的坐标
        response：响应程度，代表该点强壮大小，更确切的说，是该点角点的程度。
        size：该点直径的大小

        para2: fea:描述子, 在关键点尺度空间内4*4的窗口中计算的8个方向的梯度信息，共4*4*8=128维向量表征
        """
        (kpsA, feaA) = self.sift.detectAndCompute(imgA, None)
        (kpsB, feaB) = self.sift.detectAndCompute(imgB, None)
        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])

        M = self.matchkps(kpsA, kpsB, feaA, feaB)
        if M is None:
            return None

        (matches, H, status) = M
        
        shft = np.array([[1.0,0,imgA.shape[1]],[0,1.0,0],[0,0,1.0]])
        M = np.dot(shft, H)
        result = cv2.warpPerspective(imgA, M, (imgA.shape[1]+imgB.shape[1], imgA.shape[0]))
        result[0:imgB.shape[0], imgA.shape[1]:] = imgB
        
        #获取拼接图像的去除背景的最大内接区域
        result = self.enhancement(result)
       
        return result, kpsA, kpsB, matches, status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    parser.add_argument("--image_path", type=str, default="testImage/tajm/", 
                        help="Enter the image path which including in order of left to right in way of concantenation images")

    args = parser.parse_args()
    #include image path at least
    initF = Stitcher(args, threshold=4.0, ratio=0.75, use_ransac=True)
    try:
        initF.multiStitch(saveMatch=True)
    except Exception:
        raise print("check everything")
        