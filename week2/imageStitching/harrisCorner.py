# -*- encoding:utf-8 -*-
import os
import cv2
import numpy as np
import argparse

from skimage.feature import corner_harris, corner_peaks
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import signal as sig

"""
process of harris corner detection algorithm:
    1:Color image to Grayscale conversion
    2:Spatial derivative calculation
    3:Structure tensor setup
    4:Harris response calculation
    5:Find edges and corners using R
"""

class HarrisCorner():
    def __init__(self, args):
        self.rootPath = os.getcwd()
        #step 1: img to gray                   
        self.img = imread(os.path.join(self.rootPath, "./5.jpg"))
        self.imgGray = rgb2gray(self.img)
        cv2.imshow('grayImg', self.imgGray)

    def interface(self, k=0.04, window_size=3):
        #step 2: Spatial derivative calculation
        I_x = self.gradient_x(self.imgGray)
        I_y = self.gradient_y(self.imgGray)
        
        #step 3: Structure tensor setup
        Ixx, Iyy, Ixy = self.structureTensor(I_x, I_y)

        ##step 4:
        #window_size = 3, offset=windowsize/2=1
        # k range[0.04, 0.06] k is the sensitivity factor to separate corners from edges, 
        # typically a value close to zero, for my analysis I have taken k=0.04. 
        # Small values of k result in detection of sharp corners.
        imgGray = cv2.dilate(self.imgGray, None)
        h, w = imgGray.shape
        offset = int(window_size/2)
        for y in range(offset, h):
            for x in range(offset, w):
                Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
                Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
                Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

                det = (Sxx * Syy) - (Sxy ** 2)
                trace = Sxx + Syy
                r = det - k*(trace ** 2)
                #step 5: 
                if r > 0:
                    self.img[y,x] = [255,0,0]
                    
        self.img = cv2.dilate(self.img , None)

        cv2.imshow('img_harris1', self.img)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

    ## step 2:
    def gradient_x(self, imgGray):
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        return sig.convolve2d(imgGray, kernel_x, mode="same")

    def gradient_y(self, imgGray):
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        return sig.convolve2d(imgGray, kernel_y, mode="same")    

    def structureTensor(self, I_x, I_y):
        ##step 3: structure tensor setup
        Ixx = I_x ** 2
        Iyy = I_y ** 2
        Ixy = I_x * I_y
        return Ixx, Iyy, Ixy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    parser.add_argument("--input_image", type=int, default=None, help="Enter the image name in order of left to right in way of concantenation:")
 
    args = parser.parse_args()
    harris = HarrisCorner(args)
    harris.interface(k=0.04, window_size=3)
