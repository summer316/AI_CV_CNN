# -*- encoding:utf-8 -*-
"""
http://shartoo.github.io/HOG-feature/
https://www.cnblogs.com/zhazhiqiang/p/3595266.html
progress of HOG:
    step 1: gamma correct and to gray
    step 2: compute gradient of every pixes
    step 3: compute gradient histogram of ervery cell
    step 4: compute gradient histogram of every block
    step 5: normalization every block
"""

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

# code resource:https://www.qingtingip.com/h_282713.html
def getHOGfeat(image, stride=8, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(2,2)):

    #step 1: gamma correct, reduce illumination influence
    image = np.power(image/float(np.max(image)), 1/1.5)

    cx, cy = pixels_per_cell
    bx, by = cells_per_block
    sx, sy = image.shape
    n_blocksx = (sx - cx*bx)//(cx*bx) + 1 
    n_blocksy = (sy - cy*by)//(cy*by) + 1 
    gx = np.zeros((sx, sy), dtype=np.double)
    gy = np.zeros((sx, sy), dtype=np.double)
    eps = 1e-5
    grad = np.zeros((sx, sy, 2), dtype=np.double)

   #step 2: compute Horizontal and vertical gradient for every pixes
    for i in range(1, sx-1):
        for j in range(1, sy-1):
            gx[i, j] = image[i, j-1] - image[i, j+1]
            gy[i, j] = image[i+1, j] - image[i-1, j]
            grad[i, j, 0] = np.arctan(gy[i, j] / (gx[i, j] + eps)) * 180 / math.pi
            if gx[i, j] < 0:
                grad[i, j, 0] += 180
            grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
            grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)

    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by * bx * orientations))
    #step 3: compute gradient histogram of ervery cell  and of every block
    for y in range(n_blocksy): # for every block
        for x in range(n_blocksx):
            block = grad[y*stride:y*stride+16, x*stride:x*stride+16]
            hist_block = np.zeros(32, dtype=np.double)
            for k in range(by): #for every cell
                for m in range(bx):
                    cell = block[k*8:(k+1)*8, m*8:(m+1)*8]
                    # print(np.shape(cell))
                    hist_cell = np.zeros(8, dtype=np.double)
                    for i in range(cy):
                        for j in range(cx):
                            try:
                                n = int(cell[i, j, 0] / 45) 
                                hist_cell[n] += cell[i, j, 1]
                            except IndexError:
                                print("......")    
                            
                    hist_block[(k * bx + m) * orientations:(k * bx + m + 1) * orientations] = hist_cell[:]
            # step 5: normalization every block
            normalised_blocks[y, x, :] = hist_block / np.sqrt(hist_block.sum() ** 2 + eps)
    return normalised_blocks

if __name__ == "__main__":
    curPath = os.path.abspath(__file__)
    fatherPath = os.path.abspath(os.path.dirname(curPath) + os.path.sep + ".")
    img = cv2.imread(os.path.join(fatherPath, "testImage/Hill/Hill1.jpg"), 0)
    cv2.imshow("origin", img)
    key = cv2.waitKey()
    if 27 == key:
        cv2.destroyAllWindows()

    if True:
        hog_image = getHOGfeat(img, 8, 8, (8, 8),(2,2))
    else:
        _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

    key = cv2.waitKey()
    if 27 == key:
        cv2.destroyAllWindows()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()