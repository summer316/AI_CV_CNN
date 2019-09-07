# -*- encoding:utf-8 -*-
import cv2
import numpy as np


import numpy as np

def soft_nms(bounding_boxes, iou_thresh=0.3, sigma2=0.5, score_thresh=0.001, method=2):
    '''
    # para 1 [[x1, y1, x2, y2, cls_scores], [...], [...], ...]
    # para 2: threshold of IOU
    # para 3: sigma2
    # para 4：score_thresh
    # para 5: gassuian and linear method
    '''
    ## step 1: get areas of every box
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    scores = bounding_boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
     
    ## step 2: add index in last dimension
    N = bounding_boxes.shape[0]
    indexes = np.array([np.arange(N)])
    bounding_boxes = np.concatenate((bounding_boxes, indexes.T), axis=1)

    for i in range(N):
        ## step 3: find out max scores and index
        pos = i + 1
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
    
        ## step 4: makesure max scores is current value
        if scores[i] < maxscore:
            bounding_boxes[[i, maxpos+i+1]] = bounding_boxes[[maxpos+i+1, i]]
            scores[[i, maxpos+i+1]] = scores[[maxpos+i+1, i]]
            areas[[i, maxpos+i+1]] = areas[[maxpos+i+1, i]]
        
        ## step 5: calculate iou
        xx1 = np.maximum(x1[i], x1[pos:])
        yy1 = np.maximum(y1[i], y1[pos:])
        xx2 = np.minimum(x2[i], x2[pos:])
        yy2 = np.minimum(y2[i], y2[pos:])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 - 1)
        inner_area = w * h
        iou = (inner_area)/(areas[i] + areas[pos:] - inner_area)
            
        ## step 6: calculate weight
        if 1 == method:
            weight = np.zeros(iou.shape)
            weight[iou > iou_thresh] = weight[iou > iou_thresh] - iou[iou > iou_thresh]
        elif 2 == method:
            weight = np.exp(-(iou * iou) / sigma2)
        else:
            weight = np.zeros(iou.shape)
            weight[iou > iou_thresh] = 0
        
        ## step 7: weight * scores
        scores[pos:] = weight * scores[pos: ]

        
    ## step :8 将iou小于阈值的BBox 保存作为下一轮迭代输入
    index = np.where(scores > score_thresh)
    keep = bounding_boxes[index]
    return keep


def nms(bounding_boxes, threshold=0.7):
    '''
    # para 1 [[x1, y1, x2, y2, cls_scores], [...], [...], ...]
    # para 2: threshold of IOU
    '''
    
    ## step 1: get areas of every box
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    scores = bounding_boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    ## step 2: sort BBoxes according scores from min to max
    order = np.argsort(scores)

    ## step 3: 迭代保留下来的BBoxes
    target = []
    while order.size > 0:
        index = order[-1] # get max bbox
        target.append(bounding_boxes[index])

        ## step 4:求置信度最大的框与其它框的IOU
        xx1 = np.maximum(x1[index], x1[order[:-1]])
        yy1 = np.maximum(y1[index], y1[order[:-1]])
        xx2 = np.minimum(x2[index], x2[order[:-1]])
        yy2 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 - 1)
        inner_area = w * h

        iou = (inner_area)/(areas[index] + areas[order[: -1]] - inner_area)
        
        ## step 5: 将iou小于阈值的BBox 保存作为下一轮迭代输入
        left = np.where(iou < threshold)
        order = order[left]
    
    return target





data = np.array([[200, 200, 400, 400, 0.9], [220, 220, 420, 420, 0.8], [200, 240, 400, 440, 0.7], 
                  [240, 200, 440, 400, 0.6], [1, 1, 2, 2, 0.5]], dtype=np.float32)

if __name__ == "__main__":
    img = np.zeros((500, 500)) + 255
    if False:
        ret = nms(data, threshold=0.7)
    else:
        ret = soft_nms(data, 0.3, 0.5, 0.001, 1)
    print("origin data: {}\n, after nms: {}".format(data, ret))
    for box in ret:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
    
    cv2.imwrite("./result.jpg", img)
   
        
    