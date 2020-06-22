import numpy as np
import torch
import cv2

def iou(box,boxes, is_min = False):
    box_area = (box[2]-box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:,2]-boxes[:,0]) * (boxes[:,3] - boxes[:,1])

    inter_x1 = np.maximum(box[0],boxes[:,0])
    inter_y1 = np.maximum(box[1],boxes[:,1])
    inter_x2 = np.minimum(box[2],boxes[:,2])
    inter_y2 = np.minimum(box[3],boxes[:,3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)

    inter = inter_w * inter_h

    if is_min:
        return inter / np.minimum(box_area,boxes_area)
    else:
        return inter / (box_area + boxes_area - inter)

def nms(boxes,threshold, is_min = False):
    if boxes.shape[0] == 0: return np.array([])
    _boxes = boxes[(-boxes[:,4]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]

        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        _boxes = b_boxes[iou(a_box,b_boxes,is_min)<threshold]

    if _boxes.shape[0] == 1:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)

def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size,expected_size
    scale = min(eh / ih, ew / iw) # 最大边缩放至416得比例
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC) # 等比例缩放，使得有一边416
    top = (eh - nh) // 2 # 上部分填充的高度
    bottom = eh - nh - top  # 下部分填充的高度
    left = (ew - nw) // 2 # 左部分填充的距离
    right = ew - nw - left # 右部分填充的距离
    # 边界填充
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img, nh, nw, top, left

if __name__ == '__main__':
    # a = np.array([1,1,11,11])
    # bs = np.array([[1,1,10,10],[11,11,20,20]])
    # print(iou(a,bs))

    bs = torch.tensor([[1, 1, 10, 10, 40,8], [1, 1, 9, 9, 10,9], [9, 8, 13, 20, 15,3], [6, 11, 18, 17, 13,2]])
    # print(bs[:,3].argsort())
    print(nms(bs))
