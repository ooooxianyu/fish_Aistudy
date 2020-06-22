import numpy as np
import tool

TRAIN_LABEL_FILE_PATH = "data/train_label.txt"
PRE_LABEL_FILE_PATH = "data/pre_label.txt"
train_label = open(TRAIN_LABEL_FILE_PATH,"r")
pre_label = open(PRE_LABEL_FILE_PATH,"r")
threshold = np.arange(0,1,0.1)

hx = np.zeros((5,3),dtype=np.int) # 5个类别 每个类别都有 TP FP TP+FN
for thresh in threshold:
    for line1, line2 in zip(train_label, pre_label):
        #print(line1,line2)
        str1 = line1.split()
        str2 = line2.split()
        _boxes1 = np.array(list(map(float, str1[1:])))
        _boxes2 = np.array(list(map(float, str2[1:])))
        #print(_boxes1,_boxes2)
        boxes1 = np.split(_boxes1, len(_boxes1)//5)
        tag_boxes1 = []
        for box in boxes1:
            #print(box)
            # cx cy w h
            x1,x2,y1,y2 = box[1]-box[3]/2, box[1]+box[3]/2, box[2]-box[4]/2, box[2]+box[4]/2
            tag_boxes1.append([int(box[0]),int(x1),int(y1),int(x2),int(y2)])


        boxes2 = np.split(_boxes2, len(_boxes2)//5)
        boxes1 = np.array(boxes1)
        tag_boxes1 = np.array(tag_boxes1)
        boxes2 = np.array(boxes2)
        # print("boxes1:", boxes1)
        # print("tag:", tag_boxes1)
        # print("boxes2:", boxes2)

        #mask = (boxes1[...,0]==0)
        for box in boxes1:
            hx[int(box[0]),2] +=1

        for box in boxes2:

            #print(tool.iou(box,boxes2))
            t = tool.iou(box,boxes2)
            #print(np.argmax(t))
            max_iou_idx = np.argmax(t)
            #print(tag_boxes1[np.argmax(t)])
            #print(box)
            if t[max_iou_idx]>thresh and tag_boxes1[np.argmax(t),0]==box[0]:
                hx[int(box[0]), 0] += 1
                #print("TP+1")
            else:
                hx[int(box[0]), 1] += 1
                #print("FP+1")
    print("threshold:",thresh)
    print("P:",hx[:,0]/(hx[:,0]+hx[:,1]))
    print("R:", hx[:, 0] / hx[:,2])
        #print(hx)

