import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
import numpy as np
import cfg
import os
from PIL import Image
import math

LABEL_FILE_PATH = "data/train_label.txt" # 标签文件
IMG_BASE_DIR = "data" # 图片目录

# 图片数据预处理
transfroms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor() #转tensor
])

# 获得onehot编码
def one_hot(cls_num,i):
    b = np.zeros(cls_num)
    b[i] = 1.
    return b

# 数据对象
class MyDataset(Dataset):
    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines() # 逐行读取标签文本
            # img_path 类别1 anchor框4 类别1 anchor框4 类别1 anchor框4…
    def __len__(self):
        return len(self.dataset) # 每行代表一个数据 有多少行就有多少数据

    def __getitem__(self, index):
        labels = {} # 标签 字典 图片path ： 类别
        line = self.dataset[index] # 索引index
        strs = line.split() # 按空格划分每一行
        #_img_data = Image.open(os.path.join(IMG_BASE_DIR,strs[0])) # 路径dir+path
        _img_data = Image.open(os.path.join(strs[0]))
        img_data = transfroms(_img_data) # 数据预处理

        # 将读取得标签 后面类别和置信度从字符格式转换为浮点型 并且用list保存
        #_boxes = np.array(float(x) for x in strs[1:])
        _boxes = np.array(list(map(float,strs[1:])))

        # 例如有三个box  则为 1 *  15（类别1 x1 y1 x2 y2 ……）: 转换成 1 * 3 * 5
        boxes = np.split(_boxes, len(_boxes) // 5) # 将anchor分开

        # ANCHORS_GROUP 特征大小（13，26，52）：3个anchor的尺寸（w，h）
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            # F*F *3*(5+c) 每个格子有三个anchor box c个分类： 置信度1+box位置4+类别数c
            # 初始化标签存储字典为key：list / 且表初始化为零
            labels[feature_size] = np.zeros(shape=(feature_size,feature_size,3,5+cfg.CLASS_NUM))
            for box in boxes:
                cls, cx, cy, w, h = box # 获取标签文本中的真实框

                # modf 分别取出小数部分和整数部分  （实际框的位置等于 特征大小*（索引+偏移量））
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)
                # 取出不同特征大小下 提前规定好的anchor尺寸信息
                for i, anchor in enumerate(anchors):
                    # 当前anchor尺寸下的面积
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    # pw ph 实际框和anchor的比值 再取对数 ： 网络训练得到_pw，_ph -> 通过 exp(_pw)*anchor_w / exp(_ph)*anchor_h 得到真实框
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    _p_w, _p_h = np.log(p_w),np.log(p_h)
                    #实际框的面积
                    p_area = w * h
                    # iou取最小框/最大框 ： 为了使得iou（充当置信度）小于1大于0 又偏向1
                    iou = min(p_area,anchor_area) / max(p_area, anchor_area)
                    # 对标签存储字典进行幅值
                    # 置信度 中心的偏移x y 宽和高的偏移值（相对于anchor） onehot类别
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou,cx_offset,cy_offset,_p_w,_p_h,*one_hot(cfg.CLASS_NUM,int(cls))])
        return labels[13], labels[26], labels[52], img_data

if __name__ == '__main__':
    x = one_hot(10, 2)
    print(x)
    data = MyDataset()
    dataloader = DataLoader(data, 3, shuffle=True)
    # for i,x in enumerate(dataloader):
    #     print(x[0].shape)
    #     print(x[1].shape)
    #     print(x[2].shape)
    #     print(x[3].shape)
    #print(dataloader)
    for target_13, target_26, target_52, img_data in dataloader:
        print(target_13.shape)
        print(target_26.shape)
        print(target_52.shape)
        print(img_data.shape)