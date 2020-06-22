from model import yoloV3_net
import cfg
import torch
import numpy as np
import PIL.Image as pimg
import PIL.ImageDraw as draw
import seaborn as sns
import tool
import cv2
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #是否有cuda
_device = torch.device("cpu" if torch.cuda.is_available() else "cuda")

class Detector(torch.nn.Module):
    def __init__(self, save_path):
        super(Detector, self).__init__()

        self.net = yoloV3_net().to(device) # 加载网络
        self.net.load_state_dict(torch.load(save_path)) # 加载权重
        self.net.eval() # 标记这里是测试

    def forward(self, input, thresh, anchors):
        # 输入416*416 输出不同尺寸下每个格子的数据 F*F*3*（5+c）
        output_13, output_26, output_52 = self.net(input)

        #print(output_13.shape)
        # 根据置信度筛选boxes 返回索引位置和筛选后的boxes信息
        idxs_13, vecs_13 = self._filter(output_13.to(_device), thresh)
        # 反算box在原照片中的位置
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])
        idxs_26, vecs_26 = self._filter(output_26.to(_device), thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])
        idxs_52, vecs_52 = self._filter(output_52.to(_device), thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])
        # 返回最终得到的box信息
        # print(boxes_13.shape)
        # print(boxes_26.shape)
        # print(boxes_52.shape)
        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0,2,3,1) # N 3*(5+C) F F -> N F F 3*(5+C)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1) # N F F 3*(5+C) -> N F F 3 5+C
        #print("1", output[:, 0])
        torch.sigmoid_(output[..., 0])
        #print("2",output[:, 0])

        mask = output[..., 0]>thresh # 5+C = 置信度 + cx cy w h 第0位即为置信度
        #print(output[:, 0])
        idxs = mask.nonzero() # 得到筛选后的位置索引
        #print("idxs",idxs.shape)
        vecs = output[mask] # 得到筛选后的boxes信息
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)
        a = idxs[:,3] # 三个anchor框
        confidence = vecs[:, 0]
        _classify = vecs[:, 5:]
        if len(_classify) == 0:
            classify = torch.Tensor([])
        else:
            classify = torch.argmax(_classify, dim=1).float() # one-hot -> 类别

        # 反算过程
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t
        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = x1 + w
        y2 = y1 + h
        out = torch.stack([confidence, x1, y1, x2, y2, classify],dim=1)
        return out
if __name__ == '__main__':
    save_path = "data/checkpoints/myyolo.pt"
    detector = Detector(save_path)
    #img_1 = pimg.open(r"000000000036.jpg")
    img_1= cv2.imread("40.jpg", 1)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    h,w = img_1.shape[0], img_1.shape[1]
    print(w,h)
    img_2, nh, nw ,top, left= tool.cv2_letterbox_image(img_1,416)
    #img_1 = torch.Tensor(img_1)

    print(nh, nw, top, left)

    img_1 = pimg.fromarray(img_1.astype('uint8')).convert('RGB')
    img = np.array(img_2) / 255
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    #print(img.shape)
    #exit()
    img = img.cuda()

    out_value = detector(img, 0.6, cfg.ANCHORS_GROUP)
    out_value = out_value.detach().to(_device)
    #print(out_value.shape)

    palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    classify_text = ["person","motor","plane","car","dog"]
    #print(palette)
    #exit()
    boxes = []
    for j in range(10):
        classify_mask = (out_value[..., -1] == j)
        #print(out_value[..., -1])
        _boxes = out_value[classify_mask]
        boxes.append(tool.nms(_boxes, 0.5))
        #boxes.append(_boxes)
    #print(boxes)
    #img_draw = draw.ImageDraw(img_1)

    for box in boxes:
        try:
            img_draw = draw.ImageDraw(img_1)
            for _box in box:
                c, x1, y1, x2, y2, classify = _box[0:]

                s_x1, s_x2 = (x1-left)/nw , (x2-left)/nw
                s_y1, s_y2 = (y1-top)/nh , (y2-top)/nh
                x1, x2, y1, y2 = s_x1*w, s_x2*w, s_y1*h, s_y2*h

                img_draw.rectangle((x1, y1, x2, y2),outline=palette[int(classify)])
                img_draw.text((x1, y1-20), classify_text[int(classify)], fill=palette[int(classify)])
        except:
            continue
    img_1.show()

