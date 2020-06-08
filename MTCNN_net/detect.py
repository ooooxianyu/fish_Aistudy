from MTCNN_net import *
from data import *
from PIL import Image
from PIL import ImageDraw
import utils
import numpy as np

class Detector:

    def __init__(self):
        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load("12/loss/pnet_052.pt"))

        self.rnet = RNet()
        self.rnet.load_state_dict(torch.load("24/loss/rnet_008.pt"))

        self.onet = ONet()
        self.onet.load_state_dict(torch.load("48/loss/onet_002.pt"))

    def __call__(self, img):
        boxes = self.detPnet(img)
        if boxes is None: return np.array([])

        # return boxes

        boxes = self.detRnet(img)
        if boxes is None: return np.array([])

        # return boxes

        boxes = self.detOnet(img)
        if boxes is None: return np.array([])

        return boxes

    def detPnet(self, img):

        w, h = img.size
        scale = 1.
        img_scale = img

        min_side = min(w,h)

        _boxes = []
        while min_side>12:
            _img_scale = tf(img_scale)
            y = self.pnet(_img_scale[None,...])
            y = y.cpu().detach()

            torch.sigmoid_(y[:,0,...])

            c = y[0,0]
            c_mask = c>0.60
            strat_index = c_mask.nonzero()

            _x1 = (strat_index[:,1].float() * 2) / scale
            _y1 = (strat_index[:,0].float() * 2) / scale
            _x2 = (strat_index[:,1].float() * 2) / scale
            _y2 = (strat_index[:,0].float() * 2) / scale

            ow = _x2 - _x1
            oh = _y2 - _y1

            p = y[0,1:,c_mask]

            x1 = _x1 + ow * p[0, :]
            y1 = _y1 + oh * p[1, :]
            x2 = _x2 + ow * p[3, :]
            y2 = _y2 + oh * p[4, :]

            cc = y[0, 0, c_mask]
            # print(x1,y1,x2,y2)
            # print(cc)
            px1 = (_x1 + p[4, :] * ow)
            py1 = (_y1 + p[5, :] * oh)
            px2 = (_x1 + p[6, :] * ow)
            py2 = (_y1 + p[7, :] * oh)
            px3 = (_x1 + p[8, :] * ow)
            py3 = (_y1 + p[9, :] * oh)
            px4 = (_x1 + p[10, :] * ow)
            py4 = (_y1 + p[11, :] * oh)
            px5 = (_x1 + p[12, :] * ow)
            py5 = (_y1 + p[13, :] * oh)

            _boxes.append(torch.stack([x1, y1, x2, y2, cc, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5]))

            # 图像金字塔
            scale = 0.7
            _w, _h = int(w * scale), int(h * scale)
            img_scale = img_scale.resize((_w, _h))
            min_side = min(_w, _h)

        boxes = torch.cat(_boxes, dim=0)
        return utils.nms(boxes.cpu().detach().numpy(), 0.6)


    def detRnet(self, img, boxes):

        _boxes = self._ro_net(img, boxes, 24)

        return utils.nms(_boxes, 0.65)

    def detOnet(self, img, boxes):

        _boxes = self._ro_net(img,boxes,48)

        _boxes = utils.nms(_boxes, 0.7)

        _boxes = utils.nms(_boxes, 0.3, is_min=True)

        return _boxes

    def _ro_net(self, img, boxes, s):
        imgs = []
        for box in boxes:
            _img = img.crop(box[0:4])
            _img = _img.resize((s,s))
            imgs.append(tf(_img))
        _imgs = torch.stack(imgs, dim=0)

        if s == 24:
            y = self.rnet(_imgs)
        elif s == 48:
            y = self.onet(_imgs)

        y = y.cpu().detach()
        torch.sigmoid_(y[:, 0])
        y = y.numpy()

        if s == 24:
            threshold = 0.7
        elif s == 48:
            threshold = 0.9

        c_mask = y[:, 0] > threshold
        _boxes = boxes[c_mask]

        _y = y[c_mask]

        _w, _h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
        x1, y1 = _boxes[:, 0] + _y[:, 1] * _w, _boxes[:, 1] + _y[:, 2] * _h
        x2, y2 = _boxes[:, 2] + _y[:, 3] * _w, _boxes[:, 3] + _y[:, 4] * _h
        cc = _y[:, 0]

        px1, py1 = _boxes[:, 5] + _y[:, 1] * _w, _boxes[:, 6] + _y[:, 2] * _h
        px2, py2 = _boxes[:, 7] + _y[:, 1] * _w, _boxes[:, 8] + _y[:, 2] * _h
        px3, py3 = _boxes[:, 9] + _y[:, 1] * _w, _boxes[:, 10] + _y[:, 2] * _h
        px4, py4 = _boxes[:, 11] + _y[:, 1] * _w, _boxes[:, 12] + _y[:, 2] * _h
        px5, py5 = _boxes[:, 13] + _y[:, 1] * _w, _boxes[:, 14] + _y[:, 2] * _h

        _boxes = np.stack([x1, y1, x2, y2, cc, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5], axis=1)

        return _boxes

if __name__ == '__main__':
    img = Image.open("1234.jpg")
    detector = Detector()
    boxes = detector(img)
    #print(boxes[:, 0:5])
    drawImg = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        drawImg.rectangle((box[0], box[1], box[2], box[3]), None, 'red')
        drawImg.ellipse((box[5] - 2, box[6] - 2, box[5] + 2, box[6] + 2), 'red')
        drawImg.ellipse((box[7] - 2, box[8] - 2, box[7] + 2, box[8] + 2), 'red')
        drawImg.ellipse((box[9] - 2, box[10] - 2, box[9] + 2, box[10] + 2), 'red')
        drawImg.ellipse((box[11] - 2, box[12] - 2, box[11] + 2, box[12] + 2), 'red')
        drawImg.ellipse((box[13] - 2, box[14] - 2, box[13] + 2, box[14] + 2), 'red')
    img.show()