from PIL import Image
import os
import numpy as np
import utils

class GanData:
    def __init__(self,root,img_size):
        self.img_size = img_size

        self.positive_image_dir = f"{root}/{img_size}/positive"
        self.negative_image_dir = f"{root}/{img_size}/positive"
        self.part_image_dir = f"{root}/{img_size}/part"

        self.positive_label = f"{root}/{img_size}/positive.txt"
        self.negative_label = f"{root}/{img_size}/negative.txt"
        self.part_label = f"{root}/{img_size}/part.txt"

        if not os.path.exists(self.positive_image_dir):
            os.makedirs(self.positive_image_dir)
        if not os.path.exists(self.positive_image_dir):
            os.makedirs(self.positive_image_dir)
        if not os.path.exists(self.positive_image_dir):
            os.makedirs(self.positive_image_dir)

        #文件路径
        self.anno_box_path = r"D:/AIstudyCode/data/CelebA/Anno/list_bbox_celeba.txt"
        self.anno_landmark_path = r"D:/AIstudyCode/data/CelebA/Anno/list_landmarks_celeba.txt"
        self.img_path = r"D:/AIstudyCode/data/CelebA/Img/img_celeba.7z/img_celeba.7z/img_celeba"

    def run(self,epoch):

        positive_label_txt = open(self.positive_label,"w")
        negative_label_txt = open(self.negative_label,"w")
        part_label_txt = open(self.part_label,"w")

        # count 限制生产数据的比例 正/负/部分 =  1：1：3
        positive_count = 0
        negative_count = 0
        part_count = 0

        box_file = open(self.anno_box_path,"r")
        landmark_file = open(self.anno_landmark_path,"r")
        i = 0

        for _ in range(epoch):
            for line1,line2 in zip(box_file,landmark_file):
                if i < 2:
                    i+=1
                    continue
                print(line1)
                print(line2)
                img_strs = line1.split()
                img_file = f"{self.img_path}/{img_strs[0]}"
                img = Image.open(img_file)

                landmark_strs = line2.split()
                x,y,w,h = int(img_strs[1]),int(img_strs[2]),int(img_strs[3]),int(img_strs[4])

                px1,py1 = int(landmark_strs[1]),int(landmark_strs[2])
                px2, py2 = int(landmark_strs[3]), int(landmark_strs[4])
                px3, py3 = int(landmark_strs[5]), int(landmark_strs[6])
                px4, py4 = int(landmark_strs[7]), int(landmark_strs[8])
                px5, py5 = int(landmark_strs[9]), int(landmark_strs[10])

                if max(w,h) < 40 or x < 0 or y < 0 or w < 0 or h < 0: continue

                # 适当做矫正
                x1, y1 = int(x + w * 0.12), int(y + h * 0.1)
                x2, y2 = int(x + w * 0.9), int(y + h * 0.85)
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                # 正样本
                cx, cy = int(x + w / 2), int(y + h / 2)
                _cx, _cy = cx + np.random.randint(-w * 0.12, w * 0.12), cy + np.random.randint(-h * 0.12, h * 0.12)
                _w, _h = w + np.random.randint(-w * 0.12, w * 0.12), h + np.random.randint(-h * 0.12, h * 0.12)

                _x1, _y1, _x2, _y2 = int(_cx - _w / 2), int(_cy - _h / 2), int(_cx + _w / 2), int(_cy + _h / 2)

                _x1_off, _y1_off, _x2_off, _y2_off = (x1 - _x1) / _w, (y1 - _y1) / _h, (x2 - _x2) / _w, (y2 - _y2) / _h

                _px1, _py1 = (px1 - _x1) / _w, (py1 - _y1) / _h
                _px2, _py2 = (px2 - _x1) / _w, (py2 - _y1) / _h
                _px3, _py3 = (px3 - _x1) / _w, (py3 - _y1) / _h
                _px4, _py4 = (px4 - _x1) / _w, (py4 - _y1) / _h
                _px5, _py5 = (px5 - _x1) / _w, (py5 - _y1) / _h

                # 裁剪偏移后的照片框
                clip_img = img.crop([_x1, _y1, _x2, _y2])
                clip_img = clip_img.resize((self.img_size, self.img_size))
                # clip_img.show()

                iou = utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))
                if iou > 0.65:
                    if positive_count <= 40000:
                        clip_img.save(f"{self.positive_image_dir}/{positive_count}.jpg")

                        positive_label_txt.write(
                            f"{positive_count}.jpg 1 {_x1_off} {_y1_off} {_x2_off} {_y2_off} {_px1} {_py1} {_px2} {_py2} {_px3} {_py3} {_px4} {_py4} {_px5} {_py5}\n")
                        positive_label_txt.flush()
                        positive_count += 1
                elif iou > 0.4:
                    if part_count <= 40000:
                        clip_img.save(f"{self.part_image_dir}/{part_count}.jpg")

                        part_label_txt.write(
                            f"{part_count}.jpg 2 {_x1_off} {_y1_off} {_x2_off} {_y2_off} {_px1} {_py1} {_px2} {_py2} {_px3} {_py3} {_px4} {_py4} {_px5} {_py5}\n")
                        part_label_txt.flush()
                        part_count += 1
                elif iou < 0.4:
                    if negative_count <= 80000:
                        clip_img.save(f"{self.negative_image_dir}/{negative_count}.jpg")

                        negative_label_txt.write(f"{negative_count}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
                        negative_label_txt.flush()
                        negative_count += 1

                # 生成负样本
                img_w, img_h = img.size
                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0 or min(img_w, img_h) / 2 <= 50:
                    continue
                _x1, _y1 = np.random.randint(1, img_w - 1), np.random.randint(1, img_h - 1),
                _w, _h = np.random.randint(1, img_w - _x1), np.random.randint(1, img_h - _y1)
                _x2, _y2 = _x1 + _w, _y1 + _h
                clip_img = img.crop([_x1, _y1, _x2, _y2])
                clip_img = clip_img.resize((self.img_size, self.img_size))

                _x1_off, _y1_off, _x2_off, _y2_off = (x1 - _x1) / _w, (y1 - _y1) / _h, (x2 - _x2) / _w, (
                            y2 - _y2) / _h

                _px1, _py1 = (px1 - _x1) / _w, (py1 - _y1) / _h
                _px2, _py2 = (px2 - _x1) / _w, (py2 - _y1) / _h
                _px3, _py3 = (px3 - _x1) / _w, (py3 - _y1) / _h
                _px4, _py4 = (px4 - _x1) / _w, (py4 - _y1) / _h
                _px5, _py5 = (px5 - _x1) / _w, (py5 - _y1) / _h

                # 裁剪负样本照片框
                clip_img = img.crop([_x1, _y1, _x2, _y2])
                clip_img = clip_img.resize((self.img_size, self.img_size))
                # clip_img.show()

                iou = utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))
                if iou > 0.65:
                    if positive_count <= 40000:
                        clip_img.save(f"{self.positive_image_dir}/{positive_count}.jpg")

                        positive_label_txt.write(
                            f"{positive_count}.jpg 1 {_x1_off} {_y1_off} {_x2_off} {_y2_off} {_px1} {_py1} {_px2} {_py2} {_px3} {_py3} {_px4} {_py4} {_px5} {_py5}\n")
                        positive_label_txt.flush()
                        positive_count += 1
                elif iou > 0.4:
                    if part_count <= 40000:
                        clip_img.save(f"{self.part_image_dir}/{part_count}.jpg")

                        part_label_txt.write(
                            f"{part_count}.jpg 2 {_x1_off} {_y1_off} {_x2_off} {_y2_off} {_px1} {_py1} {_px2} {_py2} {_px3} {_py3} {_px4} {_py4} {_px5} {_py5}\n")
                        part_label_txt.flush()
                        part_count += 1
                elif iou < 0.4:
                    if negative_count <= 80000:
                        clip_img.save(f"{self.negative_image_dir}/{negative_count}.jpg")

                        negative_label_txt.write(f"{negative_count}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
                        negative_label_txt.flush()
                        negative_count += 1

            positive_label_txt.close()
            negative_label_txt.close()
            part_label_txt.close()

if __name__ == '__main__':
    genData = GanData(r"D:/AIstudyCode/MTCNN_new", 12)
    genData.run(100)

