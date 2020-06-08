from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2

# 数据预处理 可以添加数据增强操作
tf = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,root,img_size):
        self.dataset = []

        self.img_root_dir = f"{root}/{img_size}"

        with open(f"{self.img_root_dir}/positive.txt","r") as f:
            self.dataset.extend(f.readlines())
        with open(f"{self.img_root_dir}/negative.txt","r") as f:
            self.dataset.extend(f.readlines())
        with open(f"{self.img_root_dir}/part.txt","r") as f:
            self.dataset.extend(f.readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        strs = data.split()

        img_path = None

        if strs[1] == "1":
            img_path = f"{self.img_root_dir}/positive/{strs[0]}"
        elif strs[1] == "2":
            img_path = f"{self.img_root_dir}/negative/{strs[0]}"
        else:
            img_path = f"{self.img_root_dir}/part/{strs[0]}"

        img_data = cv2.imread(img_path,1)
        img_data = tf(img_data)

        c,x1,y1,x2,y2 = float(strs[1]),float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])

        px1, py1 = float(strs[6]), float(strs[7])
        px2, py2 = float(strs[8]), float(strs[9])
        px3, py3 = float(strs[10]), float(strs[11])
        px4, py4 = float(strs[12]), float(strs[13])
        px5, py5 = float(strs[14]), float(strs[15])

        return img_data, np.array([c, x1, y1, x2, y2, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5],
                                  dtype=np.float32)

if __name__ == '__main__':
    dataset = MyDataset(r"D:/AIstudyCode/MTCNN_new",48)
    print(dataset[0])
