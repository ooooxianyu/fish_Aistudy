from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class FaceMyData(Dataset):
    def __init__(self,root):
        super().__init__()
        self.root= root
        self.dataset = os.listdir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pic_name = self.dataset[index]
        img_data = cv2.imread(f"{self.root}/{pic_name}")
        img_data = img_data[...,::-1]
        img_data = img_data.transpose([2,0,1])
        img_data = ((img_data/255. -0.5) *  2).astype(np.float32)
        return img_data

if __name__ == '__main__':
    facemydata = FaceMyData(r"D:\AIstudyCode\data\seeprettyface_chs_wanghong\data")
    print(facemydata[0].shape)
