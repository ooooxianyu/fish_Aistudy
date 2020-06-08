from torch.utils.data import Dataset
import torch
from torch import nn
import os
from PIL import Image
from torchvision import transforms
import numpy as np

tf = transforms.ToTensor()


class MyDataset(Dataset):

    def __init__(self, root):
        self.dataset = os.listdir(root)
        self.root = root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fn = self.dataset[index]
        strs = fn.split(".")[0]
        lables = np.array([int(x) for x in strs])
        img_data = tf(Image.open(f"{self.root}/{fn}"))
        return img_data, lables


if __name__ == '__main__':
    myDataset = MyDataset("../../code/train")
    print(myDataset[0][1].shape)