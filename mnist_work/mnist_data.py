import torch, os, cv2
from torch.utils.data import Dataset
import numpy as np

# root : 存放MNIST的路径
class MNISTDataset(Dataset):

    def __init__(self, root, is_train=True):
        self.dataset = []  # 记录所有数据
        sub_dir = "TRAIN" if is_train else "TEST"
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir = f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                self.dataset.append((img_path, tag))

    # 数据集有多少数据
    def __len__(self):
        return len(self.dataset)

    # 每条数据的处理方式
    def __getitem__(self, index):
        data = self.dataset[index]

        img_data = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)
        img_data = img_data.reshape(-1) #数据展平
        img_data = img_data / 255 #归一化

        #one_hot
        tag_one_hot = np.zeros(10)
        tag_one_hot[int(data[1])] = 1

        return np.float32(img_data), np.float32(tag_one_hot)


if __name__ == '__main__':
    dataset = MNISTDataset("../data/MNIST_IMG")
    # print(len(dataset))
    print(dataset[30000])
