import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os
from torchvision.utils import save_image

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])

class makeData(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        black_jpg = torchvision.transforms.ToPILImage()(torch.zeros(3,256,256))
        black_png = torchvision.transforms.ToPILImage()(torch.zeros(3,256,256))

        namepng = self.name[index]
        namejpg = namepng[:-3] + 'jpg'

        img_jpg_path = os.path.join(self.path,'JPEGImages')
        img_png_path = os.path.join(self.path,'SegmentationClass')
        img_jpg = Image.open(os.path.join(img_jpg_path, namejpg))
        img_png = Image.open(os.path.join(img_png_path, namepng))

        img_size = torch.Tensor(img_jpg.size)
        l_max_index = img_size.argmax()
        ratio = 256/img_size[l_max_index.item()]
        img_re2size = img_size * ratio
        img_jpg_use = img_jpg.resize(img_re2size)
        img_png_use = img_png.resize(img_re2size)

        w,h = img_re2size.tolist()
        black_jpg.paste(img_jpg_use, (0, 0, int(w), int(h)))
        black_png.paste(img_png_use, (0, 0, int(w), int(h)))

        return transform(black_jpg), transform(black_png)

if __name__ == '__main__':
    i = 1
    dataset = makeData(r"D:/AIstudyCode/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/")
    for a,b in dataset:
        print(i)
        print(a.shape)
        print(b.shape)
        save_image(a, "D:/AIstudyCode/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/train/jpg/{0}.jpg".format(i),nrow=1)
        save_image(b, "D:/AIstudyCode/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/train/png/{0}.png".format(i),nrow=1)
        i += 1
