from torch.utils.data import Dataset
import torchaudio

import numpy as np
from torch.nn import functional as F

def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.max()


tf = torchaudio.transforms.MFCC(sample_rate=8000)


class MyDataset(Dataset):

    def __init__(self, root):
        self.train_data = np.loadtxt("train_total_data.txt",dtype=np.str)
        self.root = root


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        fn = self.train_data[index]
        strs = fn[0]
        lables = fn[1]

        filename = self.root + strs

        waveform, sample_rate = torchaudio.load(filename)
        specgram = normalize(tf(waveform))
        specgram = F.adaptive_avg_pool2d(specgram, (32, 256))

        return specgram, np.array([lables],dtype=int)


if __name__ == '__main__':
    myDataset = MyDataset("D:/AIstudyCode/data/audio_data/speech_commands_train_set_v0.02/")
    print(myDataset[1500])