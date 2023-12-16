import numpy as np
import random
import torch
import os
from PIL import Image
from torch.utils.data import Dataset

lable_dic = {'maze': 0, 'icedid': 1, 'egregor': 2, 'shamoon': 3,'gandcrab': 4}

def binary_2_image(binary, file_name):
    img = Image.fromarray(binary)
    img.save(file_name)
    
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class ImageDataset(Dataset):
    def __init__(self, data_path, df, transform):
        self.df = df
        self.loader = default_image_loader
        self.transform = transform
        self.dir = data_path
        self.label_dic = lable_dic

    def __getitem__(self, index):
        Img = self.df.Img[index]
        image = self.loader(os.path.join(self.dir, Img))
        image = self.transform(image)
        label = self.df.label[index]
        return image, self.label_dic[label]

    def __len__(self):
        return self.df.shape[0]