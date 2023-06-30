import pandas as pd
from torch.utils import data
import numpy as np
import os
from PIL import Image, ImageFile
from torchvision import transforms
import torch
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.size = [512, 512]
        self.new_path = "./data/new_test"
        self.transform = transforms.Compose([
            transforms.CenterCrop(2560),
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        image_name = self.img_name[index] + '.pt'
        path = os.path.join(self.root, image_name)
        img = torch.load(path)
        label = self.label[index]
        # img = Image.open(path)
        # img = self.transform(img)
        return img, label

    def save_tensor(self, index):
        image_name = self.img_name[index] + '.jpeg'
        path = os.path.join(self.root, image_name)
        img = Image.open(path)
        img = self.transform(img)
        new_path = os.path.join(self.new_path, self.img_name[index] + '.pt')
        torch.save(img, new_path)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# test_data = RetinopathyLoader(root='./data/test', mode='test')
# for i in tqdm(range(len(test_data))):
#     test_data.save_tensor(i)
# train_data = RetinopathyLoader(root='./data/train', mode='train')
# for i in tqdm(range(len(train_data))):
#     train_data.save_tensor(i)