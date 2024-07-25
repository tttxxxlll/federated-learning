import os
import pandas as pd
from torchvision import datasets, transforms
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        """
            csv_file: 标签文件的路径.
            root_dir: 所有图片的路径.
            transform: 一系列transform操作
        """
        self.data_frame = pd.read_csv(csv_file)
        # self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)  # 返回数据集长度

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]  # 获取图片所在路径
        img = Image.open(img_path).convert('RGB')  # 防止有些图片是RGBA格式

        label_number = self.data_frame.iloc[idx, 1]  # 获取图片的类别标签
        img = img.crop((0, 80, 320, 140))
        if self.transform:
            img = self.transform(img)

        return img, label_number  # 返回图片和标签


custom_dataset = MyDataset(csv_file='D:/carladata/Learn-Carla-main/outputs/output1/dataset.csv',
                           transform=transforms.Compose([
                               transforms.Resize([128, 128]),
                               # transforms.ColorJitter(brightness=[0.1, 1]),
                               # transforms.RandomHorizontalFlip(0.5),# 水平翻转
                               transforms.ToTensor(),
                               # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ])
                           )



print(len(custom_dataset))
train_size = int(len(custom_dataset) * 5/6)
test_size = int(len(custom_dataset) /6)

train_dataset,  test_dataset = torch.utils.data.random_split(custom_dataset, [train_size,  test_size])
print(len(train_dataset),len(test_dataset))