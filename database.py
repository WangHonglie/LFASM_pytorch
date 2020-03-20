from os import listdir
from os.path import join
from random import random
from PIL import Image, ImageDraw, ImageOps
import json
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class AllData(data.Dataset):
    def __init__(self, root, batchSize, transform=None):
        super().__init__()
        self.batchSize = batchSize
        self.samples = []
        # print(self.samples[0])
        with open(root) as f:
            for line in f:
                infos = line.split(',')
                self.samples.append(infos)

        self.transformer = transforms.Compose(transform)


    def __getitem__(self, index):
        img = Image.open(self.samples[index][0])
        brand = int(self.samples[index][1])
        color = int(self.samples[index][2])
        direct = int(self.samples[index][3])
        cltype = int(self.samples[index][4])
        labels = int(self.samples[index][6])

        imgT = self.transformer(img)

        return imgT, labels, color, cltype, brand, direct

    def __len__(self):
        num = len(self.samples) - (len(self.samples)%self.batchSize)
        return num

class SingleData(data.Dataset):
    def __init__(self, root, batchSize, transform=None):
        super().__init__()
        self.batchSize = batchSize
        self.imgs = []
        self.labels = []
        self.cams = []
        count = 0
        # print(self.samples[0])
        with open(root) as f:
            for line in f:
                count += 1
                if count == 64:
                    break
                infos = line.split(',')
                self.imgs.append(infos[1])
                self.labels.append(int(infos[3]))
                self.cams.append(int(infos[4][1:]))
        self.labels_set = set(np.array(self.labels))
        self.labels_to_indices = {label: np.where(np.array(self.labels)==label)[0]  for label in self.labels_set}
        # print(self.labels_to_indices)

        self.transformer=transforms.Compose(transform)

    def __getitem__(self, index):
        img1 = Image.open(self.imgs[index])
        label1 = self.labels[index]
        cam1 = self.cams[index]

        target = np.random.randint(0,2)
        # print(target)
        if target == 1:
            img2 = img1
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.labels_to_indices[label1])
            label2 = label1
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.labels_to_indices[siamese_label])
            label2 = siamese_label
            img2 = self.imgs[siamese_index]
            img2 = Image.open(img2)


        img1 = self.transformer(img1)
        img2 = self.transformer(img2)


        return img1, img2, label1, label2, target

    def __len__(self):
        num = len(self.imgs)
        return num
if __name__ == "__main__":
    transform_val_list = [
            transforms.Resize(size=(234,234),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    query_datasets = SingleData('/home/honglie.whl/ReID/baseline_ReID/data/VeRi/train.csv', 1, transform_val_list)
    loader = DataLoader(dataset=query_datasets,
                        num_workers=0,
                        batch_size=1,
                        shuffle=False)
    for batch in loader:
        img, img2,label, label2, target =batch
        print(img,img2,target)
        exit()
