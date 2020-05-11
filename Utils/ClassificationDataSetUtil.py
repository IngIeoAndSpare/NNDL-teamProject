# image reader
from PIL import Image

## user custom dataset utils lib
from torch.utils.data import Dataset, DataLoader

## utils lib
import os
import os.path
from os import listdir
from os.path import isfile, join

class ClassificationDataSet(Dataset):

    def _init_image_data(self):
        return  [
            os.path.join(self.source_path, file_name) 
            for file_name in listdir(self.source_path) 
            if isfile(join(self.source_path, file_name))
        ]

    def __init__(self, data_source_path, data_chanels, transforms=None):
        self.source_path = data_source_path
        self.transforms = transforms
        self.data_chanels = data_chanels
        self.data_image_path_list = self._init_image_data()

    def __getitem__(self, index):
        ori_img = Image.open(self.data_image_path_list[index]).convert(self.data_chanels)            

        if self.transforms is not None:
            ori_img = self.transforms(ori_img)

        return {"image" : ori_img, "file_name" : os.path.basename(self.data_image_path_list[index])}
    
    def __len__(self):
        return len(self.data_image_path_list)