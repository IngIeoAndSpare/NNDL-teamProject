# image reader
from PIL import Image

## user custom dataset utils lib
from torch.utils.data import Dataset, DataLoader

## utils lib
import os
import os.path
from os import listdir
from os.path import isfile, join

class CustomDataset(Dataset):
    
    def init_image_data(self):
        ## get image file list and append dataset
        image_dataset = {}

        ## XXX: ori and binary file is same name and length. so, use ori file name. 
        for label_name in self.label_array:
            image_dataset[label_name] = self.get_filelist(os.path.join(self.source_path, label_name, 'ori'))

        return image_dataset

    def get_filelist(self, file_path):
        ## get file path list
        if os.path.exists(file_path):
            return [
                file_name for file_name in listdir(file_path) if isfile(join(file_path, file_name))
            ]
        else:
            return None

    def label_convert(self):
        label_dict = {}
        for index, label_item in enumerate(self.label_array):
            label_dict[label_item] = index

        return label_dict

    def convert_data(self):
        all_labels = []
        all_oimg_file_path = []
        all_bimg_file_path = []
        length = 0
        
        for label_item in self.label_array:
            ## self.data_image_path => {'line': [image fileName...], 'square': ...}
            if(self.data_image_path[label_item] != None):
                for image_path in self.data_image_path[label_item]:
                    all_labels.append(self.label_dict[label_item])
                    all_oimg_file_path.append(
                        os.path.join(self.source_path, label_item, 'ori', image_path)
                    )
                    all_bimg_file_path.append(
                        os.path.join(self.source_path, label_item, 'binary', image_path)
                    )
                    length += 1
        return all_labels, all_oimg_file_path, all_bimg_file_path, length
    
    def __init__(self, labels, data_source_path, data_chanels, transforms=None):
        self.label_array = labels
        self.source_path = data_source_path
        self.transforms = transforms
        self.data_image_path = self.init_image_data()
        
        ## image data init
        self.label_dict = self.label_convert()
        self.all_label_array, self.all_oimage_array, self.all_bimage_array, self.length = self.convert_data()
        
        self.data_chanels = data_chanels
        self.num_classes = len(labels)


    def __getitem__(self, index):
        try:
            ori_img = Image.open(self.all_oimage_array[index]).convert(self.data_chanels)
        except OSError :
            print(f'image truncated path => {self.all_oimage_array[index]}')
            return { 'ori_image' : None, 'label' : None }

        ## ori_img = ori_img.convert("RGBA")

        ## binary_img = Image.open(self.all_bimage_array[index])
        ## binary_img = binary_img.convert("RGB")

        if self.transforms is not None:
            ori_img = self.transforms(ori_img)
            ## binary_img = self.transforms(binary_img)

        return { 'ori_image' : ori_img, 'label' : self.all_label_array[index] }
        ## return { 'ori_image' : ori_img, 'binary_image' : binary_img , 'label' : self.all_label_array[index] }

    def __len__(self):
        return self.length