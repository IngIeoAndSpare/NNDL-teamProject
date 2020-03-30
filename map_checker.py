# image reader
from PIL import Image
import matplotlib.pyplot as plt

## torch module
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision
import torchvision.models as torch_models

## user custom dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

## file metadata module
import os
import os.path
from os import listdir
from os.path import isfile, join

# trainning dashboard
# import tensorwatch as tw
# import time

## file path
DATA_FOLDER_PATH = r"..\\"
RESULT_FILE_PATH = os.path.join(DATA_FOLDER_PATH, "network_result")

## train var
TRAIN_DATA_LABEL = ['line', 'square', 'unspecified_shapes', 'dispersion', 'normal']
TRAIN_DATA_VENDER = ['binary', 'ori']
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCH = 256
TRAIN_LEARINING_LATE = 0.0001

# input file setting
FILE_INCORDING_CHANNELS = "RGB"



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
    
    def __init__(self, labels, data_source_path, transforms=None):
        self.label_array = labels
        self.transforms = transforms
        self.source_path = data_source_path
        self.data_image_path = self.init_image_data()
        
        ## image data init
        self.label_dict = self.label_convert()
        self.all_label_array, self.all_oimage_array, self.all_bimage_array, self.length = self.convert_data()
        
        self.num_classes = len(labels)

    def __getitem__(self, index):
        try:
            ori_img = Image.open(self.all_oimage_array[index]).convert(FILE_INCORDING_CHANNELS)
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


class CNN_network(nn.Module):

    def __init__(self, num_class):
        super(CNN_network, self).__init__()
        
        ## network
        self.start_layer = self.conv_module(4, 16)
        self.layer_2 = self.conv_module(16, 32)
        self.layer_3 = self.conv_module(32, 64)
        self.layer_4 = self.conv_module(64, 128)
        self.layer_5 = self.conv_module(128, 256)
        self.last_layer = self.global_avg_pool(256, num_class)
        self.num_class = num_class

    def forward(self, x):
        ##network forward
        out = self.start_layer(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.last_layer(out)
        out = out.view(-1, self.num_class)

        return out

    def conv_module(self, in_num, out_num):
        ## set conv2d layer
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=2, stride=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=2, stride=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )


def get_dataset(label, file_path, transform):
    return CustomDataset(
        label, file_path, transforms = transform
    )

def get_model_output(model, item_set, device):
    ori_images = item_set['ori_image'].to(device)
    ##binary_image = item_set['binary_image'].to(device)
    labels = item_set['label'].to(device)
    
    ## network pass  --> multi file input?
    return labels, model(ori_images)

def get_network_model(model_name, class_num, device):
    return {
        'CNN' : CNN_network(num_class = class_num).to(device),
        'RES18' : torch_models.resnet18().to(device),
        'RES34' : torch_models.resnet34().to(device),
        'RES50' : torch_models.resnet50().to(device),
        'RES101' : torch_models.resnet101().to(device),
        'GoogleNet' : torch_models.googlenet().to(device),
        'AlexNet' : torch_models.alexnet().to(device),
        'ShuffleNet' : torch_models.shufflenet_v2_x1_0().to(device)
        ##'Res' : Res_network(num_class = class_num).to(device)
    }.get(model_name, None)


if __name__ == "__main__":
    #init_transforms

    transforms_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(10.),
            transforms.ToTensor()
        ]
    )

    transforms_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]
    )

    #init_train_data()
    train_data = get_dataset(
        TRAIN_DATA_LABEL,
        os.path.join(DATA_FOLDER_PATH, 'train_data'),
        transforms_train
    )
    
    train_data_loader = DataLoader(train_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)

    test_data = get_dataset(
        TRAIN_DATA_LABEL,
        os.path.join(DATA_FOLDER_PATH, 'test_data'),
        transforms_test
    )
    test_data_loader = DataLoader(test_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)

    ## device, network init
    device_name = 'ShuffleNet'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network_model = get_network_model(device_name, len(TRAIN_DATA_LABEL), device)

    # summaray
    writer = SummaryWriter("{}_{}".format(device_name, TRAIN_EPOCH))

    print(torch.cuda.device_count())

    ## load train result tr file 
    pt_filepath = os.path.join(RESULT_FILE_PATH, "{}.pt".format(device_name))
    if os.path.isfile(pt_filepath):
        network_model.load_state_dict(torch.load(pt_filepath))
        network_model.eval()

    ## update var
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network_model.parameters(), lr=TRAIN_LEARINING_LATE)

    ##tw.draw_model(network_model, [1,3,224,224])

    step = 1
    
    for epoch_count in range(TRAIN_EPOCH):
        for batch_size, data_set in enumerate(train_data_loader):
            if data_set['ori_image'] is not None:
                labels, outputs = get_model_output(network_model, data_set, device)
                
                '''
                if device_name == 'GoogleNet':
                    ## XXX : https://discuss.pytorch.org/t/question-about-googlenet/44896/5
                    outputs = outputs.logits
                '''
                
                loss = criterion(outputs, labels)
 
                ## update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_size + 1) % TRAIN_BATCH_SIZE == 0:
                    print(f'[Epoch {epoch_count} / {TRAIN_EPOCH}], Loss : [{loss.item():.4f}]')
                    writer.add_scalar('Train/loss', loss.item(), step)
                    torch.save(network_model.state_dict(), pt_filepath)
                    step += 1
            else :
                continue


    network_model.eval()

    correct = 0
    total = 0
    answer_ratio = 0
    
    for data_set in test_data_loader:
        if data_set['ori_image'] is not None:
            labels, outputs = get_model_output(network_model, data_set, device)

            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
            answer_ratio = 100 * correct / total
            print(f'Test Accuracy of the model on the {total} test images: {answer_ratio} %')
            writer.add_scalar('Test/answer', answer_ratio, total)
        else :
            continue