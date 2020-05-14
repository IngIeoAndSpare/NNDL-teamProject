## image module
from PIL import Image

## classification module
import numpy as np

## torch module
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import torchvision.models as torch_models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

## File metadata module
import os
import os.path

## Dataloader utils module
import sys
sys.path.append("..")
from Utils.CustomDataSetUtil import CustomDataset
from Utils.ClassificationDataSetUtil import ClassificationDataSet

## Netwrok utils module
from Network.ConvNetworkpy import ConvNetwork
from Network.ResNetwork import ResNet

class ClassificationModule:

    def __init__(self, train_path, test_path, validation_path, network_result_path, network_name, tr_batch_size = 32, tr_epoch = 512, tr_rate = 0.0001):
        
        ## File path params
        self.path_train = train_path
        self.path_test = test_path
        self.path_validation = validation_path
        self.path_network_result = network_result_path

        ## Train params
        self.tr_batch_size = tr_batch_size
        self.tr_epoch = tr_epoch
        self.tr_rate = tr_rate
        self.tr_image_chanels = "RGBA"
        self.tr_network_name = network_name

        ## Classification params
        self.cl_classification_labels = ['line', 'square', 'unspecified_shapes', 'dispersion', 'normal']

        ## summary
        self.summary_flag = True
        self.summary_writer = None

        ## debugger flag
        self.debugger_flag = True

    def training_network(self):

        ## Init train transforms
        transforms_train = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomRotation(10.),
                transforms.ToTensor()
            ]
        )

        transforms_validation = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        ) 

        transforms_test = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        ) 
            
        if self.debugger_flag :
            print(f"[cls module : training_network]setting path ========================================")
            print(f"train_data path : {self.path_train}")
            print(f"validation path : {self.path_validation}")
            print(f"test path : {self.path_test}")

        ## Init loader
        _, train_data_loader = self._get_data_loader(self.path_train, transforms_train)
        _, validation_data_loader = self._get_data_loader(self.path_validation, transforms_validation)
        _, test_data_loader = self._get_data_loader(self.path_test, transforms_test)

        ## Init device network
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.debugger_flag :
            print(f"[cls module : training_network]setting params ========================================")
            print(f"network name : {self.tr_network_name}")
            print(f"classification num : {len(self.cl_classification_labels)}")
            print(f"device name : {device} \n")

        network_model = self._get_network_model(
            self.tr_network_name, len(self.cl_classification_labels), device
        )

        ## Summary setting
        if self.summary_flag :
            self.summary_writer = SummaryWriter("{}_{}".format(self.tr_network_name, self.tr_epoch))            

        ## pt file params setting and load file
        pt_filepath = os.path.join(self.path_network_result, "{}.pt".format(self.tr_network_name))
        if os.path.isfile(pt_filepath):
            network_model.load_state_dict(torch.load(pt_filepath))
            network_model.eval()        

        ## Update var
        criterion = nn.CrossEntropyLoss().cuda()
        '''
        optimizer = torch.optim.Adam(
            network_model.parameters(), lr=self.tr_rate
        )
        '''

        optimizer = torch.optim.SGD(
            network_model.parameters(), self.tr_rate,
            momentum = 0.9, weight_decay= 1e-5, nesterov=True
        )

        best_validation_pt = os.path.join(self.path_network_result, "{}.pt".format(self.tr_network_name+"_best"))
        best_validation_test_rate = 0

        temp_size = 52
        total_step = 1
        for epoch_count in range(self.tr_epoch):
            for batch_size, data_set in enumerate(train_data_loader):
                if data_set['ori_image'] is not None:
                    # print(f"batch_size => {batch_size}   {self.tr_batch_size}")
                    labels, outputs = self._get_model_output(network_model, data_set, device)
                    
                    '''
                    if device_name == 'GoogleNet':
                        ## XXX : https://discuss.pytorch.org/t/question-about-googlenet/44896/5
                        outputs = outputs.logits
                    '''
                    
                    loss = criterion(outputs, labels)
    
                    ## Update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    ## if (batch_size + 1) % self.tr_batch_size == 0:
                    if (batch_size + 1) % temp_size == 0:
                        print(f'[Epoch {epoch_count} / {self.tr_epoch}], Iterate {total_step} / {self.tr_batch_size} Loss : [{loss.item():.4f}]')
                        torch.save(network_model.state_dict(), pt_filepath)
                        if self.summary_flag :
                            self.summary_writer.add_scalar('Train/loss(iter)', loss.item(), total_step)
                        total_step += 1
                    
                else :
                    continue
            ## batch, data_set loop end
            if epoch_count % 5 == 0 :
                ## check validation.
                correct = 0
                total = 0
                        
                for val_data_set in validation_data_loader:
                    if val_data_set['ori_image'] is not None:
                        val_labels, val_outputs = self._get_model_output(network_model, val_data_set, device)

                        _, predicted = torch.max(val_outputs.data, 1)
                        total += len(val_labels)
                        correct += (predicted == val_labels).sum().item()
                    else :
                        continue 
                        
                answer_ratio = 100 * correct / total
                if self.summary_flag :
                    self.summary_writer.add_scalar('validation_1/answer', answer_ratio, epoch_count)

                print(f'Validation Accuracy of the model on the {total} validation images: {answer_ratio} %')

                ## answer test
                answer_rate = self.testing_network(False, test_data_loader, network_model)
                if self.summary_flag :
                    self.summary_writer.add_scalar('validation_2/answer', answer_rate, epoch_count)

                if answer_rate > best_validation_test_rate :
                    torch.save(network_model.state_dict(), best_validation_pt)
                    print(f"best testing change {answer_rate} > {best_validation_test_rate}")
                    best_validation_test_rate = answer_rate


            ## validation block end

        ## epoch loop end    
        network_model.eval()
        self.testing_network(False, test_data_loader, network_model)



    def testing_network(self, outcall = True, test_loader = None, test_network = None):
        
        test_data_loader = None
        network_model = None
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if outcall :
            network_model = self._get_network_model(
                self.tr_network_name, len(self.cl_classification_labels), device
            )

            if self.debugger_flag :
                print(f"[cls module : testing_network]setting path ========================================")
                print(f"test_data path : {self.path_test}")
                print(f"network name : {self.tr_network_name}")
                print(f"classification num : {len(self.cl_classification_labels)}")
                print(f"device name : {device} \n")


            transforms_test = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ]
            ) 
            
            _, test_data_loader = self._get_data_loader(self.path_test, transforms_test)
        else :
            test_data_loader = test_loader
            network_model = test_network

        correct = 0
        total = 0
        answer_ratio = 0
        
        for data_set in test_data_loader:
            if data_set['ori_image'] is not None:
                labels, outputs = self._get_model_output(network_model, data_set, device)

                _, predicted = torch.max(outputs.data, 1)
                total += len(labels)
                correct += (predicted == labels).sum().item()
                answer_ratio = 100 * correct / total
                
                '''
                if self.summary_flag :
                    self.summary_writer.add_scalar('Test/answer', answer_ratio, total)
                '''

            else :
                continue
        
        print(f'Test Accuracy of the model on the {total} test images: {answer_ratio} %')
        
        return answer_ratio

    def classification_image(self, image_path, image_encode):
        
        result = []
        transforms_cl = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        ) 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        network_model = self._get_network_model(
            self.tr_network_name, len(self.cl_classification_labels), device
        )

        ## pt file params setting and load file
        pt_filepath = os.path.join(self.path_network_result, "{}.pt".format(self.tr_network_name))
        if os.path.isfile(pt_filepath):
            network_model.load_state_dict(torch.load(pt_filepath))
            network_model.eval()    

        ## data set, loader init
        dataset = ClassificationDataSet(image_path, image_encode, transforms_cl)
        data_loader = DataLoader(dataset, batch_size = 1)

        ## classification
        for data_set in data_loader:
            image = data_set["image"]
            file_name = data_set["file_name"]

            tensor_image = image.to(device)
            output = network_model(tensor_image)


            result.append({
                "class" : np.argmax(output).item(), ## max value index (label)
                "file_name" : file_name ## file name for ref image get
            })

        return result


    def _get_data_loader(self, file_path, transforms):
        dataset = self._get_dataset(
            file_path,
            transforms
        )
        data_loader = DataLoader(dataset, batch_size = self.tr_batch_size, shuffle = True)

        return dataset, data_loader


    ## Data loader getter
    def _get_dataset(self, file_path, set_transforms):
        if self.debugger_flag :
            print(f"[cls moudle : _get_dataset] =======================================")
            print(f"data set file path => {file_path} \n")
        
        data_set = CustomDataset(
            self.cl_classification_labels, file_path, self.tr_image_chanels ,set_transforms
        )

        if self.debugger_flag :
            print(f"data set load ok. \n")

        return data_set

    ## Network getter
    def _get_network_model(self, model_name, class_num, device):
        return {
            'CNN' : ConvNetwork(num_class = class_num).to(device),
            'CustomResNet' : ResNet('imagenet', 18, class_num).to(device),
            'RES18' : torch_models.resnet18().to(device),
            'RES34' : torch_models.resnet34().to(device),
            'RES50' : torch_models.resnet50().to(device),
            'RES101' : torch_models.resnet101().to(device),
            'GoogleNet' : torch_models.googlenet().to(device),
            'AlexNet' : torch_models.alexnet().to(device),
            'ShuffleNet' : torch_models.shufflenet_v2_x1_0().to(device)
        }.get(model_name, None)

    ## Get item
    def _get_model_output(self, model, item_set, device):
        ori_images = item_set['ori_image'].to(device)
        labels = item_set['label'].to(device)

        return labels, model(ori_images)