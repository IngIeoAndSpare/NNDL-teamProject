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
sys.path.append("..")
from Utils.CustomDataSetUtil import CustomDataset

## Netwrok utils module
from Network.ConvNetworkpy import ConvNetwork

class ClassificationModule:

    def __init__(self, train_path, test_path, network_result_path, tr_batch_size = 32, tr_epoch = 256, tr_rate = 0.0001):
        
        ## File path params
        self.path_train = train_path
        self.path_test = test_path
        self.path_network_result = network_result_path

        ## Train params
        self.tr_batch_size = tr_batch_size
        self.tr_epoch = tr_epoch
        self.tr_rate = tr_rate
        self.tr_image_chanels = "RGB"
        self.tr_network_name = ""

        ## Classification params
        self.cl_classification_labels = ['line', 'square', 'unspecified_shapes', 'dispersion', 'normal']

        ## summary
        self.summary_flag = False
        self.writer = None

    def training_network(self, network_name):

        ## Init train transforms
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

        ## Init loader
        _, train_data_loader = self._get_data_loader(self.path_train, transforms_train)
        _, test_data_loader = self._get_data_loader(self.path_test, transforms_test)

        ## Init device network
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network_model = self._get_network_model(
            self.tr_network_name, len(self.cl_classification_labels), device
        )

        ## Summary setting
        if self.summary_flag :
            self.writer = SummaryWriter("{}_{}".format(self.tr_network_name, self.tr_epoch))            

        ## pt file params setting and load file
        pt_filepath = os.path.join(self.path_network_result, "{}.pt".format(self.tr_network_name))
        if os.path.isfile(pt_filepath):
            network_model.load_state_dict(torch.load(pt_filepath))
            network_model.eval()        

        ## Update var
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(network_model.parameters(), lr=self.tr_rate)


        step = 1

        for epoch_count in range(self.tr_epoch):
            for batch_size, data_set in enumerate(train_data_loader):
                if data_set['ori_image'] is not None:
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

                    if (batch_size + 1) % self.tr_batch_size == 0:
                        print(f'[Epoch {epoch_count} / {self.tr_epoch}], Loss : [{loss.item():.4f}]')
                        torch.save(network_model.state_dict(), pt_filepath)
                        if self.summary_flag :
                            self.writer.add_scalar('Train/loss', loss.item(), step)

                        step += 1
                else :
                    continue

        network_model.eval()

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
                print(f'Test Accuracy of the model on the {total} test images: {answer_ratio} %')
                if self.summary_flag :
                    self.writer.add_scalar('Test/answer', answer_ratio, total)
            else :
                continue

            

    def _get_data_loader(self, file_path, transforms):
        dataset = self._get_dataset(
            file_path,
            transforms
        )
        data_loader = DataLoader(dataset, batch_size = self.tr_batch_size, shuffle = True)

        return dataset, data_loader


    ## Data loader getter
    def _get_dataset(self, file_path, transforms):
        return CustomDataset(
            self.cl_classification_labels, file_path, self.tr_image_chanels
        )

    ## Network getter
    def _get_network_model(self, model_name, class_num, device):
        return {
            'CNN' : ConvNetwork(num_class = class_num).to(device),
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