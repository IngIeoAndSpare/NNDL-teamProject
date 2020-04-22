## torch module
import torch.nn as nn

class ConvNetwork(nn.Module):
    
    def __init__(self, num_class):
        super(ConvNetwork, self).__init__()
        
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
