import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import torch
from libs.module import PartialDown, PartialUp
from libs.CustomVGG import CustomVGG
from libs.loss import GramL1Loss


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 512, 2, 2)

class UNetNetwork(nn.Module):
    '''
        CAUTION : this class is Network. so, you coding only Network method and sub module.
    '''

    def __init__(self, base=64, freeze=False):
        '''
            TODO : params init
            ex ->   self.{{PARAMS_NAME}} = {{PARAMS_INIT_VALUE}}
        '''
        super(UNetNetwork, self).__init__()
        # self.vgg = self.build_vgg()
        self.vgg = CustomVGG(model_path='vgg_conv.pth', download=True)
        for v in self.vgg.parameters():
            v.requires_grad = False
        self.l1 = nn.L1Loss()
        self.style_loss = GramL1Loss()
        # Encoder layers
        self.down1 = PartialDown(3, base, 7, 2, 3, use_batch_norm=False)
        self.down2 = PartialDown(base, base * 2, 5, 2, 2, freeze=freeze)
        self.down3 = PartialDown(base * 2, base * 4, 5, 2, 2, freeze=freeze)
        self.down4 = PartialDown(base * 4, base * 8, 3, 2, 1, freeze=freeze)
        self.down5 = PartialDown(base * 8, base * 8, 3, 2, 1, freeze=freeze)
        self.down6 = PartialDown(base * 8, base * 8, 3, 2, 1, freeze=freeze)
        self.down7 = PartialDown(base * 8, base * 8, 3, 2, 1, freeze=freeze)
        self.down8 = PartialDown(base * 8, base * 8, 3, 2, 1, freeze=freeze)

        # Define decoder layers
        self.up8 = PartialUp(base * 8, base * 8, base * 8, 3, 1, 1)
        self.up7 = PartialUp(base * 8, base * 8, base * 8, 3, 1, 1)
        self.up6 = PartialUp(base * 8, base * 8, base * 8, 3, 1, 1)
        self.up5 = PartialUp(base * 8, base * 8, base * 8, 3, 1, 1)
        self.up4 = PartialUp(base * 8, base * 4, base * 4, 3, 1, 1)
        self.up3 = PartialUp(base * 4, base * 2, base * 2, 3, 1, 1)
        self.up2 = PartialUp(base * 2, base, base, 3, 1, 1)
        self.up1 = PartialUp(base, 3, 3, 3, 1, 1, use_batch_norm=False, use_lr=False)

    def sampling(self, args):
        z_mean, z_log_var = args
        # print('z_mean : ', z_mean)
        # print('z_log_var : ', z_log_var)
        epsilon = torch.randn(size=z_mean.size()).cuda()
        # print('eps : ', epsilon)
        return z_mean + torch.exp(z_log_var * 0.5).cuda() * epsilon

    def forward(self, img, mask):
        x1, m1 = self.down1(img, mask)
        x2, m2 = self.down2(x1, m1)
        x3, m3 = self.down3(x2, m2)
        x4, m4 = self.down4(x3, m3)
        x5, m5 = self.down5(x4, m4)
        x6, m6 = self.down6(x5, m5)
        x7, m7 = self.down7(x6, m6)
        x8, m8 = self.down8(x7, m7)

        # print(e_conv6.shape)
        # Latent#
        # e_flatten = self.e_flatten(x7)  ## flatten
        # e_fc2 = F.relu(self.e_fc2(e_flatten))
        # e_fc3 = F.relu(self.e_fc3(e_fc2))

        # z_mean = self.z_mean_fc(e_fc2)
        # z_log_var = self.z_log_var_fc(e_fc2)
        z_mean, z_log_var = None, None
        # z = self.sampling([z_mean, z_log_var])
        # print('z : ', z)

        # d_fc3 = F.leaky_relu(self.d_fc3(z), negative_slope=0.2)
        # d_fc2 = F.leaky_relu(self.d_fc2(z), negative_slope=0.2)
        # d_fc1 = F.leaky_relu(self.d_fc1(d_fc2), negative_slope=0.2)
        # d_unflatten = self.d_unflatten(d_fc1)

        x_, m_ = self.up8(x8, x7, m8, m7)
        x_, m_ = self.up7(x_, x6, m_, m6)
        x_, m_ = self.up6(x_, x5, m_, m5)
        x_, m_ = self.up5(x_, x4, m_, m4)
        x_, m_ = self.up4(x_, x3, m_, m3)
        x_, m_ = self.up3(x_, x2, m_, m2)
        x_, m_ = self.up2(x_, x1, m_, m1)
        recon_img, recon_mask = self.up1(x_, img, m_, mask)
        # self.recon_img = F.tanh(self.recon_img)
        recon_img = F.sigmoid(recon_img) * 2 - 1

        return recon_img, z_mean, z_log_var

    def build_vgg(self):
        p_vgg = vgg().cuda()
        return p_vgg

    def loss(self, mask, y_true, y_pred, z_mean, z_log_var):
        # print(mask.shape, y_true.shape, y_pred.shape)
        style_list = "p1,p2,p3"
        style_list = style_list.split(',')

        y_comp = mask * y_pred + (1 - mask) * y_true

        vgg_out = self.vgg(y_pred)
        vgg_gt = self.vgg(y_true)
        vgg_comp = self.vgg(y_comp)

        l1 = self.loss_valid(mask, y_true, y_pred)
        l2 = self.loss_hole(mask, y_true, y_pred)
        l3, l4 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp, style_list)
        # l4 = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())  # KL-Divergence
        l5 = self.loss_tv(y_comp)

        # default l4 = 1 * l4
        return l1 + 6 * l2 + 0.05 * l3 + 120 * l4 + 0.1 * l5
        # l4 loss lambda was 0.001 (0510)

    def loss_hole(self, mask, y_true, y_pred):
        return self.l1((1 - mask) * y_true, (1 - mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        return self.l1(mask * y_true, mask * y_pred)

    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp, style_list):
        style_out = None
        style_comp = None
        perceptual = None
        for layer in style_list:
            if perceptual is None:
                perceptual = self.l1(vgg_out[layer], vgg_gt[layer])
            else:
                perceptual += self.l1(vgg_out[layer], vgg_gt[layer])
            if style_out is None:
                style_out = self.style_loss(vgg_out[layer], vgg_gt[layer])
            else:
                style_out += self.style_loss(vgg_out[layer], vgg_gt[layer])

            if style_comp is None:
                style_comp = self.style_loss(vgg_comp[layer], vgg_gt[layer])
            else:
                style_comp = self.style_loss(vgg_comp[layer], vgg_gt[layer])
        loss_style = style_out + style_out
        return perceptual, loss_style

    def loss_tv(self, y_comp):
        b, c, h, w = y_comp.size()
        vertical_target = y_comp[:, :, 1:, :].data
        horizontal_target = y_comp[:, :, :, 1:].data
        loss_tv = self.l1(y_comp[:, :, :h - 1, :], vertical_target) + self.l1(y_comp[:, :, :, :w - 1],
                                                                              horizontal_target)
        return loss_tv


class VAENetwork(nn.Module):
    '''
            CAUTION : this class is Network. so, you coding only Network method and sub module.
        '''

    def __init__(self, base=64, freeze=False):
        '''
            TODO : params init
            ex ->   self.{{PARAMS_NAME}} = {{PARAMS_INIT_VALUE}}
        '''
        super(VAENetwork, self).__init__()
        # self.vgg = self.build_vgg()
        self.vgg = CustomVGG(model_path='vgg_conv.pth', download=True)
        for v in self.vgg.parameters():
            v.requires_grad = False
        self.l1 = nn.L1Loss()
        self.style_loss = GramL1Loss()
        # Encoder layers
        self.down1 = PartialDown(3, base, 7, 2, 3, use_batch_norm=False)
        self.down2 = PartialDown(base, base * 2, 5, 2, 2, freeze=freeze)
        self.down3 = PartialDown(base * 2, base * 4, 5, 2, 2, freeze=freeze)
        self.down4 = PartialDown(base * 4, base * 8, 3, 2, 1, freeze=freeze)
        self.down5 = PartialDown(base * 8, base * 8, 3, 2, 1, freeze=freeze)
        self.down6 = PartialDown(base * 8, base * 8, 3, 2, 1, freeze=freeze)
        self.down7 = PartialDown(base * 8, base * 8, 3, 2, 1, freeze=freeze)
        self.down8 = PartialDown(base * 8, base * 8, 3, 2, 1, freeze=freeze)

        self.e_flatten = Flatten()
        self.e_fc2 = nn.Linear(2048, 1024)  # activation='relu'
        # self.e_fc3 = nn.Linear(2048, 1000)

        # self.z_mean_fc = nn.Linear(1000, 500)  ## z-mean
        # self.z_log_var_fc = nn.Linear(1000, 500)  ## z-log-var
        self.z_mean_fc = nn.Linear(1024, 1000)  ## z-mean
        self.z_log_var_fc = nn.Linear(1024, 1000)  ## z-log-var

        # self.d_fc3 = nn.Linear(500, 1000)  ## activation=LeakyReLU(negative_slope=0.2)
        self.d_fc2 = nn.Linear(1000, 1024)
        self.d_fc1 = nn.Linear(1024, 2048)
        self.d_unflatten = UnFlatten()

        # Define decoder layers
        self.up8 = PartialUp(base * 8, base * 8, base * 8, 3, 1, 1)
        self.up7 = PartialUp(base * 8, base * 8, base * 8, 3, 1, 1)
        self.up6 = PartialUp(base * 8, base * 8, base * 8, 3, 1, 1)
        self.up5 = PartialUp(base * 8, base * 8, base * 8, 3, 1, 1)
        self.up4 = PartialUp(base * 8, base * 4, base * 4, 3, 1, 1)
        self.up3 = PartialUp(base * 4, base * 2, base * 2, 3, 1, 1)
        self.up2 = PartialUp(base * 2, base, base, 3, 1, 1)
        self.up1 = PartialUp(base, 3, 3, 3, 1, 1, use_batch_norm=False, use_lr=False)

    def sampling(self, args):
        z_mean, z_log_var = args
        # print('z_mean : ', z_mean)
        # print('z_log_var : ', z_log_var)
        epsilon = torch.randn(size=z_mean.size()).cuda()
        # print('eps : ', epsilon)
        return z_mean + torch.exp(z_log_var * 0.5).cuda() * epsilon

    def forward(self, img, mask):
        x1, m1 = self.down1(img, mask)
        x2, m2 = self.down2(x1, m1)
        x3, m3 = self.down3(x2, m2)
        x4, m4 = self.down4(x3, m3)
        x5, m5 = self.down5(x4, m4)
        x6, m6 = self.down6(x5, m5)
        x7, m7 = self.down7(x6, m6)
        # x8, m8 = self.down8(x7, m7)

        # print(e_conv6.shape)
        # Latent#
        e_flatten = self.e_flatten(x7)  ## flatten
        e_fc2 = F.relu(self.e_fc2(e_flatten))
        # e_fc3 = F.relu(self.e_fc3(e_fc2))

        z_mean = self.z_mean_fc(e_fc2)
        z_log_var = self.z_log_var_fc(e_fc2)

        z = self.sampling([z_mean, z_log_var])
        # print('z : ', z)

        # d_fc3 = F.leaky_relu(self.d_fc3(z), negative_slope=0.2)
        d_fc2 = F.leaky_relu(self.d_fc2(z), negative_slope=0.2)
        d_fc1 = F.leaky_relu(self.d_fc1(d_fc2), negative_slope=0.2)
        d_unflatten = self.d_unflatten(d_fc1)

        # x_, m_ = self.up8(d_unflatten, x7, m8, m7)
        x_, m_ = self.up7(d_unflatten, x6, m7, m6)
        x_, m_ = self.up6(x_, x5, m_, m5)
        x_, m_ = self.up5(x_, x4, m_, m4)
        x_, m_ = self.up4(x_, x3, m_, m3)
        x_, m_ = self.up3(x_, x2, m_, m2)
        x_, m_ = self.up2(x_, x1, m_, m1)
        recon_img, recon_mask = self.up1(x_, img, m_, mask)
        # self.recon_img = F.tanh(self.recon_img)
        recon_img = F.sigmoid(recon_img) * 2 - 1

        return recon_img, z_mean, z_log_var

    def build_vgg(self):
        p_vgg = vgg().cuda()
        return p_vgg

    def loss(self, mask, y_true, y_pred, z_mean, z_log_var):
        # print(mask.shape, y_true.shape, y_pred.shape)
        style_list = "p1,p2,p3"
        style_list = style_list.split(',')

        y_comp = mask * y_pred + (1 - mask) * y_true

        vgg_out = self.vgg(y_pred)
        vgg_gt = self.vgg(y_true)
        vgg_comp = self.vgg(y_comp)

        l1 = self.loss_valid(mask, y_true, y_pred)
        l2 = self.loss_hole(mask, y_true, y_pred)
        l3 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp, style_list)
        l4 = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())  # KL-Divergence
        l5 = self.loss_tv(y_comp)

        # default l4 = 1 * l4
        return l1 + 6 * l2 + 0.05 * l3 + l4 + 0.1 * l5
        # l4 loss lambda was 0.001 (0510)

    def loss_hole(self, mask, y_true, y_pred):
        return self.l1((1 - mask) * y_true, (1 - mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        return self.l1(mask * y_true, mask * y_pred)

    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp, style_list):
        style_out = None
        style_comp = None
        for layer in style_list:
            if style_out is None:
                style_out = self.style_loss(vgg_out[layer], vgg_gt[layer])
            else:
                style_out += self.style_loss(vgg_out[layer], vgg_gt[layer])

            if style_comp is None:
                style_comp = self.style_loss(vgg_comp[layer], vgg_gt[layer])
            else:
                style_comp = self.style_loss(vgg_comp[layer], vgg_gt[layer])
        return style_out + style_comp

    def loss_tv(self, y_comp):
        b, c, h, w = y_comp.size()
        vertical_target = y_comp[:, :, 1:, :].data
        horizontal_target = y_comp[:, :, :, 1:].data
        loss_tv = self.l1(y_comp[:, :, :h - 1, :], vertical_target) + self.l1(y_comp[:, :, :, :w - 1],
                                                                              horizontal_target)
        return loss_tv


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self._vgg = vgg16(pretrained=True, progress=True)
        for v in self._vgg.parameters():
            v.requires_grad = False
        # vgg_layers = [4, 9, 16]  # 4, 9, 16 --> first pooling layers
        # vgg_outputs = [self._vgg.features[i] for i in vgg_layers]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()
        # print(vgg_outputs)
        encoder = list(self._vgg.children())[0][:23]
        self.pool1 = encoder[0:5]
        self.pool2 = encoder[5:10]
        self.pool3 = encoder[10:17]

    def forward(self, x):
        ## Normalization Again????
        x = (x - self.mean) / self.std
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        return x


def PSNR(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    The equation is:
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

    Our input is scaled with be within the range -2.11 to 2.64 (imagenet value scaling). We use the difference between these
    two values (4.75) as MAX_I

    this used for metrics
    """
    # return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
    return - 10.0 * torch.log(torch.mean(torch.pow(y_pred - y_true, 2))) / torch.log(torch.tensor(10.0))


if __name__ == '__main__':
    model = VAENetwork()
    vgg = model.build_vgg()
    for v in vgg._vgg.parameters():
        # print(v)
        v.requires_grad = False
    vgg_layers = [4, 9, 16]  # 4, 9, 16 --> first pooling layers
    vgg_outputs = [vgg._vgg.features[i] for i in vgg_layers]
    pool1 = vgg_outputs[0]
    pool2 = vgg_outputs[1]
    pool3 = vgg_outputs[2]
    # print(pool1, pool2, pool3)
    # print(list(vgg.children())[0])
    encoder = list(vgg._vgg.children())[0][:23]
    conv1 = encoder[0:5]
    conv2 = encoder[5:10]
    conv3 = encoder[10:17]
    print(conv1)
    print(conv2)
    print(conv3)



