from libs.CustomVGG import CustomVGG
from libs.module import PartialUp, PartialDown
from Network.AutoEncoderNetwork import UNetNetwork, VAENetwork, PSNR
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import copy
import argparse
from skimage.measure import compare_psnr, compare_ssim
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## TO USE REAL ERR IMAGE
class APRealTestDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.files = []
        for subdir in os.listdir(root):
            for file in os.listdir(os.path.join(root, subdir)):
                self.files.append(os.path.join('./test_data', subdir, file))


    def __getitem__(self, idx):
        err_img = Image.open(self.files[idx])


        # temp = copy.deepcopy(err_img)
        # temp = np.array(temp, dtype=np.float32)
        #gt_img = np.array(gt_img) * 1. / 255
        mask_img = np.array(err_img) * 1.
        err_img = np.array(err_img) * 1. / 255
        # mask_img = copy.deepcopy(err_img)
        # temp = np.array(temp, dtype=np.float32)

        # 기존 학습방식을 사용하려면 아래 코드 주석을 해제하시오(마스크영역 == 0,0,0)
        # mask_img = np.where(temp == (0, 0, 0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) # before err_img, (1,1,1), 0506 00:27

        #### 마스크영역 0,0,0 --> 255,255,255 로 변경해야함.
        #### 로드하는 에러이미지의 마스크영역도 255,255,255로 변경되려면? 이미 0,0,0이니까 나머지영역(1,1,1) -> (0,0,0) 에러영역(0,0,0) -> (255,255,255)
        mask = np.all((mask_img == 0), axis=2)
        no_mask = np.any((mask_img != 0), axis=2)
        mask_img[mask] = [0, 0, 0]
        mask_img[no_mask] = [1, 1, 1]

        err_img[mask] = [1, 1, 1]

        err_img = transforms.ToTensor()(err_img.astype(dtype=np.float32))
        #gt_img = transforms.ToTensor()(gt_img.astype(dtype=np.float32))
        mask_img = transforms.ToTensor()(mask_img.astype(dtype=np.float32))

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        err_img = transforms.Normalize(mean=mean, std=std)(err_img)
        #gt_img = transforms.Normalize(mean=mean, std=std)(gt_img)
        ## 마스크는 노멀라이즈 하지 않는 것이 맞는것인가? 결과 보고 노말라이즈 한것과 비교해보자
        # mask_img = transforms.Normalize(mean=mean, std=std)(mask_img)
        # masks don't apply the Normalize......
        # https://discuss.pytorch.org/t/where-are-the-masks-unnormalized-for-segmentation-in-torchvision-train-file/48113

        return (err_img, mask_img)

    def __len__(self):
        return len(self.files)

## TO USE TEST ERR IMAGE

class APTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, gt_dir, err_dir):
        self.gt_dir = os.path.join(root, gt_dir)
        self.err_dir = os.path.join(root, err_dir)
        self.data_files = os.listdir(self.err_dir)
        self.target_files = os.listdir(self.gt_dir)

    def __getitem__(self, idx):
        err_img = Image.open(os.path.join(self.err_dir, self.data_files[idx]))
        gt_img = Image.open(os.path.join(self.gt_dir, self.target_files[idx]))

        # temp = copy.deepcopy(err_img)
        # temp = np.array(temp, dtype=np.float32)
        gt_img = np.array(gt_img) * 1. / 255
        mask_img = np.array(err_img) * 1.
        err_img = np.array(err_img) * 1. / 255
        # mask_img = copy.deepcopy(err_img)
        # temp = np.array(temp, dtype=np.float32)

        # 기존 학습방식을 사용하려면 아래 코드 주석을 해제하시오(마스크영역 == 0,0,0)
        # mask_img = np.where(temp == (0, 0, 0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) # before err_img, (1,1,1), 0506 00:27

        #### 마스크영역 0,0,0 --> 255,255,255 로 변경해야함.
        #### 로드하는 에러이미지의 마스크영역도 255,255,255로 변경되려면? 이미 0,0,0이니까 나머지영역(1,1,1) -> (0,0,0) 에러영역(0,0,0) -> (255,255,255)
        mask = np.all((mask_img == 0), axis=2)
        no_mask = np.any((mask_img != 0), axis=2)
        mask_img[mask] = [0, 0, 0]
        mask_img[no_mask] = [1, 1, 1]

        err_img[mask] = [1, 1, 1]

        err_img = transforms.ToTensor()(err_img.astype(dtype=np.float32))
        gt_img = transforms.ToTensor()(gt_img.astype(dtype=np.float32))
        mask_img = transforms.ToTensor()(mask_img.astype(dtype=np.float32))

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        err_img = transforms.Normalize(mean=mean, std=std)(err_img)
        gt_img = transforms.Normalize(mean=mean, std=std)(gt_img)
        ## 마스크는 노멀라이즈 하지 않는 것이 맞는것인가? 결과 보고 노말라이즈 한것과 비교해보자
        # mask_img = transforms.Normalize(mean=mean, std=std)(mask_img)
        # masks don't apply the Normalize......
        # https://discuss.pytorch.org/t/where-are-the-masks-unnormalized-for-segmentation-in-torchvision-train-file/48113

        return (err_img, mask_img, gt_img)

    def __len__(self):
        return len(self.data_files)


def getAPDataset(root):
    test_data_root = root
    test_dataset = APTestDataset(test_data_root)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False)

    return test_data_loader


def get_parser():
    '''
        modeltype : VAE or Unet
    '''
    parser = argparse.ArgumentParser(description='Pytorch Aerial Photography Inpainting Module')
    parser.add_argument('--modeltype', type=str, help='VAE or Unet')
    parser.add_argument('--path', type=str, help='the path that you want to test')
    args = parser.parse_args()
    return args


def plotting(y_true, y_pred, err, modeltype, path):
    global i
    #now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    filename = '{}_{}'.format(i, modeltype)
    y_true = ((y_true.squeeze().cpu().detach() * 0.5) + 0.5)
    y_pred = ((y_pred.squeeze().cpu().detach() * 0.5) + 0.5)
    err = ((err.squeeze().cpu().detach() * 0.5) + 0.5)

    true = transforms.ToPILImage()(y_true)
    pred = transforms.ToPILImage()(y_pred)
    err = transforms.ToPILImage()(err)

    path = path + '/test_rlt/' + modeltype
    if not os.path.exists(path):
        os.makedirs(path)
    true_filename = filename + 'gt.png'
    pred_filename = filename + 'pred.png'
    err_filename = filename + 'err.png'

    plt.imshow(true)
    plt.savefig(path+'/'+true_filename)
    plt.close()
    plt.imshow(pred)
    plt.savefig(path +'/'+ pred_filename)
    plt.close()
    plt.imshow(err)
    plt.savefig(path + err_filename)
    plt.close()

    i += 1


def main():
    global i
    args = get_parser()

    ## model load ##
    model = None
    model_path = None
    modeltype = args.modeltype
    dataset_path = args.path
    try:
        if modeltype=='VAE':
            model = VAENetwork()
            model_path = './pth/vae.pth'
        elif modeltype=='Unet':
            model = UNetNetwork()
            model_path = './pth/unet.pth'
        else:
            print('Invalid Model Type. Usage : VAE or Unet')
            return
        model = model.to(device)
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print('model is not loaded')
        return


    i = 0
    for subdir in os.listdir(args.path):
        test_data_loader = getAPDataset(os.path.join(dataset_path, subdir))
        lst_psnr = []
        lst_ssim = []
        for err, mask, gt in test_data_loader:
            err = err.float()
            mask = mask.float()
            gt = gt.float()

            with torch.no_grad():
                output = model(err, mask)
                plotting(gt, output, err, modeltype=modeltype, path=os.path.join(dataset_path, subdir))
                for i in range(len(err)):
                    psnr = compare_psnr(
                        gt[i].detach().cpu().numpy(),
                        output[i].detach().cpu().numpy()
                    )
                    ssim = compare_ssim(
                        gt[i].detach().cpu().numpy(),
                        output[i].detach().cpu().numpy()
                    )
                    lst_psnr.append(psnr)
                    lst_ssim.append(ssim)


        with open('./{}_rlt.txt'.format(subdir), 'w') as rlt:
            rlt.write('Total Images : \t\t{}\n'.format(i))
            rlt.write('PSNR Average : \t\t{}\n'.format(np.mean(lst_psnr)))
            rlt.write('SSIM Average : \t\t{}\n'.format(np.mean(lst_ssim)))


    print('TEST DONE')