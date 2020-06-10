'''
    2020. 1학기 UST-ETRI 수업 Neural Network and Deep Learning
    Error Aerial Photography Inpainting model trainer

    Writer : Joon Gyu Maeng(2020. May, Master's Program in Computer Software)
    Usage : python --modeltype=VAE/UNet --batch-size 15~20(recommended) train.py
    ## --nproc_per_node : set the parameter to available number of the your GPUs
    ## --modeltype : VAE or UNet
    ## REQUIREMENTS ##
    Python : 3.8.1
    CUDA : 10.1.2
    pytorch : 1.4.0
    torchvision : 0.5.0
    Nvidia Apex

    ## TRAINING ENVIRONMENTS ##
    OS : CentOS
    GPU : NVIDIA GEFORCE 1080Ti (11GB) X 4
    RAM : 128GB
    CPU : INTEL Xeon

    ## NORMAL ##
    단일 그래픽카드를 이용하고, APEX를 설치하지 않은 분들을 위한 모듈입니다.
'''

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.autograd import Variable

import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import datetime
import time
from Network.AutoEncoderNetwork import VAENetwork, UNetNetwork
import threading
import argparse

lock = threading.Lock()


torch.cuda.set_device(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_root_dir = './dataset/'


class apDataset(torch.utils.data.Dataset):
    def __init__(self, root, gt_dir, err_dir):
        self.gt_dir = os.path.join(root, gt_dir)
        self.err_dir = os.path.join(root, err_dir)
        self.data_files = os.listdir(self.err_dir)
        self.target_files = os.listdir(self.gt_dir)

    def __getitem__(self, idx):
        err_img = Image.open(os.path.join(self.err_dir, self.data_files[idx]))
        gt_img = Image.open(os.path.join(self.gt_dir, self.target_files[idx]))
        
        #temp = copy.deepcopy(err_img)
        #temp = np.array(temp, dtype=np.float32)
        gt_img = np.array(gt_img) * 1./255
        mask_img = np.array(err_img) * 1.
        err_img = np.array(err_img) * 1./255
        #mask_img = copy.deepcopy(err_img)
        #temp = np.array(temp, dtype=np.float32)
        
        # 기존 학습방식을 사용하려면 아래 코드 주석을 해제하시오(마스크영역 == 0,0,0)
        #mask_img = np.where(temp == (0, 0, 0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) # before err_img, (1,1,1), 0506 00:27
        
        #### 마스크영역 0,0,0 --> 255,255,255 로 변경해야함.
        #### 로드하는 에러이미지의 마스크영역도 255,255,255로 변경되려면? 이미 0,0,0이니까 나머지영역(1,1,1) -> (0,0,0) 에러영역(0,0,0) -> (255,255,255)
        mask = np.all((mask_img==0), axis=2)
        no_mask = np.any((mask_img!=0), axis=2)
        mask_img[mask] = [0,0,0]
        mask_img[no_mask] = [1,1,1]

        err_img[mask] = [1,1,1]

        err_img = transforms.ToTensor()(err_img.astype(dtype=np.float32))
        gt_img = transforms.ToTensor()(gt_img.astype(dtype=np.float32))
        mask_img = transforms.ToTensor()(mask_img.astype(dtype=np.float32))

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        err_img = transforms.Normalize(mean=mean, std=std)(err_img)
        gt_img = transforms.Normalize(mean=mean, std=std)(gt_img)
        ## 마스크는 노멀라이즈 하지 않는 것이 맞는것인가? 결과 보고 노말라이즈 한것과 비교해보자
        #mask_img = transforms.Normalize(mean=mean, std=std)(mask_img)
        # masks don't apply the Normalize......
        # https://discuss.pytorch.org/t/where-are-the-masks-unnormalized-for-segmentation-in-torchvision-train-file/48113
        
        return (err_img, mask_img, gt_img)


    def __len__(self):
        return len(self.data_files)


def plot_callback(epoch, masked, pred_img, ori, mode='train'):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""
    
    # if you want obtain variable images, set the image index randomly
    # maximum size --> batch_size
    idx = np.random.randint(args.batch_size)
    # Get samples & Display them
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    pred_img = ((pred_img[idx].squeeze().cpu().detach() * 0.5) + 0.5)
    masked = ((masked[idx].squeeze().cpu().detach() * 0.5) + 0.5)
    ori = ((ori[idx].squeeze().cpu().detach() * 0.5) + 0.5)
    pred_img = transforms.ToPILImage()(pred_img)
    masked = transforms.ToPILImage()(masked)
    ori = transforms.ToPILImage()(ori)


    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked)
    axes[1].imshow(pred_img)
    axes[2].imshow(ori)
    axes[0].set_title('Masked Image')
    axes[1].set_title('Predicted Image')
    axes[2].set_title('Original Image')

    #plt.show()

    if mode == 'train':
        path = './results/train/'
    else:
        path = './results/validatation/'

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + 'img_{}_{}.png'.format(epoch, pred_time))
    plt.close()

# early stop from https://discuss.pytorch.org/t/early-stopping-loop-does-not-break-training-continues/57896
## it didn't work.
## patience를 초과하면 모든 GPU의 train을 종료시켜야하는데 그걸 못하겠음
def is_early_stop(val_loss, epochs_no_improve, patience, min_val_loss, gpu):
    #group = [0,1,2,3]
    #stop = torch.tensor(False)
    if val_loss < min_val_loss:
        epochs_no_improve = 0
        min_val_loss = val_loss
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve == patience:
        #group.pop(gpu)
        #workers = dist.new_group(group)
        #stop = torch.tensor(True)
        #req = dist.broadcast(stop, gpu, workers, True)
        return True, epochs_no_improve, patience, min_val_loss
    else:
        #req = dist.recv(stop)
        #if stop == torch.tensor(True):
        #    return True, epochs_no_improve, patience, min_val_loss
        return False, epochs_no_improve, patience, min_val_loss

## working test
## just test which this module could be working
'''

apdataset = apDataset('./dataset/train/', 'gt_img/', 'err_img/')
print(apdataset)
data_loader = DataLoader(dataset=apdataset, batch_size=1, shuffle=True)
model = AutoEncoderNetwork()
model = model.cuda()

for i in range(5):
    for train_data, train_mask, train_target in data_loader:
        print(train_data.shape, train_mask.shape, train_target.shape)
        train_data, train_mask, train_target = train_data.float().cuda(), train_mask.float().cuda(), train_target.float().cuda()
        output, _, _ = model(train_data, train_mask)
        print('im working!')
        plot_callback(i, train_data, output, train_target)
'''

def train_(ngpus_per_node):
    # parameter for distributed training
    global best_prec1
    
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
    print("Model_type = {}".format(args.modeltype))

    model = None
    if args.modeltype == 'VAE':
        model = VAENetwork()
    elif args.modeltype == 'UNet':
        model = UNetNetwork()
    else:
        print('model is not defined')
        return

    model = model.cuda().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)  ## set default by cosmoVAE 0.0002

    # early stop parameters
    #min_val_loss = np.Inf
    #epochs_no_improve = 0
    #patience = 5
    #early_stop=False
    ## Network Train ##
    try:
        train_dataset_root = './dataset/train/'
        val_dataset_root = './dataset/val/'
        
        train_dataset = apDataset(train_dataset_root, 'gt_img/', 'err_img/')
        #train_sampler = DistributedSampler(train_dataset, shuffle=True)   # distributed parallel
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        ## batch_size was 25
        val_dataset = apDataset(val_dataset_root, 'gt_img/', 'err_img/')
        #val_sampler = DistributedSampler(val_dataset, shuffle=True)  # distributed parallel
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
       
       #Tensor board writer declare
        writer = None
        if not os.path.exists('./tensorboard'):
            try:
                os.mkdir('./tensorboard')
            except:
                print('the other gpu already makes tensorboard dir')
        writer = SummaryWriter('./tensorboard/AP_VAE_experiment')

        if not os.path.exists('./model_checkpoint/'):
            try:
                os.makedirs('./model_checkpoint/')
            except:
                print('the other gpu already makes checkpoint dir')
        

        print('training starts')
        for epoch in range(0, 200):
            #if args.distributed:
            print('=======================================================================\n')
            print('GPU : {} epoch : {} training now\n'.format(args.gpu, epoch))
            
            train_loss = train(train_data_loader, model, optimizer, epoch)
            validation_loss = validate(val_data_loader, model, epoch)
            
            writer.add_scalars('train_loss vs val_loss', {
                'train_loss': train_loss,
                'validation_loss': validation_loss
            }, epoch)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'loss' : validation_loss
            }, './model_checkpoint/epoch-{}.pth'.format(epoch))

        writer.close()
    except Exception as e:
        print(e)


def train(train_loader, model, optimizer, epoch):
    model.train()


    #prefetcher = data_prefetcher(train_loader)
    #input, mask, target = prefetcher.next()
    
    i = 0
    train_loss = 0
    #while input is not None:
    for input, mask, target in train_loader:
        start = time.time()
        input=input.float().cuda();mask=mask.float().cuda();target=target.float().cuda()
        i += 1
        optimizer.zero_grad()
        output, z_mean, z_log_var = model(input, mask)
        loss = model.loss(mask=mask, y_pred=output, y_true=target, z_mean=z_mean, z_log_var=z_log_var)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        end = time.time()

        print('''Epoch: [{}][{}/{}]\t
              Time {}\t
              loss {}\t'''.format(
               epoch, i, len(train_loader), end-start,loss))

        #input, mask, target = prefetcher.next()

    return train_loss
            
def validate(val_loader, model, epoch):
    # switch to evaluate mode
    model.eval()



    #prefetcher = data_prefetcher(val_loader)
    #input, mask, target = prefetcher.next()

    val_loss = 0
    i = 0
    #while input is not None:
    for input, mask, target in val_loader:
        start = time.time()
        input=input.float().cuda();mask=mask.float().cuda();target=target.float().cuda()
        i += 1

        # compute output
        with torch.no_grad():
            output, z_mean, z_log_var = model(input, mask)
            loss = model.loss(mask=mask, y_pred=output, y_true=target, z_mean=z_mean, z_log_var=z_log_var)
            val_loss += loss.item()

        # measure accuracy and record loss
        end = time.time()
        print('''Test: [{}/{}]\t
              Time {}\t
              Loss {}\t'''.format(
               i, len(val_loader),end-start, loss))

        if i < 3:
            plot_callback(epoch, input, output, target, 'val')

        #input, mask, target = prefetcher.next()
        

    return val_loss
        
        
def main_():
    global args
    ## arguments for Distributed Training
    parser = argparse.ArgumentParser(description='PyTorch AP_VAE Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
                        
    parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
                        

    parser.add_argument('--opt-level', type=str, default='O0')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)

    parser.add_argument('--modeltype', type=str)
        
    ## Parameter for Distributed Training ##
    args = parser.parse_args()
    ngpus_per_node =1
    args.world_size = ngpus_per_node * args.world_size
    train_(ngpus_per_node)
            
    
if __name__=='__main__':
    main_()
