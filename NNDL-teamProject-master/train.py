'''
    2020. 1학기 UST-ETRI 수업 Neural Network and Deep Learning
    Error Aerial Photography Inpainting model trainer

    Writer : Joon Gyu Maeng(2020. May, Master's Program in Computer Software)
    Usage : python -m torch.distributed.launch --nproc_per_node=4 --modeltype=VAE train.py
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

    ## APEX ##
    Apex는 torch의 DistributedDataParellel을 최적화한 API입니다.
    Best-Fit을 찾기위해 각 GPU노드들의 Loss값을 gethering을 통해 평균을 구하는 작업이 있습니다.
    결과적으로는 rank 0번의 GPU의 weight를 저장합니다.
'''

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.autograd import Variable

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import datetime
import time
from Network.AutoEncoderNetwork import VAENetwork, UNetNetwork
import copy
import threading
import argparse
import multiprocessing as mp

lock = threading.Lock()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    idx = np.random.randint(20)
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
    
    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1


    args.gpu = 0
    args.world_size=1
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    

    
    torch.cuda.set_device(args.gpu)
    print("Use GPU : {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + args.gpu
    dist.init_process_group(backend='nccl', 
                            init_method='env://')
    args.world_size = dist.get_world_size()
    
    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    
    args.lr = args.lr * float(20*args.world_size)/256. # float(args.batch_size*args.world_size)/256.

    model = None
    if args.modeltype == 'VAE':
        model = VAENetwork()
    elif args.modeltype == 'UNet':
        model = UNetNetwork()
    else:
        print('model is not defined')
        return

    model = model.cuda().to(memory_format=memory_format)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  ## set default by cosmoVAE 0.0002    
    
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
                                      
    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    
   
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
        train_sampler = DistributedSampler(train_dataset, shuffle=True)   # distributed parallel
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=20, sampler=train_sampler)
        ## batch_size was 25
        val_dataset = apDataset(val_dataset_root, 'gt_img/', 'err_img/')
        val_sampler = DistributedSampler(val_dataset, shuffle=True)  # distributed parallel
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=20, sampler=val_sampler)
       
       #Tensor board writer declare
        writer = None    
        if args.local_rank == 0:
            if not os.path.exists('./tensorboard'):
                try:
                    os.mkdir('./tensorboard')
                except:
                    print('the other gpu already makes tensorboard dir')
            writer = SummaryWriter('./tensorboard/AP_VAE_experiment')
            #optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5,0.999))  ## set default by cosmoVAE 0.0002
            
            
            if not os.path.exists('./model_checkpoint/'):
                try:
                    os.makedirs('./model_checkpoint/')
                except:
                    print('the other gpu already makes checkpoint dir')
        

        print('training starts')
        for epoch in range(0, 200):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            print('=======================================================================\n')
            print('GPU : {} epoch : {} training now\n'.format(args.gpu, epoch))
            
            train_loss = train(train_data_loader, model, optimizer, epoch)
            validation_loss = validate(val_data_loader, model, epoch)
            
            if args.local_rank == 0:
                #best_prec1 = max(prec1, best_prec1)
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
            


        print('VAE training Done')
        writer.close()
    except Exception as e:
        print(e)

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, mask, target = prefetcher.next()
    
    i = 0
    while input is not None:
        input=input.float();mask=mask.float();target=target.float()
        i += 1
        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output, z_mean, z_log_var = model(input, mask)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        loss = model.module.loss(mask=mask, y_pred=output, y_true=target, z_mean=z_mean, z_log_var=z_log_var)

        # compute gradient and do Adam step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            #prec1, prec5 = accuracy(output.data, target, topk=(1, 1))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                #prec1 = reduce_tensor(prec1)
                #prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data
            
            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            #top1.update(to_python_float(prec1), input.size(0))
            #top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader),
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses))
        if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
        input, mask, target = prefetcher.next()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
    
    return losses.avg
            
def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, mask, target = prefetcher.next()
    
    i = 0
    while input is not None:
        input=input.float();mask=mask.float();target=target.float()
        i += 1

        # compute output
        with torch.no_grad():
            output, z_mean, z_log_var = model(input, mask)
            loss = model.module.loss(mask=mask, y_pred=output, y_true=target, z_mean=z_mean, z_log_var=z_log_var)

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            #prec1 = reduce_tensor(prec1)
            #prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data
        
        if i < 2:
            plot_callback(epoch, input, output, target, 'val')
        
        losses.update(to_python_float(reduced_loss), input.size(0))
        #top1.update(to_python_float(prec1), input.size(0))
        #top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader),
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses))

        input, mask, target = prefetcher.next()
        

    return losses.avg

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        #self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        #self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_mask, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_mask = None
            self.next_target = None
         
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_mask = self.next_mask.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #self.next_input = self.next_input.float()
            #self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        mask = self.next_mask
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if mask is not None:
            mask.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, mask, target


        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    print('1 error?', maxk, batch_size)
    _, pred = output.topk(maxk, 1, True, True)
    print('2 error?')
    pred = pred.t()
    print('3 error?')
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print('4 error?')
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    print('5 error?')
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    #print(rt, args.world_size)
    return rt
        
        
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
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    train_(ngpus_per_node)
            
    
if __name__=='__main__':
    main_()
