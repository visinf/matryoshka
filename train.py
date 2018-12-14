from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import scipy.io as sio
import functools
import PIL
import logging
import time
from utils import *
import math
import sys

# our stuff
from voxel2layer_torch import *
from ResNet import *
from DatasetLoader import *
from DatasetCollector import *

# id1, id2, id3 = generate_indices(32)

def pos_loss(pred, target, num_components=6):
    """ Modified L1-loss, which penalizes background pixels
        only if predictions are closer than 1 to being considered foreground.
    """

    fg_loss  = pred.new_zeros(1)
    bg_loss  = pred.new_zeros(1)
    fg_count = 0 # counter for normalization
    bg_count = 0 # counter for normalization

    for i in range(num_components):
        mask     = target[:,i,:,:].gt(0).float().detach()
        target_i = target[:,i,:,:]
        pred_i   = pred[:,i,:,:]

        # L1 between prediction and target only for foreground
        fg_loss  += torch.mean((torch.abs(pred_i-target_i)).mul(mask))
        fg_count += torch.mean(mask)

        # flip mask => background
        mask = 1-mask

        # L1 for background pixels > -1
        bg_loss  += torch.mean(((pred_i + 1)).clamp(min=0).mul(mask))
        bg_count += torch.mean(mask)
        pass

    return fg_loss / max(1, fg_count) + \
           bg_loss / max(1, bg_count)


def iou_voxel(pred, voxel):
    """ Computes intersection over union between two shapes.
        Returns iou summed over batch
    """
    bs,_,h,w = pred.size()
    
    inter = pred.mul(voxel).detach()
    union = pred.add(voxel).detach()
    union = union.sub_(inter)
    inter = inter.sum(3).sum(2).sum(1)
    union = union.sum(3).sum(2).sum(1)
    return inter.div(union).sum(), bs
        

def iou_shapelayer(pred, voxel, id1, id2, id3):
    """ Compares prediction and ground truth shape layers using IoU.
        Returns iou summed over batch and number of samples in batch.
    """
       
    pred  = pred.detach()
    voxel = voxel.detach()

    bs, _, side, _ = pred.shape
    vp = pred.new_zeros(bs,side,side,side, requires_grad=False)
    vt = pred.new_zeros(bs,side,side,side, requires_grad=False)
    
    for i in range(bs):
        vp[i,:,:,:] = decode_shape(pred[i,:,:,:].short().permute(1,2,0),  id1, id2, id3)
        vt[i,:,:,:] = decode_shape(voxel[i,:,:,:].short().permute(1,2,0), id1, id2, id3)

    return iou_voxel(vp,vt)
    

k_save = 0
def save(c, d, name=None):
    global k_save
    if c:
        k_save += 1
        if name is None:
            name = 'dbg_%d.mat' % k_save
        sio.savemat(name, {k:d[k].detach().cpu().numpy() for k in d.keys()})



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logging.info(sys.argv) # nice to have in log files

    # register networks, datasets, etc.
    name2net        = {'resnet': ResNet}
    net_default     = 'resnet'

    name2dataset    = {\
        'SanityCheck':SanityCollector, \
        'ShapeNetPTN':ShapeNetPTNCollector, \
        'ShapeNetCars':ShapeNetCarsOGNCollector, \
        'ShapeNet':ShapeNet3DR2N2Collector}
    dataset_default = 'ShapeNet'

    name2optim      = {'adam': optim.Adam}
    optim_default   = 'adam'


    parser = argparse.ArgumentParser(description='Train a Matryoshka Network')

    # general options
    parser.add_argument('--title',     type=str,            default='matryoshka', help='Title in logs, filename (default: matryoshka).')
    parser.add_argument('--no_cuda',   action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu',       type=int,            default=0,     help='GPU ID if cuda is available and enabled')
    parser.add_argument('--no_save',   action='store_true', default=False, help='Disables saving of final model')
    parser.add_argument('--no_val',    action='store_true', default=False, help='Disable validation for faster training')
    parser.add_argument('--batchsize', type=int,            default=32,    help='input batch size for training (default: 32)')
    parser.add_argument('--epochs',    type=int,            default=40,    help='number of epochs to train')
    parser.add_argument('--nthreads',  type=int,            default=4,     help='number of threads for loader')
    parser.add_argument('--seed',      type=int,            default=1,     help='random seed (default: 1)')
    parser.add_argument('--val_inter', type=int,            default=1,     help='Validation interval in epochs (default: 1)')
    parser.add_argument('--log_inter', type=int,            default=100,   help='Logging interval in batches (default: 100)')
    parser.add_argument('--save_inter', type=int,           default=10,    help='Saving interval in epochs (default: 10)')

    # options for optimizer
    parser.add_argument('--optim', type=str,   default=optim_default, help=('Optimizer [%s]' % ','.join(name2optim.keys())))
    parser.add_argument('--lr',    type=float, default=1e-3,          help='Learning rate (default: 1e-3)')
    parser.add_argument('--decay', type=float, default=0,             help='Weight decay for optimizer (default: 0)')
    parser.add_argument('--drop',  type=int,   default=30)

    # options for dataset
    parser.add_argument('--dataset',          type=str,            default=dataset_default, help=('Dataset [%s]' % ','.join(name2dataset.keys())))
    parser.add_argument('--basedir',          type=str,            default='./data/',       help='Base directory for dataset.')
    parser.add_argument('--no_shuffle_train', action='store_true', default=False,           help='Disable shuffling of training samples')
    parser.add_argument('--no_shuffle_val',   action='store_true', default=False,           help='Disable shuffling of validation samples')


    # options for network
    parser.add_argument('--file',  type=str, default=None, help='Savegame')
    parser.add_argument('--net',   type=str, default=net_default, help=('Network architecture [%s]' % ','.join(name2net.keys())))
    parser.add_argument('--side',  type=int, default=128, help='Output resolution [if dataset has multiple resolutions.] (default: 128)')
    parser.add_argument('--ncomp', type=int, default=1,   help='Number of nested shape layers (default: 1)')
    parser.add_argument('--ninf',  type=int, default=8,   help='Number of initial feature channels (default: 8)')
    parser.add_argument('--ngf',   type=int, default=512, help='Number of inner channels to train (default: 512)')
    parser.add_argument('--noutf', type=int, default=128, help='Number of penultimate feature channels (default: 128)')
    parser.add_argument('--down',  type=int, default=5,   help='Number of downsampling blocks. (default: 5)')
    parser.add_argument('--block', type=int, default=1,   help='Number of inner blocks at same resolution. (default: 1)')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.shuffle_train = not args.no_shuffle_train
    args.shuffle_val   = not args.no_shuffle_val 

    device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")

    id1, id2, id3 = generate_indices(args.side, device)

    torch.manual_seed(1)

    # load dataset
    try:
        logging.info('Initializing dataset "%s"' % args.dataset)
        Collector = name2dataset[args.dataset](resolution=args.side, base_dir=args.basedir)        
    except KeyError:
        logging.error('A dataset named "%s" is not available.' % args.net)
        exit(1)

    logging.info('Initializing dataset loader')
    samples = Collector.train()
    logging.info('Found %d training samples.' % len(samples))
    train_loader = torch.utils.data.DataLoader(DatasetLoader(samples, args.ncomp, \
        input_transform=transforms.Compose([transforms.ToTensor(), RandomColorFlip()])), \
        batch_size=args.batchsize, shuffle=args.shuffle_train, num_workers=args.nthreads, \
        pin_memory=True)

    if not args.no_val:
        samples = Collector.val()
        logging.info('Found %d validation samples.' % len(samples))
        val_loader = torch.utils.data.DataLoader(DatasetLoader(samples, args.ncomp, \
        input_transform=transforms.Compose([transforms.ToTensor()])), \
        batch_size=args.batchsize, shuffle=args.shuffle_val,   num_workers=args.nthreads, \
        pin_memory=True)
        pass

    samples = []

    # load network
    try:
        logging.info('Initializing "%s" network' % args.net)
        net = name2net[args.net](\
            num_input_channels=3, 
            num_initial_channels=args.ninf,
            num_inner_channels=args.ngf,
            num_penultimate_channels=args.noutf, 
            num_output_channels=6*args.ncomp,
            input_resolution=128, 
            output_resolution=args.side,
            num_downsampling=args.down, 
            num_blocks=args.block
            ).to(device)
        logging.info(net)
    except KeyError:
        logging.error('A network named "%s" is not available.' % args.net)
        exit(2)

    if args.file:
        savegame = torch.load(args.file)
        net.load_state_dict(savegame['state_dict'])

    # init optimizer
    try:
        logging.info('Initializing "%s" optimizer with learning rate = %f and weight decay = %f' % (args.optim, args.lr, args.decay))
        optimizer = name2optim[args.optim](net.parameters(), lr=args.lr, weight_decay=args.decay)
    except KeyError:
        logging.error('An optimizer named "%s" is not available.' % args.optim)
        exit(3)


   
    try:
        net.train()

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.drop, gamma=0.5)

        agg_loss  = 0.
        count     = 0

        for epoch in range(1, args.epochs + 1):

            scheduler.step()

            for batch_idx, (inputs, targets) in enumerate(train_loader):

                optimizer.zero_grad()

                inputs  = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                pred    = net(inputs)
                loss    = pos_loss(pred, shl2shlx(targets))
    
                loss.backward()        
                optimizer.step()

                agg_loss += loss.detach()
                count    += inputs.shape[0]
                pass
            
                if batch_idx % args.log_inter == 0:
                    logging.info('%d/%d: Train loss: %.5f [%s]' % (epoch, batch_idx, agg_loss.item()/count, args.title))
                    agg_loss = 0.
                    count    = 0
                # save(True, {'inputs':inputs, 'targets':shl2shlx(targets), 'pred':pred})
            
            if not args.no_save and epoch % args.save_inter == 0:
                filename = '%s_%s_%d.pth.tar' % (args.title, args.dataset, epoch)
                logging.info('Saving model to %s.' % filename)
                torch.save({'state_dict': net.state_dict(), 
                     'optimizer' : optimizer.state_dict(),
                     'ninf':args.ninf,
                     'ngf':args.ngf,
                     'noutf':args.noutf,
                     'block':args.block,
                     'side': args.side,
                     'down':args.down,
                     'epoch': epoch,
                     'optim': args.optim,
                     'lr': args.lr,
                 }, filename) 
     
            # validation
            if not args.no_val and epoch % args.val_inter == 0:

                net.eval()
        
                agg_iou = 0.
                count   = 0
                with torch.no_grad():    
                    for batch_idx, (inputs, targets) in enumerate(val_loader):

                        inputs  = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                
                        pred     = net(inputs)                
                        iou, bs  = iou_shapelayer(shlx2shl(pred), targets, id1, id2, id3)
                        agg_iou += float(iou)
                        count   += bs

                        # save(True, {'inputs':inputs, 'targets':shl2shlx(targets), 'pred':pred}, 'val')
                        pass
                    pass
            
                net.train()
                
                total_iou = (100 * agg_iou / count) if count > 0 else 0

                logging.info('%d: Val set accuracy, iou: %.1f [%s]' % (epoch, total_iou, args.title))                
                pass
            pass

    except KeyboardInterrupt:
        pass

    if not args.no_save:
        filename = '%s_%s_%d.pth.tar' % (args.title, args.dataset, epoch)
        logging.info('Saving model to %s.' % filename)
        torch.save({'state_dict': net.state_dict(), 
                     'optimizer' : optimizer.state_dict(),
                     'ninf':args.ninf,
                     'ngf':args.ngf,
                     'noutf':args.noutf,
                     'block':args.block,
                     'side': args.side,
                     'down':args.down,
                     'epoch': epoch,
                     'optim': args.optim,
                     'lr': args.lr,
                 }, filename) 
