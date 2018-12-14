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
from train import iou_voxel, iou_shapelayer
from voxel2layer_torch import *
from ResNet import *
from DatasetLoader import *
from DatasetCollector import *

# id1, id2, id3 = generate_indices(32)

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

    parser = argparse.ArgumentParser(description='Train a Matryoshka Network')

    # general options
    parser.add_argument('--title',     type=str,            default='matryoshka', help='Title in logs, filename (default: matryoshka).')
    parser.add_argument('--no_cuda',   action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu',       type=int,            default=0,     help='GPU ID if cuda is available and enabled')
    parser.add_argument('--batchsize', type=int,            default=32,    help='input batch size for training (default: 128)')
    parser.add_argument('--nthreads',  type=int,            default=4,     help='number of threads for loader')
    parser.add_argument('--save_inter', type=int,           default=10,    help='Saving interval in epochs (default: 10)')

    # options for dataset
    parser.add_argument('--dataset',          type=str,            default=dataset_default, help=('Dataset [%s]' % ','.join(name2dataset.keys())))
    parser.add_argument('--set',              type=str,            default='val',           help='Validation or test set. (default: val)', choices=['val', 'test'])
    parser.add_argument('--basedir',          type=str,            default='./data/',       help='Base directory for dataset.')

    # options for network
    parser.add_argument('--file',  type=str, default=None, help='Savegame')
    parser.add_argument('--net',   type=str, default=net_default, help=('Network architecture [%s]' % ','.join(name2net.keys())))
    parser.add_argument('--ncomp', type=int, default=1,   help='Number of nested shape layers (default: 1)')
    

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
 
    device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")

    torch.manual_seed(1)
    
    savegame = torch.load(args.file)
    args.side = savegame['side']
    id1, id2, id3 = generate_indices(args.side, device)
    
    # load dataset
    try:
        logging.info('Initializing dataset "%s"' % args.dataset)
        Collector = name2dataset[args.dataset](side=args.side, basedir=args.basedir)        
    except KeyError:
        logging.error('A dataset named "%s" is not available.' % args.dataset)
        exit(1)

    logging.info('Initializing dataset loader')
    if args.set == 'val':
        samples = Collector.val()
    elif args.set == 'test':
        samples = Collector.test()

    num_samples = len(samples)
    logging.info('Found %d test samples.' % num_samples)
    test_loader = torch.utils.data.DataLoader(DatasetLoader(samples, args.ncomp, \
        input_transform=transforms.Compose([transforms.ToTensor()])), \
        batch_size=args.batchsize, shuffle=False, num_workers=args.nthreads, \
        pin_memory=True)
    samples = []

    net = name2net[args.net](\
            num_input_channels=3, 
            num_initial_channels=savegame['ninf'],
            num_inner_channels=savegame['ngf'],
            num_penultimate_channels=savegame['noutf'], 
            num_output_channels=6*args.ncomp,
            input_resolution=128, 
            output_resolution=savegame['side'],
            num_downsampling=savegame['down'], 
            num_blocks=savegame['block']
            ).to(device)
    logging.info(net)
    net.load_state_dict(savegame['state_dict'])
    
    net.eval()

    agg_iou   = 0.
    count     = 0
    results   = torch.zeros(args.batchsize*100, 6, 128,128).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            pred    = net(inputs)

            iou, bs = iou_shapelayer(shlx2shl(pred), targets, id1, id2, id3)
            agg_iou += float(iou)
            count   += bs

            logging.info('%d: %d/%d Mean IoU = %.1f' % (batch_idx, count, num_samples, 100 * agg_iou / count))

            i = batch_idx % 100
            results[i*args.batchsize:i*args.batchsize+bs,:,:,:] = pred
            if i == 99:
                sio.savemat('b_%03d.mat' % (batch_idx//100), {'results':results.detach().cpu().numpy()}, do_compression=True)
                saved = True
                pass
            if i == 0:
                saved = False
                pass
            pass
        
        if not saved:
            results = results[:i*args.batchsize+bs,:,:,:]
            sio.savemat('b_%03d.mat' % (batch_idx//100), {'results':results.detach().cpu().numpy()}, do_compression=True)
            pass  
        pass          
    pass


