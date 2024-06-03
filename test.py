from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np


from light_dnn import  DNNDROP, DNNDROP1,DNNDROP4,DNNDROP6,DNNDROP8
from load_filelist import FileListDataLoader

parser = argparse.ArgumentParser(description='PyTorch Light DNN Training')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 100)')
# parser.add_argument('--resume', default='D:/WorkFiles/pythoncode/foodProject/model/TEXTCNN+Adam/DNNMLP_best_checkpoint.pth.tar', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='./model/DNN1best_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='./data/', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--val_list', default='./data/test.txt', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='./model/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

#outFile = open('output.txt', 'w')

def main():
    global args
    args = parser.parse_args()

    model = DNNDROP1()


    if args.cuda:
        model = model.cuda()

    print(model)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    val_loader = torch.utils.data.DataLoader(
        FileListDataLoader(root=args.root_path, fileList=args.val_list),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function and optimizer
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()

    if args.cuda:
        criterion.cuda()

    validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    losses     = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        predictList = output.tolist()[0]
        #outFile.write(str(predictList[0]) + ' ' + str(predictList[1]) + '\n')
        print(predictList)
        # print(target)
        # print(output)
        #print('=============')
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    print('\nTest set: Average loss: {}\n'.format(losses.avg))


def save_checkpoint(state, filename):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count



if __name__ == '__main__':
    main()