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


from light_dnn import DNNDROP, DNNDROP1, DNNDROP4, DNNDROP6, DNNDROP8
from load_filelist import FileListDataLoader

parser = argparse.ArgumentParser(description='PyTorch Light DNN Training')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='./data/', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='./data/train.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='./data/val.txt', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='./model/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

outFileLog = open('logdnn1.txt', 'a+')

def main():
    global args
    args = parser.parse_args()
    model = DNNDROP1()
    if args.cuda:
        model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

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

    #load image
    train_loader = torch.utils.data.DataLoader(
        FileListDataLoader(root=args.root_path, fileList=args.train_list),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        FileListDataLoader(root=args.root_path, fileList=args.val_list),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)   

    # define loss function and optimizer
    #criterion = nn.MSELoss()
    #criterion = nn.()
    criterion = nn.L1Loss()

    if args.cuda:
        criterion.cuda()

    validate(val_loader, model, criterion)
    bestValidloss = 100000
    for epoch in range(args.start_epoch, args.epochs):

        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        validloss = validate(val_loader, model, criterion)

        if(validloss<bestValidloss):
            bestValidloss = validloss
            print('bestValidloss: ' + str(bestValidloss))
            save_name = args.save_path + 'DNN1' + 'best' + '_checkpoint.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }, save_name)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.cuda:
            input   = input.cuda()
            target  = target.cuda()

        output = model(input)


        loss   = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
    losses     = AverageMeter()
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.cuda:
            input = input.cuda()
            target = target.cuda()
        #print(target)
        # compute output
        output = model(input)
        #print(output)
        loss   = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    print('\nTest set: Average loss: {}\n'.format(losses.avg))

    #print(losses.avg)
    outFileLog.write(str(losses.avg) + '\n')
    outFileLog.flush()
    return losses.avg

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

def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step  = 150
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

if __name__ == '__main__':
    main()