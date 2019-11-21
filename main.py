from __future__ import print_function
import os
import shutil
import argparse
import time
import datetime
import math
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers import MultiBoxLoss
from layers import PriorBox
from models.retinaface import RetinaFace


parser = argparse.ArgumentParser(
    description='Retinaface face Detector Training With Pytorch')
parser.add_argument('--training_dataset', default='./data/widerface/label.txt', 
                    help='Training dataset directory')
parser.add_argument('--network', default='resnet50',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size',default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--pretrained', default=True, type=str,
                    help='use pre-trained model')
parser.add_argument('--resume',default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--distributed', default=True, type=str,
                    help='use distribute training')
parser.add_argument("--local_rank", default=0, type=int)                  
parser.add_argument('--lr', '--learning-rate',default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_folder',default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--prefix',default='pyramidbox_',
                    help='the prefix for saving checkpoint models')
args = parser.parse_args()


cudnn.benchmark = True
args = parser.parse_args()
minmum_loss = np.inf

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

def main():
    global args
    global minmum_loss
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                                init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    # build dsfd network 
    print("Building net...")
    model = RetinaFace(cfg=cfg)
    print("Printing net...")

    # for multi gpu
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    model = model.cuda()
    # optimizer and loss function  
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                            weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.35, True, 0, True, 7, 0.35, False)

    ## dataset 
    print("loading dataset")
    train_dataset = WiderFaceDetection(args.training_dataset,preproc(cfg['image_size'], cfg['rgb_mean']))

    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                collate_fn=detection_collate,
                                pin_memory=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            minmum_loss = checkpoint['minmum_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('Using the specified args:')
    print(args)
    # load PriorBox
    print("Load priorbox")
    with torch.no_grad():
        priorbox = PriorBox(cfg=cfg, image_size=(cfg['image_size'], cfg['image_size']))
        priors = priorbox.forward()
        priors = priors.cuda()
    
    print("start traing")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss = train(train_loader, model, priors, criterion,optimizer, epoch)
        if args.local_rank == 0:
            is_best = train_loss < minmum_loss
            minmum_loss = min(train_loss, minmum_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': minmum_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch)


def train(train_loader, model, priors, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_loss = AverageMeter()
    cls_loss= AverageMeter()
    landm_loss = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader, 1):
        train_loader_len = len(train_loader)
        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)

        images, targets = data

        input_var = Variable(images.cuda())  
        targets = [Variable(ann.cuda(), requires_grad=False)
                                for ann in targets]
        # compute output
        output = model(input_var)
        loss_l, loss_c, loss_landm = criterion(output, priors, targets)

        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            reduced_loss_loc = reduce_tensor(loss_l.data)
            reduced_loss_cls = reduce_tensor(loss_c.data)
            reduced_loss_landm = reduce_tensor(loss_landm.data)
        else:
            reduced_loss = loss.data
            reduced_loss_loc = loss_l.data
            reduced_loss_cls = loss_c.data
            reduced_loss_landm = loss_landm.data

        losses.update(to_python_float(reduced_loss), images.size(0))
        loc_loss.update(to_python_float(reduced_loss_loc), images.size(0))
        cls_loss.update(to_python_float(reduced_loss_cls), images.size(0))
        landm_loss.update(to_python_float(reduced_loss_landm), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i >= 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'loc_loss {loc_loss.val:.3f} ({loc_loss.avg:.3f})\t'
                  'cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})\t'
                  'landm_loss {landm_loss.val:.3f} ({landm_loss.avg:.3f})'.format(
                   epoch, i, train_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, loc_loss=loc_loss, cls_loss=cls_loss, landm_loss=landm_loss))
    return losses.avg


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

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

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 10

    if epoch >= 30:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 1:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(args.local_rank == 0 and step % args.print_freq == 0 and step > 1):
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, epoch):
    filename = os.path.join(args.save_folder, args.prefix + str(epoch)+ ".pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_folder, 'model_best.pth'))

if __name__ == '__main__':
    main()