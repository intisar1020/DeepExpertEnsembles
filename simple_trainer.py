from ast import mod
import os
import sys
import argparse
import logging
import time

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

# models
import models.cifar as models

# utils
from utils.ms_net_utils import *
from utils.data_utils import *
from utils.basic_utils import count_parameters_in_MB
from utils.basic_utils import load_pretrained_model
from utils.basic_utils import AverageMeter, accuracy, transform_time

parser = argparse.ArgumentParser(description='Stable MS-NET')

# save dirs:
parser.add_argument('--save_root', type=str, default='work_space', help='models and logs are saved here')
parser.add_argument('--id', type=str, default='ti_resnet20_router', help='experiment IDS')


# training params.
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')


# learning rate, scheduler, momentum
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 300, 400],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# dataset
parser.add_argument('-d', '--dataset', default='data/cifar100png/', type=str)
parser.add_argument('-name', '--data_name', default='cifar100', type=str)



# architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help='backbone architecture')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock')



# Others
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--pretrained_wts', type=str, default='work_space/exp_0/', help='load with pretrained wts')

args = parser.parse_args() # its easier for me to keep ti global.



# INITS
torch.cuda.manual_seed(args.seed)
cudnn.enabled = True
cudnn.benchmark = True

# prepare the save dirs
args.save_root = os.path.join(args.save_root, args.id)
os.makedirs(args.save_root, exist_ok=True)
# loggers
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'msnet.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def train(epoch, model, train_loader, optimizer):
    ''' one epoch trainer '''
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top2       = AverageMeter()
    model.train()
    end = time.time()
    #loss_fn = distillation
    criterion = nn.CrossEntropyLoss()

    for i, (dta, target) in enumerate(train_loader, start=1):
        dta, target = dta.cuda(), target.cuda()
        output = model(dta) # infer, forward prop.
        loss = criterion(output, target)
        prec1, prec2 = accuracy(output, target, topk=(1,2))
        losses.update(loss.item(), dta.size(0))
        top1.update(prec1.item(), dta.size(0))
        top2.update(prec2.item(), dta.size(0))
        
        # backprop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
					   'Time:{batch_time.val:.4f} '
					   'Data:{data_time.val:.4f}  '
					   'loss:{losses.val:.4f}({losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@2:{top2.val:.2f}({top2.avg:.2f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
					   losses=losses, top1=top1, top2=top2))
            logging.info(log_str)


def test(model, test_loader):
    ''' test single epochs '''
    model.eval()
    losses = AverageMeter()
    top1   = AverageMeter()
    top2   = AverageMeter()
    for dta, target in test_loader:
        dta, target = dta.cuda(), target.cuda()
        with torch.no_grad():
            output = model(dta)
        #output = F.softmax(output, dim=1)
        loss = F.cross_entropy(output, target).item() # sum up batch loss
        
        prec1, prec2 = accuracy(output, target, topk=(1,2))
        losses.update(loss, dta.size(0))
        top1.update(prec1.item(), dta.size(0))
        top2.update(prec2.item(), dta.size(0))

    f_l = [losses.avg, top1.avg, top2.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@2: {:.2f}'.format(*f_l))
    return top1.avg, top2.avg


def adjust_learning_rate(epoch, optimizer):
    if epoch in args.schedule:
        print ("REDUCING LEARNING RATE")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def main():
    # Prepare the main model.
    model = models.__dict__[args.arch](
        num_classes=200,
        depth=args.depth,
        block_name=args.block_name)
    model = model.cuda() # Trans. to GPU
    print (model)
    # if pre-trained weight exists init.
    # wts_path = os.path.join(args.pretrained_wts, 'model_best.pth.tar')
    # pretrained_wts = torch.load(wts_path)
    # load_pretrained_model(model, pretrained_wts['net'])
    logging.info('Student param size = %fMB', count_parameters_in_MB(model))
    
    # Prepare dataloader. 
    #train_loader, test_loader, test_loader_single, val_loader_single, num_classes, list_of_classes
    trldr, tstldr, _, _, _, _ = get_dataloader(
        data_name=args.data_name,
        dataset_path=args.dataset, 
        TRAIN_BATCH=128, 
        TEST_BATCH=256)
    
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9,
        weight_decay=1e-4)
    
    best_so_far = 0
    for epoch in range(1, 500): 
        train(epoch, model, trldr, optimizer)
        adjust_learning_rate(epoch, optimizer)
        top1, top2 = test(model, tstldr)
        is_best = False
        if top1 > best_so_far:
            best_so_far = top1
            is_best = True

        if (epoch in args.schedule):
            logging.info(f"Saving checkpoint from epoch: {epoch}")
            save_path = os.path.join(args.save_root, f'checkpoint_{epoch}.pth.tar')
            torch.save({
            'epoch': epoch,
            'net': model.state_dict(),
            'prec@1': top1,
            'prec@2': top2,
            }, save_path)

        if (is_best):
            logging.info('Founds the best model, saving ......')
            save_path = os.path.join(args.save_root, 'model_best.pth.tar')
            torch.save({
            'epoch': epoch,
            'net': model.state_dict(),
            'prec@1': top1,
            'prec@2': top2,
            }, save_path)


if __name__ == '__main__':
    main()
    

    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=50, 
    #     T_mult=1, 
    #     eta_min=0.005, 
    #     last_epoch=-1,
    #     verbose=True
    #     )
    