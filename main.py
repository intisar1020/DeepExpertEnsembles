# System
from pickle import TRUE
import random
import copy
import argparse
import os
from re import split
import sys
import logging
import time
from turtle import Turtle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# math and viz.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# eval.
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# models and utisl
from msnet import MSNET
import models.cifar as models
from utils.ms_net_utils import *
from utils.data_utils import *

from utils.basic_utils import count_parameters_in_MB
from utils.basic_utils import load_pretrained_model
from utils.basic_utils import AverageMeter, accuracy, transform_time
from utils.basic_utils import is_subset, have_common_elements

# losses
from kd_losses import *



parser = argparse.ArgumentParser(description='Stable MS-NET')

#experiment tracker
parser.add_argument('--exp_id', default='ti_exp0', type=str, help='id of your current experiments')


# Hyper-parameters
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')

parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')

parser.add_argument('--corrected_images', type=str, default='./corrected_images/')

###############################################################################
parser.add_argument('--expert_train_epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train experts')
##########################################################################

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--initialize_with_router', action='store_true', default=True)

parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=80, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--evaluate_only_router', action='store_true',
                    help='evaluate router on testing set')

parser.add_argument('--weighted_sampler', action='store_true',
                    help='what sampler you want?, subsetsampler or weighted')

parser.add_argument('--finetune_experts', action='store_true', default=True,
                    help='perform fine-tuning of layer few layers of experts')

parser.add_argument('--save_images', action='store_true', default=True)

###########################################################################
parser.add_argument('--train_mode', action='store_true', default=True, help='Do you want to train or test?')


parser.add_argument('--topk', type=int, default=100, metavar='N',
                    help='how many experts you want?')
###########################################################################


# Paths
parser.add_argument('-cp', '--checkpoint_path', default='checkpoint_experts', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint_experts)')
parser.add_argument('-router_cp', '--router_cp', default='work_space/pre-trained_wts/resnet20/run2/model_best.pth.tar', type=str, metavar='PATH',
                    help='checkpoint path of the router weight')
parser.add_argument('-router_cp_icc', '--router_cp_icc', default='work_space/pre-trained_wts/resnet20_icc/model_best.pth.tar', type=str, metavar='PATH',
                    help='checkpoint path of the router weight for icc. We eval. router train on partial set of train data for ICC calculation.')

parser.add_argument('-dp', '--dataset_path', default='/path/to/dataset', type=str)
parser.add_argument('-save_log_path', '--save_log_path', default='./logs/', type=str)

# Architecture details
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='backbone architecture')

parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='LR',
                    help='initial learning rate to train')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-gpu', '--gpu_id', default=0, type=str, help='set gpu number')

args = parser.parse_args()


log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
os.makedirs(args.save_log_path, exist_ok=True)
fh = logging.FileHandler(os.path.join(args.save_log_path, 'msnet.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

state = {k: v for k, v in args._get_kwargs()}
state = str(state)
with open('state.txt', 'w') as f:
    f.write(state)
f.close()
model_weights = {}
use_cuda = torch.cuda.is_available()


if (use_cuda):
    args.cuda = True
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



# def inference_with_experts_and_routers(test_loader, experts, router, topk=2, temp_loader=None):
#     """ function to perform evaluation with experts
    
#     Args:
#         test_loader (_type_): _description_
#         experts (_type_): _description_
#         router (_type_): _description_
#         topk (int, optional): _description_. Defaults to 2.
#         temp_loader (_type_, optional): _description_. Defaults to None.
#     """
#     freqMat = np.zeros((100, 100)) # -- debug
#     router.eval()
#     experts_on_stack = []
#     expert_count = {} 
#     for k, v in experts.items():
#         experts[k].eval()
#         experts_on_stack.append(k)
#         expert_count[k] = 0
#         # test_expert(experts[k], temp_loader[k])
#         # test_expert(router, temp_loader[k])

#     count = 0
#     ext_ = '.png'
#     correct = 0
#     by_experts = 0
#     by_router = 0
#     mistake_by_experts, mistake_by_router = 0, 0
#     agree, disagree = 0, 0

#     for dta, target in test_loader:
#         count += 1
#         dta, target = dta.cuda(), target.cuda()
#         with torch.no_grad():
#             output_raw = router(dta)
#         output = F.softmax(output_raw, dim=1)
#         router_confs, router_preds = torch.sort(output, dim=1, descending=True)
#         preds = []
#         confs = []
#         for k in range(0, topk):
#             #ref = torch.argsort(output, dim=1, descending=True)[0:, k]
#             ref = router_preds[0:, k]
#             conf = router_confs[0:, k]
#             preds.append(ref.detach().cpu().numpy()[0]) # simply put the number. not the graph
#             confs.append(conf.detach().cpu().numpy()[0])
    
#         experts_output = []
     
#         list_of_experts = []
#         target_string = str(target.cpu().numpy()[0])
#         for exp in experts_on_stack: #
#             exp_cls = exp.split("_")
#             for r_pred in preds:
#                 if (str(r_pred) in exp_cls and exp not in list_of_experts):
#                     list_of_experts.append(exp)
#                     expert_count[exp] += 1
#                     break
            
                    
                    
#         with torch.no_grad():
#             experts_output = [experts[exp_](dta) for exp_ in list_of_experts]
#         experts_output.append(output_raw)
#         experts_output_avg = average(experts_output)
#         exp_conf, exp_pred = torch.sort(experts_output_avg, dim=1, descending=True)
#         #print (f"List of Experts: {list_of_experts}, Expert prediction: {exp_pred[0:, 0]}, router pred: {preds}, target: {target}")
#         pred = exp_pred[0:, 0]

#         if (pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
#             correct += 1
#         if (preds[0] == target.cpu().numpy()[0]): #or preds[1] == target.cpu().numpy()[0]):
#             by_router += 1
            
#        # mask_ = torch.zeros([1, 100])
#         # mask_ = mask_.cuda()
#         # mask_[0][preds[0]] = 1
#         # mask_[0][preds[1]] = 1
#         # experts_output_prob = F.softmax(experts_output_avg, dim=1)
#         # pred = torch.argsort(experts_output_prob, dim=1, descending=True)[0:, 0]
#         # experts_output_avg = experts_output_avg * mask_      
       
#     print (f"Expert dict: {expert_count}, MS-NET acc: {correct}, Router acc: {by_router}")



def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(
        F.log_softmax(y/T, dim=1), 
        F.softmax(teacher_scores/T, dim=1))   *   (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def train(epoch, model, teacher, train_loader, optimizer):
    ''' one epoch trainer '''
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top2       = AverageMeter()
    model.train()
    end = time.time()
    criterion_ce = nn.CrossEntropyLoss()
    for i, (dta, target) in enumerate(train_loader, start=1):
        dta, target = dta.cuda(), target.cuda()
        output = model(dta) # infer, forward prop.
        loss = criterion_ce(output, target)
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


def train_distilled_ensemble(epoch, model, teacher_msnet, teacher_list, train_loader, optimizer):
    """_summary_

    Args:
        epoch (_type_): _description_
        model (_type_): _description_
        teacher_msnet (_type_): _description_
        teacher_list (_type_): _description_
        train_loader (_type_): _description_
        optimizer (_type_): _description_
    """
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top2       = AverageMeter()
    model.train()
    end = time.time()
    #criterion = SoftTarget(T=2)
    criterion = Logits()
    criterion_ce = nn.CrossEntropyLoss()
    lambda_ = 0.5
    for i, (dta, target) in enumerate(train_loader, start=1):
        dta, target = dta.cuda(), target.cuda()
        output = model(dta) # infer, forward prop.
        with torch.no_grad():
            outputs_teacher = [teacher_msnet[idx](dta).detach() for idx in teacher_list]
            output_teacher = average(outputs_teacher)
        loss = (lambda_ * criterion(output, output_teacher)) +  ((1. - lambda_)  * criterion_ce(output, target))
        #loss = criterion_ce(output, target)
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
    """_summary_

    Args:
        model (_type_): _description_
        test_loader (_type_): _description_

    Returns:
        _type_: _description_
    """
    losses = AverageMeter()
    top1   = AverageMeter()
    top2   = AverageMeter()
    for dta, target in test_loader:
        dta, target = dta.cuda(), target.cuda()
        with torch.no_grad():
            output = model(dta)
        output = F.softmax(output, dim=1)
        loss = F.cross_entropy(output, target).item() # sum up batch loss
        prec1, prec2 = accuracy(output, target, topk=(1,2))
        losses.update(loss, dta.size(0))
        top1.update(prec1.item(), dta.size(0))
        top2.update(prec2.item(), dta.size(0))

    f_l = [losses.avg, top1.avg, top2.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@2: {:.2f}'.format(*f_l))
    return top1.avg, top2.avg


def test_expert(model, test_loader):
    ''' test single epochs '''
    model.eval()
    losses = AverageMeter()
    top1   = AverageMeter()
    top2   = AverageMeter()
    for dta, target in test_loader:
        dta, target = dta.cuda(), target.cuda()
        with torch.no_grad():
            output = model(dta)
        output = F.softmax(output, dim=1)
        loss = F.cross_entropy(output, target).item() # sum up batch loss
        prec1, prec2 = accuracy(output, target, topk=(1,2))
        losses.update(loss, dta.size(0))
        top1.update(prec1.item(), dta.size(0))
        top2.update(prec2.item(), dta.size(0))

    f_l = [losses.avg, top1.avg, top2.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@2: {:.2f}'.format(*f_l))
    return top1.avg, top2.avg


def make_router(num_classes, ckpt_path=None):
    """returns a router network (generalist model)

    Args:
        num_classes (int): _description_
        ckpt_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = models.__dict__[args.arch](
        num_classes=num_classes,
        depth=args.depth,
        block_name=args.block_name)
    model = model.cuda()
    if (ckpt_path):
        chk = torch.load(ckpt_path)
        model.load_state_dict(chk['net'])
        logging.info(f"Loading router from ckpt path: {ckpt_path}")
    
    return model


def load_experts(num_classes, list_of_index=[None], pretrained=True):
    """returns dictionary of expert network, where number of experts equal number of elements in the list_of_index

    Args:
        num_classes (int): total number of classes in the dataset.
        list_of_index (list, optional): _description_. Defaults to [None].

    Returns:
        _type_: _description_
    """
    experts = {}
    for loi in list_of_index:
        experts[loi] = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            block_name=args.block_name)
        
        experts[loi] = experts[loi].cuda()

        # we load pretrained weights for teacher networks.
        if (pretrained):
            chk_path = os.path.join("work_space", 'superset_100', args.checkpoint_path, loi + '.pth')
            chk = torch.load(chk_path)
            experts[loi].load_state_dict(chk['net'])      
    
    return experts


def load_teacher_network():
    """ return the best teacher network with state_dict. """
    teacher = models.__dict__['resnext'](
        cardinality=8,
        num_classes=100,
        depth=29,
        widen_factor=4,
        dropRate=0,
        )
    teacher = torch.nn.DataParallel(teacher).cuda()
    checkpoint = torch.load("work_space/pre-trained_wts/resnext/model_best.pth.tar")
    teacher.load_state_dict(checkpoint['state_dict'])
    return teacher

        
def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)
    

def ensemble_inference(test_loader, experts, router):
    
    router.eval()
    experts_on_stack = []
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
    
    losses = AverageMeter()
    top1   = AverageMeter()
    top2   = AverageMeter()
    for dta, target in test_loader:
        dta, target = dta.cuda(), target.cuda()
        list_of_experts = [] 
        for exp in experts_on_stack:
            list_of_experts.append(experts[exp])
        with torch.no_grad():
            all_outputs = [exp_(dta) for exp_ in list_of_experts]
            output = router(dta)
        all_outputs.append(output)
        all_outputs_avg = average(all_outputs)
        all_output_prob = F.softmax(all_outputs_avg)
        loss = F.cross_entropy(all_output_prob, target).item() # sum up batch loss
        prec1, prec2 = accuracy(all_output_prob, target, topk=(1,2))
        losses.update(loss, dta.size(0))
        top1.update(prec1.item(), dta.size(0))
        top2.update(prec2.item(), dta.size(0))

    f_l = [losses.avg, top1.avg, top2.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@2: {:.2f}'.format(*f_l))
    return top1.avg, top2.avg



def adjust_learning_rate(epoch, optimizer):
    if epoch in args.schedule:
        print ("\n\n***************CHANGED LEARNING RATE TO\n*********************\n")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        for param in optimizer.param_groups:
            print ("Lr {}".format(param['lr']))


def get_teacher_list(loi, teacher_lois):
    teacher_supersets = []
    split_indexes = lambda x: x.split("_")
    student_loi_set = set(split_indexes(loi))
    for teacher_loi in teacher_lois:
        teacher_loi_set = set(split_indexes(teacher_loi))
        if (have_common_elements(teacher_loi_set, student_loi_set)):
            teacher_supersets.append(teacher_loi)
    return teacher_supersets


def main():
    
    _, test_loader_router, test_loader_single, val_loader_single, num_classes, list_of_classes = get_dataloader(
        dataset_path=args.dataset_path,
        TRAIN_BATCH=args.train_batch, 
        TEST_BATCH=args.test_batch)

    logging.info("==> creating standalone router model")
    router = make_router(num_classes, ckpt_path=args.router_cp)

    logging.info("==> creating ICC router model")
    router_icc = make_router(num_classes, ckpt_path=args.router_cp_icc)

    size_of_router = sum(p.numel() for p in router.parameters() if p.requires_grad == True)
    print ("Network size {:.2f}M".format(size_of_router/1000000))

    logging.info("=====> Loading teacher network....")
    teacher = load_teacher_network()
    logging.info("success in teacher module load")
    
    if (args.evaluate_only_router):
        test(teacher, test_loader_router)
        return

    #########################################################################
    matrix = calculate_matrix(router_icc, val_loader_single, num_classes, only_top2=True)
    #####################################################################################
    binary_list, super_list, dict_ = return_topk_args_from_heatmap(matrix, num_classes, cutoff_thresold=3, binary_=False)

    #####################################################################
    logging.info ("Calculating the heatmap for confusing class....")
    ls = np.arange(num_classes)
    heatmap(matrix, ls, ls) # show heatmap
    barchart(dict_) # show barchart for ICC pair and corresponding number of
    #####################################################################

    expert_train_dataloaders,  expert_test_dataloaders, lois = expert_dataloader(
                data_name="cifar100",
                dataset_path=args.dataset_path,
                matrix=super_list,
                TRAIN_BATCH=args.train_batch, 
                TEST_BATCH=args.test_batch,
                weighted_sampler=True)


    _,  single_class_test_dataloaders, _ = expert_dataloader(
                data_name="cifar100",
                dataset_path=args.dataset_path,
                matrix=[[i] for i in range(100)],
                TRAIN_BATCH=args.train_batch, 
                TEST_BATCH=1,
                weighted_sampler=True) 
    

    logging.info("Printing confusing classes .. ")
    lois_named = {}

    # In the following  we first list the names of all the pre-trained super-set experts.
    # These pretrained experts will serve as good teacher network. We will use ensemble of 
    # them for the training individual experts. So we first list all the index names 
    index_list = os.listdir(os.path.join("work_space", "superset_100", args.checkpoint_path))
    split_f = lambda x: x.split(".")[0]
    teacher_lois = [split_f(index_) for index_ in index_list]
    
    for loi in lois:
        try:
            index_list = loi.split("_")
        except:
            continue 
        name_str = ""
        for index_ in index_list:
            name_str += list_of_classes[int(index_)] + ", "
        lois_named[loi] = name_str
        logging.info(f"Numeric index: {loi}, Named index: {name_str}")
    logging.info(f"Number of supersets: {len(super_list)}")
  
    lois = lois[8:] #+ lois[-3:]
  
    msnet = load_experts(num_classes, list_of_index=lois, pretrained=False) # pool of de-coupled expoert networks.
    teacher_msnet = load_experts(num_classes, list_of_index=teacher_lois, pretrained=False) # should set to true when using as teacher.
    

    args.train_mode = True
    if (not args.train_mode):
        # index_list = os.listdir(os.path.join("work_space", args.exp_id, args.checkpoint_path))
        # split_f = lambda x: x.split(".")[0]
        # lois = [split_f(index_) for index_ in index_list]
        for loi in lois:
            try:
                wts = torch.load(os.path.join("work_space", args.exp_id, args.checkpoint_path, loi+'.pth'))
                logging.info(f"Checkpoint for model: {loi} loaded")
            except:
                logging.info(f"Checkpoint for model: {loi} not found")
                continue
            msnet[loi].load_state_dict(wts['net'])
            #test_expert(msnet[loi], expert_test_dataloaders[loi])
            #test_expert(router, expert_test_dataloaders[loi])
        ensemble_inference(test_loader_router, msnet, router)
        #single_class_test_dataloaders['35'] expert_test_dataloaders['4_72_91_44_18_55_27_30']
        inference_with_experts_and_routers(test_loader_single, msnet, router, topk=3, temp_loader=expert_test_dataloaders)


    if (args.train_mode):
        for loi in lois:
            optimizer = optim.SGD(msnet[loi].parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            best_so_far = 0.0
            # teacher_supersets = get_teacher_list(loi, teacher_lois)
            # if (not len(teacher_supersets)):
            #     teacher_supersets = random.choices(teacher_lois, k=5)
            # logging.info("Teachers for distill: ", teacher_supersets)
            for epoch in range(1, args.expert_train_epochs):
                adjust_learning_rate(epoch, optimizer)
                train(epoch, msnet[loi], teacher, expert_train_dataloaders[loi], optimizer)
                #train_distilled_ensemble(epoch, msnet[loi], teacher_msnet, teacher_supersets, expert_train_dataloaders[loi], optimizer)
                t1, t2 = test_expert(msnet[loi], expert_test_dataloaders[loi])
                t1_all, t2 = test_expert(msnet[loi], test_loader_router)
                
                is_best = False
                if t1_all > best_so_far:
                    best_so_far = t1_all
                    is_best = True

                if (is_best):
                    logging.info(f'Founds the best model for {loi}, saving ......')
                    base_path = os.path.join("work_space", args.exp_id, args.checkpoint_path)
                    os.makedirs(base_path, exist_ok=True)
                    save_path = os.path.join(base_path, f'{loi}.pth')
                    torch.save({
                        'epoch': epoch,
                        'net': msnet[loi].state_dict(),
                        'prec@1': t1,
                        'prec@2': t2,
                        }, save_path)



if __name__ == '__main__':
    main() 
        
    # the following net is for inference only
    # ckpt_path_ = os.path.join("work_space", args.exp_id, args.checkpoint_path)
    # msnet = MSNET(
    #     expert_base=router, 
    #     router_base=None,
    #     named_nodes=None,
    #     pretrained=True,
    #     ckpt_path=ckpt_path_)
