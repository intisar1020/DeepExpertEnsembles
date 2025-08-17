# System
from email import header
from email.policy import strict
from enum import auto
from pickle import TRUE
from pydoc import visiblename
import random
import copy
import argparse
import os
from re import split
import sys
import logging
import time
import math
# progress bar
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
from msnet import MSNET
import models.cifar as models
from utils.ms_net_utils import *
from utils.data_utils import *

from utils.basic_utils import count_parameters_in_MB
from utils.basic_utils import load_pretrained_model
from utils.basic_utils import AverageMeter, accuracy, transform_time
from utils.basic_utils import is_subset, have_common_elements


parser = argparse.ArgumentParser(description='Infernece scripts')


#inference args.
parser.add_argument('--exp_id', default='exp6', type=str, help='id of your current experiments')
parser.add_argument('--topk', type=int, default=2, metavar='N',
                    help='how many experts you want?')
parser.add_argument('--ensemble_inference', action='store_true', default=False, help='inference with all experts')
###############################################################################


parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=80, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')



# dataset and loaders stuff.
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N', help='test batchsize')
parser.add_argument('-dp', '--dataset_path', default='/path/to/dataset', type=str)
parser.add_argument('-name', '--data_name', default='cifar100', type=str)
parser.add_argument('-save_log_path', '--save_log_path', default='./logs/', type=str)


# Paths
parser.add_argument('-cp', '--checkpoint_path', default='checkpoint_experts', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint_experts)')
parser.add_argument('-router_cp', '--router_cp', default='workspace/c10_preresnet_run1/model_best.pth.tar', type=str, metavar='PATH',
                    help='checkpoint path of the router weight')
parser.add_argument('-router_cp_icc', '--router_cp_icc', default='workspace/c10_preresnet_run1/model_best.pth.tar', type=str, metavar='PATH',
                    help='checkpoint path of the router weight for icc. We eval. router train on partial set of train data for ICC calculation.')

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


# one expert
parser.add_argument('--one_expert', action='store_true', default=True, help='only one expert inference')

args = parser.parse_args()


log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
os.makedirs(args.save_log_path, exist_ok=True)
fh = logging.FileHandler(os.path.join(args.save_log_path, 'msnet.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)
    

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
            try:
                chk_path = os.path.join("workspace", args.data_name, args.exp_id, args.checkpoint_path, loi + '.pth')
                chk = torch.load(chk_path)
                #chk_path = os.path.join("workspace", args.data_name, args.exp_id, args.checkpoint_path, loi + '.pth.tar')
            except KeyError as KE:
                error_str = f"Load error for expert {loi}"
                chk_path = os.path.join("workspace", args.data_name, args.exp_id, args.checkpoint_path, loi + '.pth.tar')
                chk = torch.load(chk_path)
                logging.info(error_str)
            except Exception as e:
                error_str = f"Load error for expert {loi}"
                chk_path = os.path.join("workspace", args.data_name, args.exp_id, args.checkpoint_path, loi + '.pth.tar')
                chk = torch.load(chk_path)
                logging.info(error_str)
            
            try:
                experts[loi].load_state_dict(chk['net'])
            except Exception as e:
                experts[loi].load_state_dict(chk['state_dict'])
    
    return experts


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
        try:
            model.load_state_dict(chk['net'], strict=False)
        except Exception as e:
            model.load_state_dict(chk['state_dict'])
        logging.info(f"Loading router from ckpt path: {ckpt_path}")
    
    return model


def ensemble_inference(test_loader, experts, router):
    router.eval()
    experts_on_stack = []
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
    
    losses = AverageMeter()
    top1   = AverageMeter()
    top2   = AverageMeter()
    for dta, target in tqdm(test_loader):
        dta, target = dta.cuda(), target.cuda()
        list_of_experts = [] 
        for exp in experts_on_stack:
            list_of_experts.append(experts[exp])
        with torch.no_grad():
            all_outputs = [exp_(dta) for exp_ in list_of_experts]
            output = router(dta)
        all_outputs.append(output)

        all_outputs_avg = average(all_outputs)
        all_output_prob = F.softmax(all_outputs_avg, dim=1)
        loss = F.cross_entropy(all_output_prob, target).item() # sum up batch loss
        prec1, prec2 = accuracy(all_output_prob, target, topk=(1,2))
        losses.update(loss, dta.size(0))
        top1.update(prec1.item(), dta.size(0))
        top2.update(prec2.item(), dta.size(0))

    f_l = [losses.avg, top1.avg, top2.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@2: {:.2f}'.format(*f_l))
    return top1.avg, top2.avg


def inference_with_experts_and_routers(test_loader, experts, router, topk=2, total_data = 10000, temp_loader=None, topk_experts=None):
    """ function to perform evaluation with experts
    
    Args:
        test_loader (_type_): _description_
        experts (_type_): _description_
        router (_type_): _description_
        topk (int, optional): _description_. Defaults to 2.
        temp_loader (_type_, optional): _description_. Defaults to None.
    """
    freqMat = np.zeros((100, 100)) # -- debug
    router.eval()
    experts_on_stack = []
    expert_count = {} 
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
        expert_count[k] = 0
        # test_expert(experts[k], temp_loader[k])
        # test_expert(router, temp_loader[k])

    count = 0
    avg_experts_usage = 0
    correct = 0
    by_router = 0

    for dta, target in tqdm(test_loader):
        count += 1
        dta, target = dta.cuda(), target.cuda()
        with torch.no_grad():
            # you can feed ensemble predictor here to make comparisons
            output_raw = router(dta)
        output = F.softmax(output_raw, dim=1)
        router_confs, router_preds = torch.sort(output, dim=1, descending=True)
        preds = []
        confs = []
        for k in range(0, topk):
            #ref = torch.argsort(output, dim=1, descending=True)[0:, k]
            ref = router_preds[0:, k]
            conf = router_confs[0:, k]
            preds.append(ref.detach().cpu().numpy()[0]) # simply put the number. not the graph
            confs.append(conf.detach().cpu().numpy()[0])
    
        experts_output = []

        list_of_experts = []
        visited = {str(cls_): 0 for cls_ in range(0, 300+1)}

        args.one_expert = False
        if (args.one_expert):
            for exp in experts_on_stack: #
                exp_cls = exp.split("_")
                if (exp not in list_of_experts and  (str(preds[0]) in exp_cls and str(preds[1]) )):#) in exp_cls and str(preds[2] in exp_cls) ) ): # and not visited[str(r_pred)]):
                    list_of_experts.append(exp)
                    expert_count[exp] += 1
                    avg_experts_usage += 1
                    # visited[str(r_pred)] = 1
                    break
        
        else:
            target_string = str(target.cpu().numpy()[0])#  --> to verfiy
            for exp in experts_on_stack: #
                exp_cls = exp.split("_")
                for r_pred in preds:
                    if (str(r_pred) in exp_cls and exp not in list_of_experts):# and str(r_pred) in target_string):# and not visited[str(r_pred)]):
                        list_of_experts.append(exp)
                        expert_count[exp] += 1
                        avg_experts_usage += 1
                        visited[str(r_pred)] = 1
                        break

        # trime the list of experts to only top-k experts. 
        # if (len(list_of_experts) > topk_experts):
        #     list_of_experts = random.choices(list_of_experts, k=topk_experts) 
        #list_of_experts = random.choices(list_of_experts, k = 5)
        with torch.no_grad():
            experts_output = [experts[exp_](dta)/3.01 for exp_ in list_of_experts]
        experts_output.append(output_raw) # (1 + math.e**(-1/len(exp_.split("_"))))
        experts_output_avg = average(experts_output)
        exp_conf, exp_pred = torch.sort(experts_output_avg, dim=1, descending=True)
        #print (f"List of Experts: {list_of_experts}, Expert prediction: {exp_pred[0:, 0]}, router pred: {preds}, target: {target}")
        pred_t1, pred_t2 = exp_pred[0:, 0], exp_pred[0:, 1]

        if (pred_t1.cpu().numpy()[0] == target.cpu().numpy()[0]):
            correct += 1
        if (preds[0] == target.cpu().numpy()[0]): #or preds[1] == target.cpu().numpy()[0]):
            by_router += 1
            
       
    # print (f"** Expert dict: {expert_count}")
    #print (f"** MS-NET samples: {correct} \n, ** Router acc: {by_router} \n")
    print (f"** MS-NET acc: {correct} \n** Router acc: {by_router}\n")
    print (f"** Average exp. usage: {avg_experts_usage/total_data}")
    print (f"Total data: {total_data}")


def main():
    """run simple inference """
    _, test_loader_router, test_loader_single, _, num_classes, list_of_classes = get_dataloader(
        data_name=args.data_name,
        dataset_path=args.dataset_path,
        TRAIN_BATCH=args.train_batch,
        TEST_BATCH=args.test_batch)
    total_data = len(test_loader_router.dataset)
    logging.info("==> creating standalone router model")
    router = make_router(num_classes, ckpt_path=args.router_cp)
    list_of_experts = os.listdir(os.path.join("workspace", args.data_name, args.exp_id, "checkpoint_experts"))
    split_f = lambda x: x.split(".")[0]
    lois = [split_f(index_) for index_ in list_of_experts]
    # lois = lois[: 10]
    # class_count_dict = {str(cls): 0 for cls in range(0, 201)}
    # for loi in lois:
    #     expert_cls = loi.split("_")
    #     for cls in expert_cls:
    #         class_count_dict[cls] += 1
    # class_count_dict = {k:v for k, v in class_count_dict.items() if v > 0}
    # print (len(class_count_dict))

    msnet = load_experts(num_classes, list_of_index=lois, pretrained=True)
    if (args.ensemble_inference):
        ensemble_inference(test_loader_router, msnet, router)
    else:
        inference_with_experts_and_routers(test_loader_single, msnet, router, total_data=total_data, topk=args.topk, topk_experts=None)


if __name__ == '__main__':
    main() 
