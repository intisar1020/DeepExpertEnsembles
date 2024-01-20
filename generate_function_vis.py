# generated graphs.

# System
from math import sqrt
import random
from numpy.linalg import norm
import copy
import argparse
import os
from re import split
import sys
import logging
import time

# progress bar
from tqdm import tqdm

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

# graphs, plots.
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Infernece scripts')


#inference args.
parser.add_argument('--exp_id', default='exp6', type=str, help='id of your current experiments')
parser.add_argument('-name', '--data_name', default='cifar100', type=str)
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



# data loaders stuff.
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')


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
            try:
                chk_path = os.path.join("workspace", args.data_name, args.exp_id, args.checkpoint_path, loi + '.pth')
                chk = torch.load(chk_path)
                experts[loi].load_state_dict(chk['net'])
            except Exception as e:
                error_str = f"Load error for expert {loi}"
                logging.info(error_str)
    
    return experts


def evaluate(test_dataloader, model):
  true_labels = []
  pred_labels = []
  
  for (imgs, labels) in tqdm(test_dataloader):
    model.eval()
    imgs = imgs.cuda()
    labels = labels.cuda()
    with torch.no_grad():
        preds = model(imgs)
    
    labels = labels.detach().cpu().numpy()
    true_labels.extend(labels)
    preds = preds.detach().cpu().numpy()
    pred_labels.extend(np.argmax(preds, axis=1))

  return np.array(true_labels), np.array(pred_labels)


def calc_disgreement(test_loader, router, msnet, lois):
    predictions = []
    _, preds = evaluate(test_loader, router)
    predictions.append(preds)

    for elem in lois:
        model = msnet[elem]
        # get predictions for model
        _, preds = evaluate(test_loader, model)
        predictions.append(preds)

    empty_arr = np.zeros(shape=(1 + len(lois), 1 + len(lois)))
    
    for i in tqdm(range(1 + len(lois))):
        preds1 = predictions[i]
        for j in range(i, 1  + len(lois)):
            preds2 = predictions[j]
            # compute dissimilarity
            dissimilarity_score = 1-np.sum(np.equal(preds1, preds2))/10000 
            empty_arr[i][j] = dissimilarity_score
            if i is not j:
                empty_arr[j][i] = dissimilarity_score

    dissimilarity_coeff = empty_arr[::-1]
    plt.figure(figsize=(sqrt(len(lois) * 2), sqrt(len(lois) * 2)))#(9,8))
    sns.heatmap(dissimilarity_coeff, cmap='RdBu_r')
    axis = np.arange(0, len(lois) + 1)
    plt.xticks(axis,axis, rotation=45, rotation_mode="anchor")
    plt.yticks(axis,np.flip(axis), rotation=45, rotation_mode="anchor")
    plt.savefig('./prediction_disagreement.png')


# Get the weights of the input model. 
def get_model_weights(model):
  model_weights = []
  # iterate through model layers.
  for name_, wts in model.named_parameters():
    # grab weights of that layer
    weights = wts.detach().cpu().numpy() # list
    # check if layer got triainable weights
    if len(wts)==0:
      continue

    # discard biases term, wrap with ndarray, flatten weights
    model_weights.extend(np.array(weights[0]).flatten())

  return np.array(model_weights)
     

def weight_similarity(router, msnet, lois):
   weights_of_models = []
   weights = get_model_weights(router)
   weights_of_models.append(weights)
   for lo in lois:
      # load model
      model = msnet[lo]
      # get predictions for model
      weights = get_model_weights(model)
      weights_of_models.append(weights)
      
   empty_arr = np.zeros(shape=(1 + len(lois), 1 + len(lois)))
   for i in tqdm(range(1 + len(lois))):
        weights1 = weights_of_models[i]
        for j in range(i, 1  + len(lois)):
            weights2 = weights_of_models[j]
            # compute cosine similarity of weights
            cos_sim = np.dot(weights1, weights2)/(norm(weights1)*norm(weights2))
            empty_arr[i][j] = cos_sim
            if i is not j:
                empty_arr[j][i] = cos_sim
   cos_sim_coeff = empty_arr[::-1]
   plt.figure(figsize=(sqrt(len(lois) * 2), sqrt(len(lois) * 2)))#(9,8))
   sns.heatmap(cos_sim_coeff, cmap='RdBu_r')
   axis = np.arange(0, len(lois) + 1)
   plt.xticks(axis,axis, rotation=45, rotation_mode="anchor")
   plt.yticks(axis,np.flip(axis), rotation=45, rotation_mode="anchor")
   plt.savefig('./func_similarity.png')


def main():
    _, test_loader_router, test_loader_single, _, num_classes, list_of_classes = get_dataloader(
        data_name=args.data_name,
        dataset_path=args.dataset_path,
        TRAIN_BATCH=args.train_batch,
        TEST_BATCH=args.test_batch)
    
    _,  single_class_test_dataloaders, _ = expert_dataloader(
            data_name=args.data_name,
            dataset_path=args.dataset_path,
            matrix=[[i] for i in range(100)],
            TRAIN_BATCH=args.train_batch, 
            TEST_BATCH=1,
            weighted_sampler=True) 



    logging.info("==> creating standalone router model")
    router = make_router(num_classes, ckpt_path=args.router_cp)
    list_of_experts = os.listdir(os.path.join("workspace", args.data_name, args.exp_id, "checkpoint_experts"))
    split_f = lambda x: x.split(".")[0]
    lois = [split_f(index_) for index_ in list_of_experts if  "2" in split_f(index_)]
    msnet = load_experts(num_classes, list_of_index=lois, pretrained=True)
    calc_disgreement(test_loader_router, router, msnet, lois)
    #weight_similarity(router, msnet, lois)

if __name__ == '__main__':
    main() 