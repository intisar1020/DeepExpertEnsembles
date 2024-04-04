from cProfile import label
from enum import unique
import torch
import argparse
import models.cifar as models
from utils.ms_net_utils import *
from utils.data_utils import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

parser = argparse.ArgumentParser(description='Stable MS-NET')
# dataset paths
parser.add_argument('-dp', '--dataset_path', default='/path/to/dataset', type=str)
parser.add_argument('-save_log_path', '--save_log_path', default='./logs/', type=str)
parser.add_argument('-name', '--data_name', default='cifar100', type=str)

# Architecture details
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help='backbone architecture')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='LR',
                    help='initial learning rate to train')

# router checkpoints
parser.add_argument('-router_cp_icc', '--router_cp_icc', default='workspace/pre-trained_wts/resnet20_icc/model_best.pth.tar', type=str, metavar='PATH',
                    help='checkpoint path of the router weight for icc. We eval. router train on partial set of train data for ICC calculation.')




args = parser.parse_args()


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
            model.load_state_dict(chk['state_dict'])
            print(f"Loading router from ckpt path: {ckpt_path}")
        except Exception as e:
            model.load_state_dict(chk['net'])
            print(f"Loading router from ckpt path: {ckpt_path}")
    
    return model


def main():
    _, _, _, val_loader_single, num_classes, _ = get_dataloader(
    data_name=args.data_name,
    dataset_path=args.dataset_path,
    TRAIN_BATCH=32, 
    TEST_BATCH=32)

    router_icc = make_router(num_classes, ckpt_path=args.router_cp_icc)
    matrix = calculate_matrix(router_icc, val_loader_single, num_classes, only_top2=True)
    binary_list, _, dict_ = return_topk_args_from_heatmap(matrix, num_classes, cutoff_thresold=1, binary_=False)
    new_dict = {} # reduce histrogram container for ease of visuilization.
    tot = 0
    for k, v in dict_.items():
        new_dict[k] = v
        tot += 1
        if (tot == 50):
            break

    ls = np.arange(num_classes)
    heatmap(matrix, ls, ls) # show heatmap
    barchart(dict_) # show barchart for ICC pair and corresponding number of
    barchart(new_dict)
    binary_unique_classes_stride = []
    uniqueClasses = []
    noofuniqueclasses = []
    strides = []
    for strd in range(10, 400, 10):
        for bl in binary_list[: strd]:
            uniqueClasses.extend(bl)
        noofuniqueclasses.append(len(set(uniqueClasses)))
        uniqueClasses = []
        strides.append(strd)

    fig, ax = plt.subplots()
    ax.set_xlabel('no. of binary subsets.')
    ax.set_ylabel('total classes covered (CIFAR-100)')
    #ax1.plot(cut_off, router, color=color)
    ax.plot(strides, noofuniqueclasses, linestyle='dashed', linewidth=1, marker='s')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(linestyle='--', linewidth=1)
    #plt.legend()
    #plt.show()

    binary_list_count_per_co = []
    super_list_count_per_co = []
    cos = [] 
    for co in range(1, 20):
        binary_list, super_list, dict_ = return_topk_args_from_heatmap(matrix, num_classes, cutoff_thresold=co, binary_=False)        
        binary_list_count_per_co.append(len(binary_list))
        super_list_count_per_co.append(len(super_list))
        cos.append(co)
        # uniqueClassesBinary = []
        # uniqueClassesSuper = []
        # for bl in binary_list:
        #     uniqueClassesBinary.extend(bl)
        # uniqueClassesBinary = set(uniqueClassesBinary)
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('no. of level-2 subsets', color=color)  # we already handled the x-label with ax1
    ax2.plot(cos, super_list_count_per_co, linestyle='dashed', linewidth=1, marker='d', color=color, label='level-2 subset')
    # ax2.tick_params(axis='y', labelcolor=color)

    ax1.plot(cos, binary_list_count_per_co, linestyle='dashed', linewidth=1, marker='s', label='binary subset')
    ax1.set_xlabel('cut-off variable (CIFAR-100)')
    ax1.set_ylabel('no. of binary subsets')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(linestyle='--', linewidth=1)
    ax1.legend()
    ax2.legend()
    plt.show()
    

main()