import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import warnings
warnings.filterwarnings("ignore", message=".*NNPACK.*")

import argparse
import logging
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import models.cifar as models
from utils.basic_utils import AverageMeter, accuracy
from utils.data_utils import get_dataloader

parser = argparse.ArgumentParser(description='Inference scripts')

parser.add_argument('--exp_id', default='exp6', type=str, help='id of your current experiments')
parser.add_argument('--topk', type=int, default=2, metavar='N', help='how many experts you want?')
parser.add_argument('--ensemble_inference', action='store_true', default=False, help='inference with all experts')

parser.add_argument('--cuda', action='store_true', default=True, help='enable CUDA training')
parser.add_argument('--seed', type=int, default=80, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')

parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N', help='test batchsize')
parser.add_argument('-dp', '--dataset_path', default='/path/to/dataset', type=str)
parser.add_argument('-name', '--data_name', default='cifar100', type=str)
parser.add_argument('-save_log_path', '--save_log_path', default='./logs/', type=str)

parser.add_argument('-cp', '--checkpoint_path', default='checkpoint_experts', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint_experts)')
parser.add_argument('-router_cp', '--router_cp', default='workspace/c10_preresnet_run1/model_best.pth.tar', type=str, metavar='PATH', help='checkpoint path of the router weight')
parser.add_argument('-router_cp_icc', '--router_cp_icc', default='workspace/c10_preresnet_run1/model_best.pth.tar', type=str, metavar='PATH', help='checkpoint path of the router weight for icc. We eval. router train on partial set of train data for ICC calculation.')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help='backbone architecture')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='LR', help='initial learning rate to train')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-gpu', '--gpu_id', default=0, type=str, help='set gpu number')
parser.add_argument('--one_expert', action='store_true', default=True, help='only one expert inference')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
os.makedirs(args.save_log_path, exist_ok=True)
# name the log file based on date and time stamp
logfilename = 'msnet_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
fh = logging.FileHandler(os.path.join(args.save_log_path, logfilename))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def average(outputs):
    return sum(outputs) / len(outputs)

def load_experts(num_classes, list_of_index=[None], pretrained=True):
    experts = {}
    for loi in list_of_index:
        experts[loi] = models.__dict__[args.arch](num_classes=num_classes, depth=args.depth, block_name=args.block_name)
        if torch.cuda.is_available():
            experts[loi] = experts[loi].cuda()

        if pretrained:
            chk_path = os.path.join("workspace", args.data_name, args.exp_id, args.checkpoint_path, loi)
            try:
                chk = torch.load(f"{chk_path}.pth", map_location='cpu')
            except FileNotFoundError:
                chk = torch.load(f"{chk_path}.pth.tar", map_location='cpu')

            try:
                experts[loi].load_state_dict(chk['net'])
            except (KeyError, TypeError):
                experts[loi].load_state_dict(chk['state_dict'])
    return experts


def make_router(num_classes, ckpt_path=None):
    model = models.__dict__[args.arch](num_classes=num_classes, depth=args.depth, block_name=args.block_name)
    if torch.cuda.is_available():
        model = model.cuda()
    if ckpt_path:
        chk = torch.load(ckpt_path, map_location='cpu')
        try:
            model.load_state_dict(chk['net'], strict=False)
        except (KeyError, TypeError):
            model.load_state_dict(chk['state_dict'])
        logging.info(f"Loading router from ckpt path: {ckpt_path}")
    return model


def ensemble_inference(test_loader, experts, router):
    router.eval()
    experts_on_stack = []
    for k in experts:
        experts[k].eval()
        experts_on_stack.append(k)

    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    for dta, target in tqdm(test_loader):
        if torch.cuda.is_available():
            dta, target = dta.cuda(), target.cuda()
        list_of_experts = [experts[exp] for exp in experts_on_stack]
        with torch.no_grad():
            all_outputs = [exp_(dta) for exp_ in list_of_experts]
            output = router(dta)
        all_outputs.append(output)

        all_outputs_avg = average(all_outputs)
        all_output_prob = F.softmax(all_outputs_avg, dim=1)
        loss = F.cross_entropy(all_output_prob, target).item()
        prec1, prec2 = accuracy(all_output_prob, target, topk=(1, 2))
        losses.update(loss, dta.size(0))
        top1.update(prec1.item(), dta.size(0))
        top2.update(prec2.item(), dta.size(0))

    f_l = [losses.avg, top1.avg, top2.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@2: {:.2f}'.format(*f_l))
    return top1.avg, top2.avg


def inference_with_experts_and_routers(test_loader, experts, router, topk=2, total_data=10000):
    router.eval()
    experts_on_stack = []
    expert_count = {}
    for k in experts:
        experts[k].eval()
        experts_on_stack.append(k)
        expert_count[k] = 0

    count = 0
    avg_experts_usage = 0
    correct = 0
    by_router = 0

    for dta, target in tqdm(test_loader):
        count += 1
        if torch.cuda.is_available():
            dta, target = dta.cuda(), target.cuda()
        with torch.no_grad():
            output_raw = router(dta)
        output = F.softmax(output_raw, dim=1)
        _, router_preds = torch.sort(output, dim=1, descending=True)
        preds = [router_preds[0:, k].detach().cpu().numpy()[0] for k in range(topk)]

        list_of_experts = []
        args.one_expert = False
        if args.one_expert:
            for exp in experts_on_stack:
                exp_cls = exp.split("_")
                if str(preds[0]) in exp_cls and str(preds[1]):
                    list_of_experts.append(exp)
                    expert_count[exp] += 1
                    avg_experts_usage += 1
                    break
        else:
            for exp in experts_on_stack:
                exp_cls = exp.split("_")
                for r_pred in preds:
                    if str(r_pred) in exp_cls and exp not in list_of_experts:
                        list_of_experts.append(exp)
                        expert_count[exp] += 1
                        avg_experts_usage += 1
                        break

        with torch.no_grad():
            experts_output = [experts[exp_](dta) / 3.01 for exp_ in list_of_experts]
        experts_output.append(output_raw)
        experts_output_avg = average(experts_output)
        _, exp_pred = torch.sort(experts_output_avg, dim=1, descending=True)
        pred_t1 = exp_pred[0:, 0]

        if pred_t1.cpu().numpy()[0] == target.cpu().numpy()[0]:
            correct += 1
        if preds[0] == target.cpu().numpy()[0]:
            by_router += 1

    print(f"** MS-NET acc: {correct} \n** Router acc: {by_router}\n")
    print(f"** Average exp. usage: {avg_experts_usage / total_data}")
    print(f"Total data: {total_data}")


def main():
    _, test_loader_router, test_loader_single, _, num_classes, _ = get_dataloader(
        data_name=args.data_name,
        dataset_path=args.dataset_path,
        TRAIN_BATCH=args.train_batch,
        TEST_BATCH=args.test_batch
    )
    total_data = len(test_loader_router.dataset)
    logging.info("==> creating standalone router model")
    router = make_router(num_classes, ckpt_path=args.router_cp)
    list_of_experts = os.listdir(os.path.join("workspace", args.data_name, args.exp_id, "checkpoint_experts"))
    split_f = lambda x: x.split(".")[0]
    lois = [split_f(index_) for index_ in list_of_experts]

    msnet = load_experts(num_classes, list_of_index=lois, pretrained=True)
    if args.ensemble_inference:
        ensemble_inference(test_loader_router, msnet, router)
    else:
        inference_with_experts_and_routers(
            test_loader_single, msnet, router, total_data=total_data, topk=args.topk
        )


if __name__ == '__main__':
    main()
