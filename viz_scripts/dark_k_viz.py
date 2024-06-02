import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
# from msnet import MSNET
# from ..models import cifar as models
import models.cifar as models
from utils.ms_net_utils import *
from utils.data_utils import *
from utils.basic_utils import load_pretrained_model
import argparse

parser = argparse.ArgumentParser(description='Stable MS-NET')

# parser.add_argument('--')


list_of_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
sample_path = './data/test_samples/'
list_of_images =  glob.glob(sample_path + "*.png")
# candidate classes for testing: leopard,
 
CKPT_PATH = 'workspace/cifar100/cifar100_resnet20_ensemble/checkpoint_experts/set1.pth.tar'
net = models.__dict__['resnet'](
    num_classes=100,
    depth=20,
    block_name='BasicBlock')
net.cuda()

try:
    ckpt_pth = torch.load(CKPT_PATH)
    net.load_state_dict(ckpt_pth['net'], strict=True)
except Exception as e:
    print (f"INFO: Load error")

# transform for images.
transform_test = transforms.Compose([
        #transforms.PILToTensor(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

net.eval()
for img in list_of_images:
    with torch.no_grad():
        raw_image = Image.open(img)
        torch_image = transform_test(raw_image)
        torch_image = torch_image.cuda()
        torch_image = torch.unsqueeze(torch_image, dim=0)
        output = net(torch_image)
        # output = F.softmax(output, dim=1)
    output = torch.sort(output, descending=True, dim=1)
    print (f"top-1: {list_of_classes[output[1][0][0]]}, top-2: {list_of_classes[output[1][0][1]]}")
    print (f"softmax-top-1: {output[0][0][0]}, softmax-top-2: {output[0][0][1]}")
    