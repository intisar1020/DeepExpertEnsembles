import os
from symbol import pass_stmt
import torch
import numpy as np
import transforms
from torchvision import datasets#, transforms
import torch.utils.data as data
from torch.utils.data import  SubsetRandomSampler, WeightedRandomSampler
# import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_train_transforms(data_name="cifar100"):
    """_summary_

    Args:
        data_name (str, optional): _description_. Defaults to "cifar100".

    Returns:
        _type_: _description_
    """

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])  
    
    if ("cifar" in data_name):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),])
        
    if ("imagenet" in data_name):
         transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),])


    if ("mnist" in data_name):
         transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
  

    return transform_train
    

def get_test_transforms(data_name="cifar100"):
    """_summary_

    Args:
        data_name (str, optional): _description_. Defaults to "cifar100".

    Returns:
        _type_: _description_
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


    if ("cifar" in data_name or "imagenet" in data_name):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    if ("mnist" in data_name):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
           
    return transform_test


def get_dataloader(
        data_name="cifar100",
        dataset_path='data/cifar100png', 
        TRAIN_BATCH=96, 
        TEST_BATCH=96):
    """_summary_

    Args:
        dataset_path (str, optional): _description_. Defaults to 'data/cifar100png'.
        TRAIN_BATCH (int, optional): _description_. Defaults to 96.
        TEST_BATCH (int, optional): _description_. Defaults to 96.

    Returns:
        _type_: _description_
    """

    print(f'==> Preparing dataset from {dataset_path} for {data_name}')

    # training dataset
    trainset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "train"),
        transform=get_train_transforms(data_name=data_name)
        )
    testset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "test"),
        transform=get_test_transforms(data_name=data_name)
        )
    
    valset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "val"),
        transform=get_test_transforms(data_name=data_name)
        )

    num_classes = len(trainset.classes)
    list_of_classes = trainset.classes
    train_loader = DataLoader(trainset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=TEST_BATCH, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_single = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    val_loader_single = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, test_loader_single, val_loader_single, num_classes, list_of_classes


def expert_dataloader(
        data_name="cifar100",
        dataset_path="path/to/dataset",
        matrix=[], 
        TRAIN_BATCH=128, 
        TEST_BATCH=128,
        weighted_sampler=True):
    """_summary_

    Args:
        data_name (str, optional): _description_. Defaults to "cifar100".
        dataset_path (str, optional): _description_. Defaults to "path/to/dataset".
        matrix (list, optional): _description_. Defaults to [].
        TRAIN_BATCH (int, optional): _description_. Defaults to 128.
        TEST_BATCH (int, optional): _description_. Defaults to 128.
        weighted_sampler (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    print('==> Preparing dataset %s' % data_name)
    
    # if data_name == 'cifar10':
    #     dataloader = datasets.CIFAR10
    #     num_classes = 10
    
    if data_name == 'fmnist':
        dataloader = datasets.FashionMNIST
        num_classes = 10
    
    if data_name == 'svhn':
        dataloader = datasets.SVHN
        num_classes = 10

    
    if (data_name == 'svhn'):
        train_set = dataloader(root='./data', split='train', download=True, transform=get_train_transforms(data_name=data_name))
        test_set = dataloader(root='./data', split='test', download=True, transform=get_test_transforms(data_name=data_name))

    else:
        train_set = datasets.ImageFolder(
            root=os.path.join(dataset_path, "train"),
            transform=get_train_transforms(data_name=data_name)
            )
        test_set = datasets.ImageFolder(
            root=os.path.join(dataset_path, "test"),
            transform=get_test_transforms(data_name=data_name)
            )
        
    if (data_name == 'svhn'):
        class_sample_count = np.array(
            [len(np.where(train_set.labels == t)[0])
             for t in np.unique(train_set.labels)])
    else:
        class_sample_count = np.array(
            [len(np.where(train_set.targets == t)[0])
             for t in np.unique(train_set.targets)])


    train_loader_expert = {}
    test_loader_expert = {}
    list_of_index = []
 
    if (weighted_sampler):
        print ("***************Preparing weighted Sampler ********************************")
        for sub in matrix: # for each confusing class pair
            weight = class_sample_count / class_sample_count
            for sb in sub:
                weight[sb] *= 3.5 # all exp done with 2
            print (weight)
            samples_weight = np.array([weight[t] for t in train_set.targets])
            samples_weight = torch.from_numpy(samples_weight)
            sampler_ = WeightedRandomSampler(samples_weight, len(samples_weight))
            index = ""
            
            for i, sb_ in enumerate(sub):
                index += str(sb_)
                if (i < len(sub)-1):
                    index += "_"
            
            train_loader_expert[index] = torch.utils.data.DataLoader(
                train_set,
                batch_size=TRAIN_BATCH,
                sampler = sampler_)
            
            # for test only sample corresponding expert dataset.
            indices_test = [j for j,k in enumerate(test_set.targets) if k in sub] # sub is a list of classes like [35, 98]
            
            test_loader_expert[index] = torch.utils.data.DataLoader(
                test_set,
                batch_size=TEST_BATCH,
                sampler = SubsetRandomSampler(indices_test))
            list_of_index.append(index)
        
        return train_loader_expert, test_loader_expert, list_of_index
 
    # if subsetRandomSampler
    else:
        for sub in matrix:
            if (data_name == 'svhn'):
                indices_train = [i for i,e in enumerate(train_set.labels) if e in sub] 
                indices_test = [j for j,k in enumerate(test_set.labels) if k in sub]
            else:
                indices_train = [i for i,e in enumerate(train_set.targets) if e in sub] 
                indices_test = [j for j,k in enumerate(test_set.targets) if k in sub]
            index = ""
            for i, sb_ in enumerate(sub):
                index += str(sb_)
                if (i < len(sub)-1):
                    index += "_"
            train_loader_expert[index] = torch.utils.data.DataLoader(
                train_set,
                batch_size=TRAIN_BATCH,
                shuffle=True,
                sampler = SubsetRandomSampler(indices_train))
            test_loader_expert[index] = torch.utils.data.DataLoader(
                test_set,
                batch_size=TEST_BATCH,
                sampler = SubsetRandomSampler(indices_test))
            list_of_index.append(index)
    
        return train_loader_expert, test_loader_expert, list_of_index
    

class MSNetBaseDataLoader:

    def __init__(
            self,
            dataset_name="cifar100",
            dataset_path="path",
            TRAIN_BATCH=128,
            TEST_BATCH=128,
            ) -> None:
        """_summary_

        Args:
            dataset_name (str, optional): _description_. Defaults to "cifar100".
            dataset_path (str, optional): _description_. Defaults to "path".
            TRAIN_BATCH (int, optional): _description_. Defaults to 128.
            TEST_BATCH (int, optional): _description_. Defaults to 128.
        """
        
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.train_batch = TRAIN_BATCH
        self.test_batch = TEST_BATCH
    
    def _train_transforms(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),
        ])
        return transform_train
    
    def _test_transforms(self):
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform_test
    
    def get_data_loader(self):
        
        trainset = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, "train"),
            transform=self._train_transforms())
        
        testset = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, "test"),
            transform=self._test_transforms())
        


class MSNetRouterDataLoader(MSNetBaseDataLoader):
    def __init__(self, dataset_name="cifar100", dataset_path="path", TRAIN_BATCH=128, TEST_BATCH=128) -> None:
        super().__init__(dataset_name, dataset_path, TRAIN_BATCH, TEST_BATCH)