import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data import (DataLoader, SubsetRandomSampler,
                              WeightedRandomSampler)

import transforms


def get_train_transforms(data_name="none"):
    if "cifar" in data_name:
        print("Train DataLoader for CIFAR-10/100")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3, ),
        ])
    elif "imagenet" in data_name:
        print("Train DataLoader for Tiny-ImageNet")
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3, ),
        ])
    elif "pets" in data_name:
        print("Train DataLoader for Pets")
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3, ),
        ])
    elif "mnist" in data_name:
        print("Train DataLoader for MNIST")
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])
    else:
        print(" *************** -- returning default transforms -- ***************")
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform_train


def get_test_transforms(data_name="cifar100"):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if "cifar" in data_name or "imagenet" in data_name:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if "pets" in data_name:
        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if "mnist" in data_name:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])
    return transform_test


def get_dataloader(
        data_name="cifar100",
        dataset_path='data/cifar100png',
        TRAIN_BATCH=96,
        TEST_BATCH=96
    ):
    print(f'==> Preparing dataset from {dataset_path} for {data_name}')

    trainset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "train"),
        transform=get_train_transforms(data_name=data_name)
    )
    testset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "test"),
        transform=get_test_transforms(data_name=data_name)
    )
    valset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "test"),
        transform=get_test_transforms(data_name=data_name)
    )

    num_classes = len(trainset.classes)
    list_of_classes = trainset.classes
    train_loader = DataLoader(trainset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=TEST_BATCH, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_single = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    val_loader_single = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader, test_loader_single, val_loader_single, num_classes, list_of_classes


def expert_dataloader(
        data_name="none",
        dataset_path="path/to/dataset",
        matrix=[],
        TRAIN_BATCH=128,
        TEST_BATCH=128,
        weighted_sampler=True,
        p_beta=2
    ):
    print('==> Preparing dataset %s' % data_name)

    if data_name == 'fmnist':
        dataloader = datasets.FashionMNIST
    if data_name == 'svhn':
        dataloader = datasets.SVHN

    if data_name == 'svhn':
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

    if data_name == 'svhn':
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

    if weighted_sampler:
        print("***************Preparing weighted Sampler ********************************")
        print(f"Probability P(x=1) == {p_beta}")
        for sub in matrix:
            weight = class_sample_count / class_sample_count
            for sb in sub:
                weight[sb] *= p_beta
            samples_weight = np.array([weight[t] for t in train_set.targets])
            samples_weight = torch.from_numpy(samples_weight)
            sampler_ = WeightedRandomSampler(samples_weight, len(samples_weight))

            index = "_".join(map(str, sub))

            train_loader_expert[index] = torch.utils.data.DataLoader(
                train_set,
                batch_size=TRAIN_BATCH,
                sampler=sampler_,
                num_workers=4,
                pin_memory=True
            )

            indices_test = [j for j, k in enumerate(test_set.targets) if k in sub]
            test_loader_expert[index] = torch.utils.data.DataLoader(
                test_set,
                batch_size=TEST_BATCH,
                sampler=SubsetRandomSampler(indices_test),
                num_workers=4,
                pin_memory=True
            )
            list_of_index.append(index)
        return train_loader_expert, test_loader_expert, list_of_index
    else:
        print("***************Preparing Subset Sampler ********************************")
        for sub in matrix:
            if data_name == 'svhn':
                indices_train = [i for i, e in enumerate(train_set.labels) if e in sub]
                indices_test = [j for j, k in enumerate(test_set.labels) if k in sub]
            else:
                indices_train = [i for i, e in enumerate(train_set.targets) if e in sub]
                indices_test = [j for j, k in enumerate(test_set.targets) if k in sub]

            index = "_".join(map(str, sub))

            train_loader_expert[index] = torch.utils.data.DataLoader(
                train_set,
                batch_size=TRAIN_BATCH,
                sampler=SubsetRandomSampler(indices_train),
                num_workers=4,
                pin_memory=True
            )

            test_loader_expert[index] = torch.utils.data.DataLoader(
                test_set,
                batch_size=TEST_BATCH,
                sampler=SubsetRandomSampler(indices_test),
                num_workers=4,
                pin_memory=True
            )
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
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.train_batch = TRAIN_BATCH
        self.test_batch = TEST_BATCH

    def _train_transforms(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3, ),
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
            transform=self._train_transforms()
        )
        testset = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, "test"),
            transform=self._test_transforms()
        )


class MSNetRouterDataLoader(MSNetBaseDataLoader):
    def __init__(
            self,
            dataset_name="cifar100",
            dataset_path="path",
            TRAIN_BATCH=128,
            TEST_BATCH=128
        ) -> None:
        super().__init__(dataset_name, dataset_path, TRAIN_BATCH, TEST_BATCH)