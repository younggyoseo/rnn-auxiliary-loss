import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def sequential_mnist(bsz, permute=False, valid_size=0.1,
                     valid_shuffle=False, random_seed=1004):
    data_dir = './data/mnist'

    if permute:
        perm = torch.randperm(784)
    else:
        perm = torch.arange(0, 784).long()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1/255,)),
        transforms.Lambda(lambda x: x.contiguous().long().view(-1).index_select(0, perm))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=transform)
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if valid_shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bsz, sampler=train_sampler, drop_last=True)
    
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bsz, sampler=valid_sampler, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bsz, drop_last=True)
    
    return train_loader, valid_loader, test_loader
