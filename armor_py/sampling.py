import numpy as np
import torch
from easydict import EasyDict
from torchvision import datasets, transforms


def ld_cifar10():
    test_transforms = transforms.ToTensor()
    test_dataset = datasets.CIFAR10('./dataset/cifar10/', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return EasyDict(test=test_loader)


def ld_mnist():
    test_transforms = transforms.ToTensor()
    test_dataset = datasets.MNIST('./dataset/mnist/', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return EasyDict(test=test_loader)


def mnist_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    return dict_users


def mnist_noniid(dataset, num_users):
    if num_users >= 25:
        num_shards = int(300)
    else:
        num_shards = int(200)
    num_imgs = int(dataset.data.shape[0] / num_shards)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    print(idxs)
    if num_users >= 25:
        chosen_shards = int(3)
    else:
        chosen_shards = int(5)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    return dict_users


def cifar_noniid(dataset, num_users):
    if num_users >= 25:
        num_shards = int(200)
    else:
        num_shards = int(100)
    num_imgs = int(dataset.data.shape[0] / num_shards)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    if num_users >= 25:
        chosen_shards = int(4)
    else:
        chosen_shards = int(6)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users