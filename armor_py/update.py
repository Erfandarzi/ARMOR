import numpy as np
import torch
from sklearn import metrics
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, device):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(device)
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split train and test
        idxs_train = idxs
        if (self.args.dataset == 'mnist') or (self.args.dataset == 'cifar'):
            idxs_test = idxs
            train = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.batch_size, shuffle=True)
            test = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)), shuffle=False)
        else:
            train = self.args.dataset_train[idxs]
            test = self.args.dataset_test[idxs]
        return train, test

    def update_weights(self, net, device):
        net.to(device)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0)

        epoch_loss = []
        epoch_acc = []
        for iter in range(self.args.epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = autograd.Variable(images), autograd.Variable(labels)
                images = images.to(device)
                labels = labels.to(device)
                net.zero_grad()

                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.args.clip_threshold)
                optimizer.step()
                batch_loss.append(loss.data.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            acc, _, = self.test(net, device=device)
            epoch_acc.append(acc)
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_acc = sum(epoch_acc) / len(epoch_acc)
        w = net.state_dict()
        return w, avg_loss, avg_acc

    def test(self, net, device):
        loss = 0
        log_probs = []
        labels = []
        net.to(device)
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            images, labels = autograd.Variable(images).to(device), autograd.Variable(labels).to(device)
            images = images.to(device)
            labels = labels.to(device)
            net = net.float()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
        y_pred = np.argmax(log_probs.cuda().data.cpu(), axis=1)
        acc = metrics.accuracy_score(y_true=labels.cuda().data.cpu(), y_pred=y_pred)
        loss = loss.cuda().data.cpu().item()
        return acc, loss


class ServerRetrain(object):
    def __init__(self, args, ae_images, ae_labels, device):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(device)
        self.ae_images = ae_images
        self.ae_labels = ae_labels

    def update_weights(self, net, device):
        net.to(device)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0)

        epoch_loss = []
        epoch_acc = []
        for iter in range(self.args.retrain_round):
            batch_loss = []
            for i in range(int(self.ae_images.size(0)/self.args.batch_size)):
                images = autograd.Variable(self.ae_images[self.args.batch_size*i : self.args.batch_size*(i+1)])
                labels = autograd.Variable(self.ae_labels[self.args.batch_size*i : self.args.batch_size*(i+1)])
                images = images.to(device)
                labels = labels.to(device)
                net.zero_grad()

                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.args.clip_threshold)
                optimizer.step()
                batch_loss.append(loss.data.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            acc, _, = self.test(net, device=device)
            epoch_acc.append(acc)
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_acc = sum(epoch_acc) / len(epoch_acc)
        w = net.state_dict()
        return w, avg_loss, avg_acc

    def test(self, net, device):
        loss = 0
        log_probs = []
        labels = []
        net.to(device)
        images = autograd.Variable(self.ae_images)
        labels = autograd.Variable(self.ae_labels)
        images = images.to(device)
        labels = labels.to(device)
        net = net.float()
        log_probs = net(images)
        loss = self.loss_func(log_probs, labels)
        y_pred = np.argmax(log_probs.cuda().data.cpu(), axis=1)
        acc = metrics.accuracy_score(y_true=labels.cuda().data.cpu(), y_pred=y_pred)
        loss = loss.cuda().data.cpu().item()
        return acc, loss
