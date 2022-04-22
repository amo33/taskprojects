import torch 
import numpy as np 
from torch.utils.data import TensorDataset, DataLoader
import pickle 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from CIfar import CifarNet
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision import datasets 
from torchvision.datasets import ImageFolder
import time 
import PIL, os
from PIL import Image
from cifadataprocess import *
from pytorchtools import EarlyStopping
from pytorch_lightning.callbacks import EarlyStopping


PATH = './cifar_net.pth'

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124),(0.24703233,0.24348505,0.26158768)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124),(0.24703233,0.24348505,0.26158768))
    ]),
}

data_dir= 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

print(dataset_sizes)
net = CifarNet()

hypothesis = nn.CrossEntropyLoss()

#optimizer = optim.Adam(net.parameters(), lr=0.01)
optimizer = optim.SGD(net.parameters(), lr=0.01) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)




def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    early_stopping = EarlyStopping(patience = 7, verbose = True)
    since = time.time()

    best_ck = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train -> validation -> train -> ... To compare validation accuracy for model update 
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True) 
            else:
                model.train(False) 
                # can do model.eval()
            running_loss = 0.0
            running_corrects = 0


            for data in dataloaders[phase]:

                inputs, labels = data
                optimizer.zero_grad()
                
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # 학습 단계에서만 수행, 역전파 + 옵티마이즈(최적화)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 통계
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'valid':
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                # 가중치 저장 
                 
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_ck = model.state_dict()

        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # 최적의 모델 가중치 로딩
    model.load_state_dict(best_ck)

    torch.save(best_ck, PATH)


train_model(net, hypothesis,optimizer, scheduler, num_epochs = 25)