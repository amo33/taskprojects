from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
PATH = './cifar_net.pth'

def get_files_count(folder_path):
	dirListing = os.listdir(folder_path)
	return len(dirListing)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
      
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=data_transforms['train'])
val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=data_transforms['val'])
class_names = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataloaders = {}

dataloaders['train'] =torch.utils.data.DataLoader(train_set, batch_size=4,
                                             shuffle=True, num_workers=4)
dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=4,
                                             shuffle=False, num_workers=4)

use_gpu = torch.cuda.is_available()
dataset_sizes = {}
dataset_sizes['train'] = len(train_set)
dataset_sizes['val'] = len(val_set)
all_prediction = torch.tensor([]).to(DEVICE)
real_labels = torch.tensor([]).to(DEVICE)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.to(DEVICE))
                    labels = Variable(labels.to(DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                if phase == 'valid':
                    output2 = outputs.data.cpu().numpy()
                    output2 = torch.from_numpy(output2)
                    output2 = output2.type(torch.float32)
                    all_prediction = torch.cat((all_prediction, output2),dim=0)
                    label2 = labels.data.cpu().numpy()
                    label2 = torch.from_numpy(label2)
                    label2 = label2.type(torch.float32)
                    real_labels = torch.cat((real_labels, label2), dim=0)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    torch.save(best_model_wts, PATH)
    model.load_state_dict(best_model_wts)
    return model

model_conv = torchvision.models.resnet18(pretrained=True)
class MyResNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyResNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(
                                           nn.Linear(1000,500),
                                           nn.ReLU(),
                                           nn.Linear(500,10)  
                                            )
    def forward(self, x):
        x = self.pretrained(x)
        x =self.my_new_layers(x)

        return x 
my_extended_model = MyResNet(my_pretrained_model=model_conv).to(DEVICE)



#for param in my_extended_model.pretrained.parameters():
#    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
if use_gpu:
    my_extended_model = my_extended_model.cuda()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(my_extended_model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(my_extended_model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
with open('datalabel.pickle', 'wb') as f:
    pickle.dump(real_labels, f)
with open('datapredict.pickle', 'wb') as g:
    pickle.dump(all_prediction.argmax(dim=1), g)
