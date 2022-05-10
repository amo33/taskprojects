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
from torchvision.datasets import ImageFolder
import time 
import PIL, os
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding="bytes")
    return data

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


transform_train = transforms.Compose([ 
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def load_cifar_10_data(data_dir, negatives=False): #흩어져있는 batches들 합친다.
   
    meta_data_dict = unpickle(data_dir+"/batches.meta")
    if not os.path.exists("data/train"):
        for target in classes:
            os.makedirs("data/train/"+target)
    if not os.path.exists("data/test"):
        for target in classes:
            os.makedirs("data/test/"+target)
    if not os.path.exists("data/valid"):
        for target in classes:
            os.makedirs("data/valid/"+target)
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)
    
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []
    
    for i in range(1,7):
        print(i)
        if i == 6:
            cifar_train_data_dict = unpickle(data_dir + "/test_batch")
        else:    
            cifar_train_data_dict = unpickle(data_dir+"/class{}".format(i))
        
        if i==1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data'])) # 아래로 합쳐
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']
    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))

    cifar_train_data = np.array(cifar_train_data, dtype=np.float32)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)
    

    size = 0.3 
    territory = int(np.round(len(cifar_train_data)*size))
    print(territory)
    
    cifar_test_data = cifar_train_data[0:territory]
    #cifar_test_filenames = cifar_train_filenames[0:territory]
    cifar_test_labels = cifar_train_labels[0:territory]
    cifar_train_data = cifar_train_data[territory:]
    cifar_train_filenames = cifar_train_filenames[territory:]
    cifar_train_labels = cifar_train_labels[territory:]

    val_range = int(np.round(len(cifar_train_data) * 0.2))
    cifar_valid_data = cifar_train_data[0:val_range]
    cifar_train_data = cifar_train_data[val_range:]
    cifar_valid_labels = cifar_train_labels[0:val_range]
    cifar_train_labels = cifar_train_labels[val_range:]
    #cifar_valid_filenames = cifar_train_filenames[0:val_range]
    cifar_train_filenames = cifar_train_filenames[val_range: ]
    print(len(cifar_train_data))
    for idx in range(len(cifar_train_data)):
        train_label = cifar_train_labels[idx]
        train_image = cifar_train_data[idx].transpose(1,2,0)  #PIL.Image는 H*W*C 형태여서 transpose
        train_image = train_image.astype(np.uint8)
        train_image = Image.fromarray(train_image)
        # 클래스 별 폴더에 파일 저장
        train_image.save('data/train/{}/{}.jpg'.format(classes[train_label], idx))
    for idx in range(len(cifar_valid_data)):
        valid_label = cifar_valid_labels[idx]
        valid_image = cifar_valid_data[idx].transpose(1,2,0)
        valid_image = valid_image.astype(np.uint8)
        valid_image = Image.fromarray(valid_image)
        valid_image.save('data/valid/{}/{}.jpg'.format(classes[valid_label], idx))
    for idx in range(len(cifar_test_data)):
        test_label = cifar_test_labels[idx]
        test_image = cifar_test_data[idx].transpose(1,2,0)
        test_image = test_image.astype(np.uint8)
        test_image = Image.fromarray(test_image)
        test_image.save('data/test/{}/{}.jpg'.format(classes[test_label], idx))
