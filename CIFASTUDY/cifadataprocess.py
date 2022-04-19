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
'''
def load_cifar_10_data(data_dir, negatives=False): #흩어져있는 batches들 합친다.
    tf = transforms.ToTensor()
    tfnormal = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    meta_data_dict = unpickle(data_dir+"/batches.meta")
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
            cifar_train_data_dict = unpickle(data_dir+"/data_batch_{}".format(i))
        
        if i==1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data'])) # 아래로 합쳐
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']
    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    #if negatives: # 왜 이게 필요한걸까
    #    cifar_train_data = cifar_train_data.transpose(0,2,3,1).astype(np.float32)
    #else:
    #    cifar_train_data = np.rollaxis(cifar_train_data, 1,4)
    cifar_train_data = np.array(cifar_train_data, dtype=np.float32)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)
    
    size = 0.3 
    territory = int(np.round(len(cifar_train_data)*size))
    print(territory)
    
    cifar_test_data = cifar_train_data[0:territory]
    cifar_test_filenames = cifar_train_filenames[0:territory]
    cifar_test_labels = cifar_train_labels[0:territory]
    cifar_train_data = cifar_train_data[territory:]
    cifar_train_filenames = cifar_train_filenames[territory:]
    cifar_train_labels = cifar_train_labels[territory:]

    with open('testdata.pickle','wb') as f:
        pickle.dump(cifar_test_data,f)
    with open('testlabel.pickle','wb') as fw:
        pickle.dump(cifar_test_labels, fw)
    #cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    #cifar_test_data = cifar_test_data_dict[b'data']
    #cifar_test_filenames = cifar_test_data_dict[b'filenames']
    #cifar_test_labels = cifar_test_data_dict[b'labels']

    #cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    #if negatives:
    #    cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    #else:
    #    cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
        
    #cifar_test_filenames = np.array(cifar_test_filenames)
    #cifar_test_labels = np.array(cifar_test_labels)
    #print(cifar_train_data[2])
    for i in range(cifar_train_data.shape[0]):
        img = cifar_train_data[i].reshape(-1, 32*32*3)
        img = img / 255
        img = img.reshape(32,32,3)
        cifar_train_data[i] = tf(img)
    x_train = torch.from_numpy(cifar_train_data)
    x_train = tfnormal(x_train)
    print(x_train.shape)
    #x_train = cifar_train_data
    #print(x_train)
    val_range = int(np.round(len(x_train) * 0.2))
    valid_set = x_train[0:val_range]
    train_set = x_train[val_range:]
    
    y_train = to_categorical(cifar_train_labels, 10)
    
    train_label_set = y_train[val_range:]
    val_label_set = y_train[0:val_range]
    
    # x_test = (cifar_test_data.astype('float32')/255 -0.5) / 0.5
    # print(x_test.shape)
    #x_test = cifar_test_data
    #y_test = to_categorical(cifar_test_labels, 10)
    # ---train_images_tensor = torch.tensor(train_set)
    train_images_tensor= train_set
    #print(train_images_tensor.shape)
    #train_images_tensor=transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(train_images_tensor)
    # ---valid_images_tensor = torch.tensor(valid_set)
    valid_images_tensor = valid_set
    #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(valid_images_tensor)
    #train_set, val_set = torch.utils.data.random_split(train_images_tensor, [len(train_images_tensor)*0.8, len(train_images_tensor)*0.2])
    train_label_tensor = torch.tensor(train_label_set)
    valid_label_tensor = torch.tensor(val_label_set)
    #train_label_set, val_label_set = torch.utils.data.random_split(train_label_tensor, [len(train_label_tensor)*0.8, len(train_label_tensor)*0.2])

    train_tensor = TensorDataset(train_images_tensor, train_label_tensor)
    train_loader = DataLoader(train_tensor, batch_size=4, num_workers= 0 , shuffle = True)
    
    valid_tensor = TensorDataset(valid_images_tensor, valid_label_tensor)
    valid_loader = DataLoader(valid_tensor, batch_size=4, num_workers= 0, shuffle = False)
    
    #test_images_tensor = torch.tensor(x_test)
    #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(test_images_tensor)
    #test_label_tensor = torch.tensor(y_test)
    #test_tensor = TensorDataset(test_images_tensor, test_label_tensor)
    #test_loader = DataLoader(test_tensor, batch_size=4, num_workers = 0, shuffle = False)
    return train_loader, valid_loader
'''

transform_train = transforms.Compose([ 
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def load_cifar_10_data(data_dir, negatives=False): #흩어져있는 batches들 합친다.
    #tf = transforms.ToTensor()
    #tfnormal = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   
    meta_data_dict = unpickle(data_dir+"/batches.meta")
    #for target in classes:
    #    os.makedirs("data/train/"+target)
    #for target in classes:
    #    os.makedirs("data/valid/"+target)
    #for target in classes:
    #    os.makedirs("data/test/"+target)
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
    cifar_test_filenames = cifar_train_filenames[0:territory]
    cifar_test_labels = cifar_train_labels[0:territory]
    cifar_train_data = cifar_train_data[territory:]
    cifar_train_filenames = cifar_train_filenames[territory:]
    cifar_train_labels = cifar_train_labels[territory:]

    #with open('data/test/testdata.','wb') as f:
    #    pickle.dump(cifar_test_data,f)
    #with open('data/test/testlabel.pickle','wb') as fw:
    #    pickle.dump(cifar_test_labels, fw)
    '''
    for i in range(cifar_train_data.shape[0]):
        img = cifar_train_data[i].reshape(-1, 32*32*3)
        img = img / 255
        img = img.reshape(32,32,3)
        cifar_train_data[i] = tf(img)
    '''
    val_range = int(np.round(len(cifar_train_data) * 0.2))
    cifar_valid_data = cifar_train_data[0:val_range]
    cifar_train_data = cifar_train_data[val_range:]
    cifar_valid_labels = cifar_train_labels[0:val_range]
    cifar_train_labels = cifar_train_labels[val_range:]
    cifar_valid_filenames = cifar_train_filenames[0:val_range]
    cifar_train_filenames = cifar_train_filenames[val_range: ]
    print(len(cifar_train_data))
    for idx in range(len(cifar_train_data)):
        train_label = cifar_train_labels[idx]
        train_image = cifar_train_data[idx].transpose(1,2,0) #.reshape(32,32,3) # 1 1 3 error 해결용으로 3줄 사용 
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
    '''
    y_train = to_categorical(cifar_train_labels, 10)
    
    train_label_set = y_train[val_range:]
    val_label_set = y_train[0:val_range]
    
    train_tensor = ImageFolder(root='data/train', transform=transform_train)
    valid_tensor = ImageFolder(root='data/valid', transform=transform_train)

    print(train_tensor)

    train_loader = DataLoader(train_tensor, batch_size=4, num_workers= 0 , shuffle = True)
 
    valid_loader = DataLoader(valid_tensor, batch_size=4, num_workers= 0, shuffle = False)
'''