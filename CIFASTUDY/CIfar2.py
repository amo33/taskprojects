import torch.nn as nn 
import torch.nn.functional as F 
import torch 
import numpy as np 

class CifarNet(nn.Module):

    def __init__(self, num_classes=10):
        
        super(CifarNet, self).__init__()
        
        self.conv_layer = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1), 
                        nn.ReLU(), 
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
                        nn.BatchNorm2d(128), 
                        nn.ReLU(), 
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
                        nn.ReLU(), 
                        nn.MaxPool2d(2, 2), 
                        nn.Dropout(p=0.5),               
        )


        self.fc_layer = nn.Sequential(
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )
   
    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]
