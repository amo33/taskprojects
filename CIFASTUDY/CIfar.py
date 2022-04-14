import torch.nn as nn 
import torch.nn.functional as F 
import torch 
import numpy as np 
class CifarNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.conv1 = nn.Conv2d(3,16,kernel_size=3, stride=2, padding=2) #input_channel = 채널 수 (rgb = 3개다.)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32,64, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = x.permute(0,3,1,2)#batch, 얼만큼의 채널, 가로, 세로 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) #이렇게만 해도 되려나?
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x= F.log_softmax(x)
        #print(x)
        return x
    # permute를 하는 이유 : 애초에 reshape을 저 axis 순으로 하면 되지 않나?
    # ---> 먼저 reshape 시 3,32,32를 함으로써 rgb의 형태로 만든다.
    # ---> 이후에 학습하기 위해 다시 permute로 axis를 바꿔주는 것이기 때문에 불필요한 과정이 아니라고 판단.