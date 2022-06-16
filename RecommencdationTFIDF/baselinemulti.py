import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, Bottleneck
from torch import Tensor

class Resnet50DeepFashion(ResNet):
 def __init__(self, num_category, num_texture, num_fabric, num_shape, num_part, num_style,**kwargs):
  super(Resnet50DeepFashion, self).__init__(Bottleneck, [3,4,6,3], **kwargs)
  self.fc = nn.Linear(2048, num_category)
  self.fc_texture = nn.Linear(2048, num_texture)
  self.fc_fabric = nn.Linear(2048, num_fabric)
  self.fc_shape = nn.Linear(2048, num_shape)
  self.fc_part = nn.Linear(2048, num_part)
  self.fc_style = nn.Linear(2048, num_style)

 def _forward_impl(self, x: Tensor) -> [Tensor]:
  x = self.conv1(x)
  x = self.bn1(x)
  x = self.relu(x)
  x = self.maxpool(x)

  x = self.layer1(x)
  x = self.layer2(x)
  x = self.layer3(x)
  x = self.layer4(x)

  x = self.avgpool(x)
  x = torch.flatten(x, 1)
  
  category = self.fc(x)
  texture = self.fc_texture(x)
  fabric = self.fc_fabric(x)
  shape = self.fc_shape(x)
  part = self.fc_part(x)
  style = self.fc_style(x)

  return [category, texture, fabric, shape, part, style]

 def forward(self, x: Tensor) -> [Tensor]:
  return self._forward_impl(x)

if name == '__main__':
 net = Resnet50DeepFashion(50,100,100,100,100,100)
 print(net)