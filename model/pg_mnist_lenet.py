import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.pg_utils as q

from torch.autograd import Variable

__all__ = ['LeNet']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = q.PGConv2d(1,32,kernel_size = 5,padding=0)
        self.conv2 = q.PGConv2d(32,64,kernel_size = 5,padding=0)
        self.fc1 = nn.Linear(1024,512)
        self.bn  = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,10)
    def forward(self,x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        #print(x.shape)
        x = x.view(-1,1024)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x
        


