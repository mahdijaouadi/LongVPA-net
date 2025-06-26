import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
from sklearn.metrics import mean_absolute_error
import random
import matplotlib.pyplot as plt


class Block(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,downsample=None,stride=1):
        super(Block,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3=nn.Conv2d(out_channels,out_channels*Block.expansion,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels*Block.expansion)
        self.downsample=downsample
    def forward(self,x):
        identity=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=F.relu(out)
        out=self.conv3(out)
        out=self.bn3(out)
        if self.downsample is not None:
            identity=self.downsample(identity)
        out+=identity
        out=F.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self,layers):
        super(Resnet,self).__init__()
        self.in_channels=64
        self.conv=nn.Conv2d(3,self.in_channels,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn= nn.BatchNorm2d(self.in_channels)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1) 
        self.layer1=self._make_layer(layers[0],64,1)
        self.layer2=self._make_layer(layers[1],128,2)
        self.layer3=self._make_layer(layers[2],256,2)
        self.layer4=self._make_layer(layers[3],512,2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self,num_blocks,out_channels,stride=1):
        downsample=None
        if stride!=1 or self.in_channels!=out_channels*Block.expansion:
            downsample=nn.Sequential(nn.Conv2d(self.in_channels,out_channels*Block.expansion,kernel_size=1,stride=stride,padding=0,bias=False),
                                     nn.BatchNorm2d(out_channels*Block.expansion))
        layers=[]
        layers.append(Block(in_channels=self.in_channels,out_channels=out_channels,downsample=downsample,stride=stride))
        self.in_channels=out_channels*Block.expansion
        for _ in range(num_blocks-1):
            layers.append(Block(in_channels=self.in_channels,out_channels=out_channels))
        return nn.Sequential(*layers)
    def forward(self,x):
        out=self.conv(x)
        out=self.bn(out)
        out=self.maxpool(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        return out
class MyModel(nn.Module):
    def __init__(self,layers,output_size,device='cpu'):
        super(MyModel,self).__init__()
        self.device=device
        self.bottleneck_1d=nn.Conv2d(3,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bottleneck_1mo=nn.Conv2d(3,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.hidden_size=64
        self.num_layers=3
        self.lstm=nn.LSTM(input_size=1,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,dropout=0.5)
        #Resnet layers
        self.resnet=Resnet(layers)


        self.fc=nn.Linear(512*Block.expansion,output_size)

    def forward(self, inputs):
        chart_1d=torch.stack([t[0] for t in inputs]).to(self.device)
        chart_1mo=torch.stack([t[1] for t in inputs]).to(self.device)
        sequential_spy=torch.stack([t[2] for t in inputs]).to(self.device)
        chart_1d_out=self.bottleneck_1d(chart_1d)
        chart_1mo_out=self.bottleneck_1mo(chart_1mo)
        mix_chart = torch.cat((chart_1d_out, chart_1mo_out), dim=1)
        mix_chart_out=self.resnet(mix_chart)
        sequential_spy_out, (hidden, cell) = self.lstm(sequential_spy)
        sequential_spy_out=sequential_spy_out[:,-1,:]
        out = torch.cat((mix_chart_out, sequential_spy_out), dim=1)

        out=self.fc(out)
        return out
def init_weights(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias,0)
    elif isinstance(m,nn.LSTM):
        for name,param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param,nonlinearity='relu')
            if 'bias' in name:
                nn.init.constant_(param,0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
