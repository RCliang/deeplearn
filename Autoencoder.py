#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:33:15 2019

@author: dongliang
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:24:58 2019
@author: 8000725635
"""

from sklearn import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt

def PCA(data,k=2):
    X=torch.from_numpy(data)
    X_mean=torch.mean(X,0)
    X=X-X_mean.expand_as(X)
    
    U,S,V=torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

iris=datasets.load_iris()
X=iris.data
y=iris.target
X_PCA=PCA(X)
pca=X_PCA.numpy()

plt.figure()
color=['red','green','blue']
for i, target_name in enumerate(iris.target_names):
    plt.scatter(pca[y==i,0],pca[y==i,1],label=target_name,color=color[i])
    plt.legend()
    plt.title('PCA for IRIS dataset')
    plt.show()
    

import os 
import pdb
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt

torch.manual_seed(1)
batch_size=128
learning_rate=0.01
num_epochs=10

train_dataset = datasets.MNIST(root='F:/数据/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset=datasets.MNIST(root='F:/数据/data',train=False,transform=transforms.ToTensor())
train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder=nn.Sequential(
                nn.Linear(28*28, 1000),
                nn.ReLU(True),
                nn.Linear(1000,500),
                nn.ReLU(True),
                nn.Linear(500,250),
                nn.ReLU(True),
                nn.Linear(250,2))
        self.decoder=nn.Sequential(
                nn.Linear(2,250),
                nn.ReLU(True),
                nn.Linear(250,500),
                nn.ReLU(True),
                nn.Linear(500,1000),
                nn.ReLU(True),
                nn.Linear(1000,28*28),
                nn.Tanh())
    
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
    
model=autoencoder()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(
        model.parameters(),lr=learning_rate,weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in train_loader:
        img, _=data
        img=img.view(img.size(0),-1)
        img=Variable(img)
        output=model(img)
        loss=criterion(output,img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())
        
model.eval()
eval_loss=0
for data in test_loader:
    img,label=data
    img=img.view(img.size(0),-1)
    img=Variable(img,volatile=True)
    label=Variable(label,volatile=True)
    out=model(img)
    y=(label.data).numpy()
    plt.scatter(out[:,0],out[:,1],c=y)
    plt.colorbar()
    plt.title('autocoder of MNIST test dataset')
    plt.show()