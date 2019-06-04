#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:02:14 2019

@author: dongliang
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
CONTEXT_SIZE=2
EMBEDDING_DIM=10
N_EPOCHS=10

from mxnet import autograd, nd

x=nd.arange(4).reshape((4,1))
x.attach_grad()
with autograd.record():
    y=2*nd.dot(x.T,x)
assert(x.grad-4*x).norm().asscalar()==0
x.grad

import d2lzh

