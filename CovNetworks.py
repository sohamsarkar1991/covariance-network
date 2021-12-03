#!/usr/bin/env python
# coding: utf-8

"""
Module to create CovNet models.
A model can be created by using
CovNetShallow(d,N,R,act_fn,init) - for shallow
or CovNetDeep(d,N,R,depth,n_nodes,act_fn,init) - for deep
or CovNetDeepShared(d,N,R,depth,act_fn,init)
N - Number of fields (integer)
d - dimension (integer)
R, depth (L), n_nodes(p1=...=pL) - network parameters (integer)
act_fn - activation function \sigma; needs to be an element from torch.nn activation function
init - initialization for the weights of the model; biases are initialized as zero.
"""

import torch
from collections import OrderedDict
from operator import methodcaller

##### Shallow CovNet #####
class CovNetShallow(torch.nn.Module):
    def __init__(self,d,N,R,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        super(CovNetShallow, self).__init__()
        self.init = init
        self.act_fn = act_fn
        self.first_layer = torch.nn.Linear(d,R,bias=True)
        self.par_init(self.first_layer)
        self.final_layer = torch.nn.Linear(R,N,bias=False)
        self.par_init(self.final_layer)
        self.params = list(self.first_layer.parameters()) + list(self.final_layer.parameters())
    
    def par_init(self,m):
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
        self.init(m.weight)
    
    def first_step(self,u):
        return self.act_fn(self.first_layer(u))
    
    def forward(self,u):
        return self.final_layer(self.first_step(u)).T


##### Deep CovNet #####
class cnet(torch.nn.Module): ## module to create one deep network
    def __init__(self,d,depth,n_nodes,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        super(cnet, self).__init__()
        self.init = init
        layers = [('layer1',torch.nn.Linear(d,n_nodes,bias=True)),
                  ('act1',act_fn)]
        for i in range(depth-2):
            layers.append(('layer'+str(i+2),torch.nn.Linear(n_nodes,n_nodes,bias=True)))
            layers.append(('act'+str(i+2),act_fn))
        if depth==2:
            i=-1
        layers.append(('layer'+str(i+3),torch.nn.Linear(n_nodes,1,bias=True)))
        layers.append(('act'+str(i+3),act_fn))
        self.fn = torch.nn.Sequential(OrderedDict(layers))
        self.fn.apply(self.par_init_sequential)
        self.params = list(self.fn.parameters())
    def par_init(self,m):
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
        self.init(m.weight)
    def par_init_sequential(self,m):
        if type(m) == torch.nn.Linear:
            self.par_init(m)
    
    def forward(self,u):
        return self.fn(u)
    
class CovNetDeep(torch.nn.Module):
    def __init__(self,d,N,R,depth,n_nodes,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        super(CovNetDeep, self).__init__()
        self.models = []
        self.params = []
        for i in range(R):
            self.models.append(cnet(d,depth,n_nodes,act_fn,init))
            self.params += self.models[i].params
        self.final_layer = torch.nn.Linear(R,N,bias=False)
        init(self.final_layer.weight)
        self.params += self.final_layer.parameters()
        
    def first_step(self,u):
        return torch.cat(list(map(methodcaller('__call__', u), self.models)),dim=1)
        #return torch.cat([model(u) for model in self.models],dim=1)
    
    def forward(self,u):
        return self.final_layer(self.first_step(u)).T

        
##### Deepshared CovNet #####
class CovNetDeepShared(torch.nn.Module):
    def __init__(self,d,N,R,depth,act_fn=torch.nn.Sigmoid(),init=torch.nn.init.xavier_normal_):
        '''if depth==1:
            print('For depth=1, use CovNetShallow to define the model.')'''
        super(CovNetDeepShared, self).__init__()
        self.init = init
        layers = [('layer1',torch.nn.Linear(d,R,bias=True)),
                  ('act1',act_fn)]
        for i in range(depth-1):
            layers.append(('layer'+str(i+2),torch.nn.Linear(R,R,bias=True)))
            layers.append(('act'+str(i+2),act_fn))
        self.initial_layers = torch.nn.Sequential(OrderedDict(layers))
        self.initial_layers.apply(self.par_init_sequential)
        self.final_layer = torch.nn.Linear(R,N,bias=False)
        self.par_init(self.final_layer)
        self.params = list(self.initial_layers.parameters()) + list(self.final_layer.parameters())
    def par_init(self,m):
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
        self.init(m.weight)
        
    def par_init_sequential(self,m):
        if type(m) == torch.nn.Linear:
            self.par_init(m)
            
    def first_step(self,u):
        return self.initial_layers(u)
    
    def forward(self,u):
        return self.final_layer(self.first_step(u)).T

