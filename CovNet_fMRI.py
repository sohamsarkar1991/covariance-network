#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import numpy as np
import nibabel as nib
import time

import CovNetworks as CN
import Important_functions_fMRI as Ifn
import Other_functions as Ofn

cwd = os.getcwd()
print(cwd)

import current_setup_fMRI as setup

method = input("Model [Shallow/Deepshared/Deep]: ")
if method.lower() == 'deepshared' or method.lower() == 'deep':
    depth = input("Depth of the network: ")
    depth = int(depth)
R = input("Number of components (R): ")
R = int(R)

act_fn = setup.act_fn
init = setup.init
#loss = 'COV'
loss_fn = Ifn.loss_COV

if method.lower()=='shallow':
    err_file = 'Err_'+method+'_'+str(R)+'.txt'
elif method.lower()=='deepshared' or method.lower() == 'deep':
    err_file = 'Err_'+method+'_'+str(depth)+'_'+str(R)+'.txt'

dirc = setup.directory

print(time.ctime())

gr = nib.load(dirc+"locations_3D.nii.gz").get_fdata()
K1,K2,K3,d = gr.shape
u = np.empty([56*56*29,3],dtype="float32")
for i in range(d):
    u[:,i] = gr[:,:,:,i].flatten()
del gr
D = u.shape[0]
u = torch.from_numpy(u)

data = nib.load(dirc+"Data.nii.gz").get_fdata()
N = data.shape[3]
x = np.empty([N,K1*K2*K3],dtype="float32")
for n in range(N):
    x[n,:] = data[:,:,:,n].flatten()
del data
x = torch.from_numpy(x)

print("N={}, d={}, D={}x{}x{}" .format(N,d,K1,K2,K3))

if method.lower()=='shallow':
    model = CN.CovNetShallow(d,N,R,act_fn,init)
    checkpoint_file = loss+'_'+method+'_'+str(R)+'.pt'
    epochs = setup.epochs_shallow
    burn_in = setup.burn_in_shallow
    interval = setup.interval_shallow
elif method.lower()=='deepshared':
    model = CN.CovNetDeepShared(d,N,R,depth,act_fn,init)
    checkpoint_file = loss+'_'+method+'_'+str(depth)+'_'+str(R)+'.pt'
    epochs = setup.epochs_deepshared
    burn_in = setup.burn_in_deepshared
    interval = setup.interval_deepshared
elif method.lower()=='deep':
    n_nodes = R
    model = CN.CovNetDeep(d,N,R,depth,n_nodes,act_fn,init)
    checkpoint_file = loss+'_'+method+'_'+str(depth)+'_'+str(R)+'.pt'
    epochs = setup.epochs_deep
    burn_in = setup.burn_in_deep
    interval = setup.interval_deep
split = setup.split
optimizer = setup.optimizer(model.params,lr=setup.lr)
if setup.scheduler is not None:
    scheduler = setup.scheduler(optimizer)
else:
    scheduler = None
loss_file = loss+'_'+method+'_'+str(depth)+'_'+str(R)+'_losses'+'.txt'

print(time.ctime())
Ifn.cnet_optim_best(x,u,model,loss_fn,optimizer,split,scheduler,epochs,burn_in,interval,checkpoint_file,loss_file)
del x,u
print(time.ctime())
