#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time

import torch
import numpy as np

sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Desktop", "CovNet", "source_codes"))

import CovNetworks as CN
import Important_functions as Ifn
import Other_functions as Ofn


import current_setup as setup

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
replicates = setup.replicates

for repl in replicates:
    print('Example'+str(repl+1)+':')
    file = dirc+'locations'+str(repl+1)+'.dat'
    u = np.loadtxt(file,dtype='float32')
    D, d = u.shape
    u = torch.from_numpy(u)
    file = dirc+'Example'+str(repl+1)+'.dat'
    x = np.loadtxt(file,dtype='float32')
    N = x.shape[0]
    if x.shape[1] != D:
        exit('Data shape mismatch!! Aborting..')
    print('N='+str(N)+', D='+str(D)+', d='+str(d))
    
    x = torch.from_numpy(x)
    #x = x - torch.mean(x,dim=0,keepdim=True)

    if method.lower()=='shallow':
        epochs = setup.epochs_shallow
        burn_in = setup.burn_in_shallow
        interval = setup.interval_shallow
        model = CN.CovNetShallow(d,N,R,act_fn,init)
        checkpoint_file = 'Best_'+method+'_'+str(R)+'.pt'
    elif method.lower()=='deepshared':
        epochs = setup.epochs_deepshared
        burn_in = setup.burn_in_deepshared
        interval = setup.interval_deepshared
        model = CN.CovNetDeepShared(d,N,R,depth,act_fn,init)
        checkpoint_file = 'Best_'+method+'_'+str(depth)+'_'+str(R)+'.pt'
    elif method.lower()=='deep':
        epochs = setup.epochs_deep
        burn_in = setup.burn_in_deep
        interval = setup.interval_deep
        n_nodes = R
        model = CN.CovNetDeep(d,N,R,depth,n_nodes,act_fn,init)
        checkpoint_file = 'Best_'+method+'_'+str(depth)+'_'+str(R)+'.pt'
        
    optimizer = setup.optimizer(model.params,lr=setup.lr)
    split = setup.split
    
    print(time.ctime())
    l_tr, l_va, epoch = Ifn.cnet_optim_best(x,u,model,loss_fn,optimizer,split,epochs,burn_in,interval,checkpoint_file)
    del x,u
    
    file = dirc+'True_locations'+str(repl+1)+'.dat'
    loc = np.loadtxt(file,dtype='float32')
    u = torch.from_numpy(loc[:,:d])
    v = torch.from_numpy(loc[:,d:])
    del loc
    cov_file = dirc+'True_cov'+str(repl+1)+'.dat'
    err = Ofn.cnet_error_MC(model,u,v,cov_file)
    
    f_Err = open(err_file,'a')
    f_Err.write('Example{}:\n' .format(repl+1))
    f_Err.write('Error = {:.10f}\n' .format(err))
    f_Err.write('Number of epochs = {}\n' .format(epoch))
    f_Err.close()
    print('Error = {:.6f}'.format(err))
    print('Number of epochs = {}' .format(epoch))
    print(time.ctime())
    print('\n')
    os.remove(checkpoint_file)
