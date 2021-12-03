#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import os
import time

import CovNetworks as CN
import Important_functions as Ifn
import Other_functions as Ofn

cwd = os.getcwd()
print(cwd)

import current_setup as setup

loss_fn = Ifn.loss_COV

method = input("Model [Shallow/Deepshared/Deep]: ")
if method.lower() == 'deepshared' or method.lower() == 'deep':
    depth = input("Depth of the network: ")
    depth = int(depth)
R = input("Number of components (R): ")
R = int(R)

if method.lower()=='shallow':
    err_file = 'Err_'+method+'_'+str(R)+'.txt'
elif method.lower()=='deepshared' or method.lower() == 'deep':
    err_file = 'Err_'+method+'_'+str(depth)+'_'+str(R)+'.txt'

folds = setup.folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=folds)
CV_scores = []

act_fn = setup.act_fn
init = setup.init

dirc = setup.directory

u = np.loadtxt(dirc+'locations.dat',dtype='float32')
D, d = u.shape
u = torch.from_numpy(u)
x = np.loadtxt(dirc+'Example.dat',dtype='float32')
N = x.shape[0]
if x.shape[1] != D:
    exit('Data shape mismatch!! Aborting..')
print('N='+str(N)+', D='+str(D)+', d='+str(d))
    
x = torch.from_numpy(x)

f_err = open(err_file,'w')
f_err.write('CV scores:\n')
f_err.close()
k = 1
for tr_idx, va_idx in kf.split(x):
    x_tr, x_va = x[tr_idx], x[va_idx]
    N_tr = x_tr.shape[0]
    if method.lower()=='shallow':
        model = CN.CovNetShallow(d,N_tr,R,act_fn,init)
        checkpoint_file = method+'_'+str(R)+'.pt'
        epochs = setup.epochs_shallow
        burn_in = setup.burn_in_shallow
        interval = setup.interval_shallow
    elif method.lower()=='deepshared':
        model = CN.CovNetDeepShared(d,N_tr,R,depth,act_fn,init)
        checkpoint_file = method+'_'+str(depth)+'_'+str(R)+'.pt'
        epochs = setup.epochs_deepshared
        burn_in = setup.burn_in_deepshared
        interval = setup.interval_deepshared
    elif method.lower()=='deep':
        n_nodes = R
        model = CN.CovNetDeep(d,N_tr,R,depth,n_nodes,act_fn,init)
        checkpoint_file = method+'_'+str(depth)+'_'+str(R)+'.pt'
        epochs = setup.epochs_deep
        burn_in = setup.burn_in_deep
        interval = setup.interval_deep
    split = setup.split
    optimizer = setup.optimizer(model.params,lr=setup.lr)
    if setup.scheduler is not None:
        scheduler = setup.scheduler(optimizer)
    else:
        scheduler = None
    print(time.ctime())
    Ifn.cnet_optim_best(x_tr,u,model,loss_fn,optimizer,split,scheduler,epochs,burn_in,interval,checkpoint_file)
    del x_tr
    with torch.no_grad():
        CV_scores.append(loss_fn(x_va,model(u)).item())
    os.remove(checkpoint_file)
    f_err = open(err_file,'a')
    f_err.write('Fold{}: {:.10f}\n' .format(k,CV_scores[-1]))
    f_err.close()
    print('Fold{}: {:.10f}' .format(k,CV_scores[-1]))
    k += 1
print(time.ctime())

CV_score = np.mean(CV_scores)
f_err = open(err_file,'a')
f_err.write('\n\n\n')
f_err.write('Average CV score = {:.10f}' .format(CV_score))
print('CV score = {:.6f}' .format(CV_score))
f_err.close()

if method.lower()=='shallow':
    model = CN.CovNetShallow(d,N,R,act_fn,init)
    checkpoint_file = method+'_'+str(R)+'.pt'
    epochs = setup.epochs_shallow
    burn_in = setup.burn_in_shallow
    interval = setup.interval_shallow
elif method.lower()=='deepshared':
    model = CN.CovNetDeepShared(d,N,R,depth,act_fn,init)
    checkpoint_file = method+'_'+str(depth)+'_'+str(R)+'.pt'
    epochs = setup.epochs_deepshared
    burn_in = setup.burn_in_deepshared
    interval = setup.interval_deepshared
elif method.lower()=='deep':
    n_nodes = R
    model = CN.CovNetDeep(d,N,R,depth,n_nodes,act_fn,init)
    checkpoint_file = method+'_'+str(depth)+'_'+str(R)+'.pt'
    epochs = setup.epochs_deep
    burn_in = setup.burn_in_deep
    interval = setup.interval_deep
split = setup.split
optimizer = setup.optimizer(model.params,lr=setup.lr)
if setup.scheduler is not None:
    scheduler = setup.scheduler(optimizer)
else:
    scheduler = None
print(time.ctime())
Ifn.cnet_optim_best(x,u,model,loss_fn,optimizer,split,scheduler,epochs,burn_in,interval,checkpoint_file)
del x,u


file = dirc+'True_locations.dat'
loc = np.loadtxt(file,dtype='float32')
u = torch.from_numpy(loc[:,:d])
v = torch.from_numpy(loc[:,d:])
del loc
cov_file = dirc+'True_cov.dat'
err = Ofn.cnet_error_MC(model,u,v,cov_file)

f_err = open(err_file,'a')
f_err.write('\n\n\n')
f_err.write('Error = {:.10f}\n' .format(err))
f_err.close()
print('Error = {:.6f}'.format(err))
print(time.ctime())

