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

cwd = os.getcwd()
print(cwd)

import current_setup as setup

loss_fn = Ifn.loss_COV

method = input("Model [Shallow/Deepshared/Deep]: ")
if method.lower() == "shallow":
    depths = [1] # for compatibility
    Rs = [5,10,20,40,80] # choice of R's
elif method.lower() == 'deepshared' or method.lower() == 'deep':
    depths = [2,3,4] # choice of depths
    Rs = [5,10,20,40] # choice of R's

if method.lower()=='shallow':
    err_file = 'Err_CovNet_CV_'+method+'.txt'
elif method.lower()=='deepshared' or method.lower() == 'deep':
    err_file = 'Err_CovNet_CV_'+method+'.txt'

folds = setup.folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=folds)
act_fn = setup.act_fn
init = setup.init
dirc = setup.directory

u = np.loadtxt(dirc+'locations.dat',dtype='float32')
if len(u.shape)==1:
    D, d = len(u), 1
    u = u.reshape(D,1)
else:
    D, d = u.shape
x = np.loadtxt(dirc+'Example.dat',dtype='float32')
N = x.shape[0]
if x.shape[1] != D:
    exit('Data shape mismatch!! Aborting..')
print('N='+str(N)+', D='+str(D)+', d='+str(d))    
u = torch.from_numpy(u)
x = torch.from_numpy(x)

CV_scores = np.zeros(len(depths)*len(Rs),dtype=float)
keys = []
if method.lower()=="shallow":
    for R in Rs:
        keys.append("{}".format(R))
elif method.lower()=="deepshared" or method.lower()=="deep":
    for depth in depths:
        for R in Rs:
            keys.append("{}_{}".format(depth,R))

f_err = open(err_file,'w')
f_err.write('CV scores:\n\t')
for key in keys:
    f_err.write("\t\t{}" .format(key))
f_err.write("\n")
f_err.close()

print(time.ctime())
k = 0
for tr_idx, va_idx in kf.split(x):
    k += 1
    x_tr, x_va = x[tr_idx], x[va_idx]
    N_tr = x_tr.shape[0]
    models = []
    checkpoints = []
    if method.lower()=='shallow':
        epochs = setup.epochs_shallow
        burn_in = setup.burn_in_shallow
        interval = setup.interval_shallow
        for R in Rs:
            model = CN.CovNetShallow(d,N_tr,R,act_fn,init)
            checkpoint_file = method+'_'+str(R)+'.pt'
            models.append(model)
            checkpoints.append(checkpoint_file)
    elif method.lower()=='deepshared':
        epochs = setup.epochs_deepshared
        burn_in = setup.burn_in_deepshared
        interval = setup.interval_deepshared
        for depth in depths:
            for R in Rs:
                model = CN.CovNetDeepShared(d,N_tr,R,depth,act_fn,init)
                checkpoint_file = method+'_'+str(depth)+'_'+str(R)+'.pt'
                models.append(model)
                checkpoints.append(checkpoint_file)
    elif method.lower()=='deep':
        epochs = setup.epochs_deep
        burn_in = setup.burn_in_deep
        interval = setup.interval_deep
        for depth in depths:
            for R in Rs:
                n_nodes = R
                model = CN.CovNetDeep(d,N_tr,R,depth,n_nodes,act_fn,init)
                checkpoint_file = method+'_'+str(depth)+'_'+str(R)+'.pt'
                models.append(model)
                checkpoints.append(checkpoint_file)
    split = setup.split
                
    f_err = open(err_file,'a')
    f_err.write("Fold{}:" .format(k))
    f_err.close()
    print(time.ctime())
    for i,model in enumerate(models):
        optimizer = setup.optimizer(model.params,lr=setup.lr)
        Ifn.cnet_optim_best(x_tr,u,model,loss_fn,optimizer,split,epochs,burn_in,interval,checkpoints[i])
        with torch.no_grad():
            err = loss_fn(x_va,model(u)).item()
            CV_scores[i] += err
            f_err = open(err_file,'a')
            f_err.write("\t{:.6f}" .format(err))
            f_err.close()
        os.remove(checkpoints[i])
    f_err = open(err_file,'a')
    f_err.write("\n")
    f_err.close()
    print('Fold{}:' .format(k))
    print(np.round(CV_scores/k,3))
    del x_tr, x_va, models, checkpoints

CV_scores = CV_scores/folds
f_err = open(err_file,'a')
f_err.write("Average:")
for err in CV_scores:
    f_err.write("\t{:.6f}" .format(err))

best = np.argmin(CV_scores)
best = keys[best]
if method.lower()=="shallow":
    R = int(best)
    f_err.write("\n\nThe best CV-score is achieved for R={}\n" .format(R))
    print("The best CV-score is achieved for R={}" .format(R))
elif method.lower()=="deepshared" or method.lower()=="deep":
    best = best.split('_')
    depth = int(best[0])
    R = int(best[1])
    f_err.write("\n\nThe best CV-score is achieved for depth={}, R={}\n" .format(depth,R))
    print("The best CV-score is achieved for depth={}, R={}" .format(depth,R))
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
    
Ifn.cnet_optim_best(x,u,model,loss_fn,optimizer,split,epochs,burn_in,interval,checkpoint_file)
ellapsed = time.time() - current
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
