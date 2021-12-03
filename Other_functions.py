#!/usr/bin/env python
# coding: utf-8

##### Some functions useful for CovNet #####

import torch
import numpy as np


def cov_mat(X):
    X_center = X - torch.mean(X, dim=0)
    cov = torch.matmul(X_center.T,X_center)/X_center.size(0)
    return cov

##### Test error computation #####

def cnet_error(model,u,C_tr):
    with torch.no_grad():
        lam = cov_mat(model.final_layer.weight)
        z = model.first_step(u)
        C_fit = torch.matmul(z,torch.matmul(lam,z.T))
        err = torch.norm(C_fit-C_tr).item()
        nrm = torch.norm(C_tr).item()
    return err/nrm

def cnet_error_grid(model,u,cov_file='True_cov.dat'):
    with torch.no_grad():
        lam = cov_mat(model.final_layer.weight)
        z = model.first_step(u)
        err = torch.tensor(0.)
        nrm = torch.tensor(0.)
        f = open(cov_file,'r')
        for i,line in enumerate(f):
            c_tr = torch.from_numpy(np.asarray(line.strip().split(),dtype='float32'))
            c_fit = torch.matmul(z[i,:],torch.matmul(lam,z.T))
            err += torch.sum((c_tr-c_fit)**2)
            nrm += torch.sum(c_tr**2)
        del c_tr, c_fit, z, lam
        f.close()
        err = torch.sqrt(err).item()
        nrm = torch.sqrt(nrm).item()
    return err/nrm

def cnet_error_MC(model,u,v,cov_file='True_cov.dat'):
    with torch.no_grad():
        lam = cov_mat(model.final_layer.weight)
        z1 = model.first_step(u)
        z2 = model.first_step(v)
        err = torch.tensor(0.)
        nrm = torch.tensor(0.)
        cov = torch.from_numpy(np.loadtxt(cov_file,dtype='float32'))
        for i,c in enumerate(cov):
            c_fit = torch.matmul(z1[i,:],torch.matmul(lam,z2[i,:].T))
            err +=(c-c_fit)**2
            nrm += c**2
        del cov
        err = torch.sqrt(err).item()
        nrm = torch.sqrt(nrm).item()
    return err/nrm

##### Fitted covariance plot #####
def plot_cov(model,u,C_tr,file):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    with torch.no_grad():
        lam = cov_mat(model.final_layer.weight)
        z = model.first_step(u)
        C_fit = torch.matmul(z,torch.matmul(lam,z.T))
    
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    im1 = ax1.imshow(C_tr,cmap='RdYlBu',interpolation='spline16')
    plt.colorbar(im1,cax=cax1,ax=ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('True covariance')

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    im2 = ax2.imshow(C_fit,cmap='RdYlBu',interpolation='spline16')
    plt.colorbar(im2,cax=cax2,ax=ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Fitted covariance')
    
    file.savefig(fig,dpi=75)
    plt.close('all')
    
def plot_cov_from_file(model,u,cov_file,file):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    C_tr = np.loadtxt(cov_file,dtype='float32')
    
    with torch.no_grad():
        lam = cov_mat(model.final_layer.weight)
        z = model.first_step(u)
        C_fit = torch.matmul(z,torch.matmul(lam,z.T))
    
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    im1 = ax1.imshow(C_tr,cmap='RdYlBu',interpolation='spline16')
    plt.colorbar(im1,cax=cax1,ax=ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('True covariance')

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    im2 = ax2.imshow(C_fit,cmap='RdYlBu',interpolation='spline16')
    plt.colorbar(im2,cax=cax2,ax=ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Fitted covariance')
    
    file.savefig(fig,dpi=75)
    plt.close('all')
    
def plot_loss_error(l_tr,l_va,errors,interval,file):
    import matplotlib.pyplot as plt
    
    fig, (ax1,ax2) = plt.subplots(figsize=(12,5),ncols=2)
    ax1.plot(l_tr[interval-1:],'r-',label='Training')
    ax1.plot(l_va[interval-1:],'b-',label='Validation')
    x_ticks = np.linspace(1,len(l_tr[interval-1:]),6)-1
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([int(i+interval) for i in x_ticks])
    ax1.legend()
    ax1.set_title('Evolution of loss during training')

    ax2.plot(errors,'b-')
    x_ticks = np.linspace(0,len(errors)-1,6)
    ax2.set_title('Evolution of Error')
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([int((i+1)*interval) for i in x_ticks])
    
    file.savefig(fig)
    plt.close('all')
    
def plot_loss(l_tr,l_va,burn_in,file):
    import matplotlib.pyplot as plt
    
    fig, (ax1,ax2) = plt.subplots(figsize=(12,5),ncols=2)
    x_ticks = np.linspace(1,len(l_tr)+1,6)-1
    x_ticklabels = [int(i+burn_in) for i in x_ticks]
    ax1.plot(l_tr,'r-',label='Training')
    ax1.set_title('Training loss')
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticklabels)
    ax2.plot(l_va,'b-',label='Validation')
    ax2.set_title('Validation loss')
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_ticklabels)
    
    file.savefig(fig)
    plt.close('all')
    
def Rename_file(file):
    import os
    n = 1
    new_file = file+'_'+str(n)
    while os.path.exists(new_file+'.pdf'):
        n += 1
        new_file = file+'_'+str(n)
    if new_file != file:
        print(file+'.pdf already exists! Plotting figures in '+new_file+'.pdf')
    return new_file

