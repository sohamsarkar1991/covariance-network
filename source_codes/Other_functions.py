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
