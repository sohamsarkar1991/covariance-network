#!/usr/bin/env python
# coding: utf-8

"""
Module for eigen-decomposition of the CovNet operators
The function PCA(model,u,M) computes eigen-functions
of the model and evaluates them at locations defined by u
M is the number of Monte-Carlo samples used to evaluate
the integrals. model is an element of the class CovNetworks.
A function to get an estimate of the mean is also included.
The function mean(model,u) computes the mean field from the
model and evaluates it at locations speicified by u.
"""

import numpy as np
from scipy.linalg import eigh,eigvalsh
import torch
import itertools

def positivize(A,B):
    l_min = eigvalsh(B)[0]
    if l_min < 0:
        B = B - (2*l_min)*np.identity(B.shape[0],dtype='float32')
        A = A - (2*l_min)*np.identity(A.shape[0],dtype='float32')
    return A,B

def locations(K,d):
    gr = np.arange(1,K+1)/(K+1)
    grid = list(itertools.product(gr,repeat=d))
    grid = np.asarray(grid,dtype='float32')
    return grid

def cov_mat(X):
    X_center = X - torch.mean(X, dim=0)
    cov = torch.matmul(X_center.T,X_center)/X_center.size(0)
    return cov

def PCA_weights(model,d,M=50000):
    u = np.array(np.random.uniform(low=0.,high=1.,size=[M,d]),dtype='float32')
    u = torch.from_numpy(u)
    with torch.no_grad():
        G = model.first_step(u)
        G = torch.matmul(G.T,G)/M
        del u
        lam = cov_mat(model.final_layer.weight)
        A = torch.matmul(G,torch.matmul(lam,G.T)).numpy()
        B = G.numpy()
        A,B = positivize(A,B)
        eta, W = eigh(A,B)
    
    return eta, W

def PCA(model,u,M=50000):
    d = u.shape[1]
    u_ = torch.from_numpy(np.array(u,dtype='float32'))
    eta, W = PCA_weights(model,d,M)
    with torch.no_grad():
        u_ = model.first_step(u_)
        W = torch.from_numpy(W)
        psi = torch.matmul(u_,W).numpy()
    
    return eta, psi

def mean(model,u):
    with torch.no_grad():
        beta = torch.mean(model.final_layer.weight, axis=0)
        u_ = torch.from_numpy(np.array(u,dtype='float32'))
        u_ = model.first_step(u_)
        mu = torch.matmul(u_,beta).numpy()
    return mu