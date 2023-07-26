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
model and evaluates it at locations u.
"""

import numpy as np
from scipy.linalg import eigh
import torch
import itertools

def gen_eigh(A,B,tol=1e-10):
    """ Generalized eigen-decomposition
        Particularly useful when B is singular
    """
    lam_B, w_B = eigh(B)
    idx = lam_B>tol
    lam_B = lam_B[idx]
    w_B = w_B[:,idx]
    C = np.matmul(w_B.T,np.matmul(A,w_B))
    D = np.diag(lam_B)
    lam, Alpha = eigh(C,D)
    w = np.matmul(w_B,Alpha)
    return lam,w

def locations(K,d):
    gr = np.arange(1,K+1)/(K+1)
    grid = list(itertools.product(gr,repeat=d))
    grid = np.asarray(grid,dtype='float32')
    return grid

def cov_mat(X):
    X_center = X - torch.mean(X, dim=0)
    cov = torch.matmul(X_center.T,X_center)/X_center.size(0)
    return cov

def PCA_weights(model,d,M=50000,tol=1e-10):
    u = np.array(np.random.uniform(low=0.,high=1.,size=[M,d]),dtype="float32")
    u = torch.from_numpy(u)
    with torch.no_grad():
        G = model.first_step(u)
        G = torch.matmul(G.T,G)/M
        del u
        lam = cov_mat(model.final_layer.weight)
        A = torch.matmul(G,torch.matmul(lam,G.T)).numpy()
        B = G.numpy()
        try:
            eta, W = gen_eigh(A,B,tol)
        except:
            print("Error: PCA weights could not be computed!")
            return 0.
    return eta, W

def PCA(model,u,M=50000,tol=1e-10):
    d = u.shape[1]
    u_ = torch.from_numpy(np.array(u,dtype='float32'))
    weights = PCA_weights(model,d,M,tol)
    if weights==0.:
        print("Error: PCA could not be computed!")
        return 0.
    eta, W = weights
    with torch.no_grad():
        u_ = model.first_step(u_)
        W = torch.from_numpy(W)
        psi = torch.matmul(u_,W).numpy()
    return eta, psi

def PCA_weights_grid(model,d,res=100,tol=1e-10):
    """
    Finds the eigen-decomposition of the CovNet model stored in 'model'
    Returns the eigen-values and coefficients of the eigen-functions
    Numerical approximation of the integrals are done by complete enumeration on a grid of resolution 'res'
    """
    u = locations(res,d)
    u = torch.from_numpy(u)
    M = u.shape[0]
    with torch.no_grad():
        G = model.first_step(u)
        G = torch.matmul(G.T,G)/M
        del u
        lam = cov_mat(model.final_layer.weight)
        A = torch.matmul(G,torch.matmul(lam,G.T)).numpy()
        B = G.numpy()
        try:
            eta, W = gen_eigh(A,B,tol)
        except:
            print("Error: PCA weights could not be computed!")
            return 0.
    return eta, W

def PCA_grid(model,u,res=100,tol=1e-10):
    """
    Finds the eigen-decomposition of the CovNet model stored in 'model'
    Returns the eigen-values and evaluations of the eigen-functions at locations specified by 'u' 
    Numerical approximation of the integrals are done by complete enumeration on a grid of resolution 'res'
    """
    d = u.shape[1]
    u_ = torch.from_numpy(np.array(u,dtype='float32'))
    weights = PCA_weights_grid(model,d,res,tol)
    if weights==0.:
        print("Error: PCA could not be computed!")
        return 0.
    eta, W = weights
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
