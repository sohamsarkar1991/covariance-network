#!/usr/bin/env python
# coding: utf-8


import numpy as np
from scipy import special
import itertools


def locations(K,d):
    gr = np.arange(1,K+1)/(K+1)
    grid = list(itertools.product(gr,repeat=d))
    grid = np.asarray(grid)
    return grid

def locations_unif(M,d):
    loc = np.random.uniform(low=0.,high=1.,size=[M,d])
    return loc

def print_locations(K,d,file):
    u = locations(K,d)
    np.savetxt(file,u,fmt='%.10f')
    
def pixelate_single(s,K):
    for j in np.arange(1,K):
        if s<(j+0.5)/(K+1):
            return j-1
    return K-1

def pixelate(u,K):
    idx = []
    for s in u:
        idx.append(pixelate_single(s,K))
    return idx

def indices(u,K,d):
    idx = np.apply_along_axis(pixelate,1,u,K)
    c = np.zeros(idx.shape[0],dtype=int)
    for i in range(d):
        c += idx[:,i]*K**(d-1-i)
    return c

def print_indices(u,v,K,d,file='indices.dat'):
    idx1 = indices(u,K,d)
    idx2 = indices(v,K,d)
    idx = np.vstack((idx1,idx2))
    np.savetxt(file,idx,fmt='%d')
    
def indices_perm(u,K,d,perm=None):
    idx = np.apply_along_axis(pixelate,1,u,K)
    if perm is not None:
        idx = idx[:,perm]
    c = np.zeros(idx.shape[0],dtype=int)
    for i in range(d):
        c += idx[:,i]*K**(d-1-i)
    return c

def print_indices_perm(u,v,K,d,perm=None,file='indices_PSCA3D.dat'):
    idx1 = indices_perm(u,K,d,perm)
    idx2 = indices_perm(v,K,d,perm)
    idx = np.vstack((idx1,idx2))
    np.savetxt(file,idx,fmt='%d')
    
def bessel(s,t,nu):
    r = abs(s-t)
    c = 2**nu * special.gamma(nu+1) * r**(-nu) * special.jv(nu,r)
    return c

def BM(s,t):
    if s<=t and s>0.:
        return s
    elif s>t and t>0.:
        return t
    elif s<0. and t<0.:
        return BM(-s,-t)
    else:
        return 0.

def fBM(s,t,alpha):
    c = 0.5 * (abs(t)**(2*alpha) + abs(s)**(2*alpha) - abs(t-s)**(2*alpha))
    return c

def fBM2(s,t,H):
    alpha = 2*H
    if s<=t and s>0.:
        return 0.5*(t**alpha + s**alpha - (t-s)**alpha)
    elif s>t and t>0.:
        return 0.5*(t**alpha + s**alpha - (s-t)**alpha)
    elif s<0. and t<0.:
        return fBM2(-s,-t,H)
    else:
        return 0.

def iBM(s,t):
    if s<=t and s>0.:
        return s**2*t/2 - s**3/6
    elif s>t and t>0.:
        return t**2*s/2 - t**3/6
    elif s<0. and t<0.:
        return iBM(-s,-t)
    else:
        return 0.

def Bbridge(s,t,T=1.):
    if s<=t and s>0.:
        return s*(T-t)/T
    elif s>t and t>0.:
        return t*(T-s)/T
    elif s<0. and t<0.:
        return Bbridge(-s,-t,T)
    else:
        return 0.

def cauchy(s,t,gamma):
    r = abs(s-t)
    return 1./(1+r**2)**gamma

def dampedcos(s,t,lam):
    r = abs(s-t)
    c = np.exp(-lam*r) * np.cos(r)
    return c

def fractgauss(s,t,alpha):
    r = abs(s-t)
    c = 0.5*((r+1)**alpha + abs(r-1)**alpha - 2*r**alpha)
    return c

def matern(s,t,nu,rho=1.):
    r = abs(s-t)/rho
    if r==0:
        r = np.finfo(float).eps
    if nu == 0.5:
        c = np.exp(-r)
    elif nu == 1.5:
        c = r * np.sqrt(3)
        c = (1.+c) * np.exp(-c)
    elif nu == 2.5:
        c = r*np.sqrt(5)
        c = (1.+c+c**2/3.) * np.exp(-c)
    elif nu == np.inf:
        c = np.exp(-r**2/2.)
    else:  # general case; expensive to evaluate
        tmp = np.sqrt(2*nu) * r
        c = 2**(1.-nu) / special.gamma(nu)
        c *= tmp ** nu
        c *= special.kv(nu,tmp)
    return c

def cov_mat(K,d,method,O=None):
    if O is None:
        O = np.eye(d)
    u = locations(K,d)
    u_ = np.matmul(u,O.T)
    del u
    D = int(K**d)
    C = np.zeros([D,D],dtype=float)
    for i in range(D):
        r1,s1,t1 = u_[i,:]
        for j in range(i,D):
            r2,s2,t2 = u_[j,:]
            C[i,j] = method(r1,r2) * method(s1,s2) * method(t1,t2)
            C[j,i] = C[i,j]
    return C

def cross_cov(u,v,method,O=None):
    if O is None:
        d = u.shape[1]
        O = np.eye(d)
    if u.shape != v.shape:
        print('Mismatch in shapes of u and v!')
        return
    M = u.shape[0]
    u_ = np.matmul(u,O.T)
    v_ = np.matmul(v,O.T)
    C = np.zeros(M,dtype=float)
    for i in range(M):
        r1,s1,t1 = u_[i,:]
        r2,s2,t2 = v_[i,:]
        C[i] = method(r1,r2) * method(s1,s2) * method(t1,t2)
    return C

def sqrt_mat(C):
    lam, E = np.linalg.eigh(C)
    idx = lam>0.
    lam = np.sqrt(lam[idx])
    E = E[:,idx]
    return [lam,E]


def datagen_and_print(N,K,d,method=None,rot=None,M=10000,permute_dims=False):
    print('Basis generation started')
    lam, E = sqrt_mat(cov_mat(K,d,method,rot))
    print('Basis generated')
    
    #np.random.seed(int(np.random.rand()*(2**32-1)))
    filename = 'locations.dat'
    print_locations(K,d,filename)
    filename = 'Example.dat'
    f=open(filename,'w')
    f.close()
    f = open(filename,'a')
    for n in range(N):
        x = np.random.normal(loc=0.,scale=1.,size=len(lam))
        x = np.sum(E*lam*x,axis=1)
        x = x.reshape(1,-1)
        np.savetxt(f,x,fmt='%.10f')
    f.close()
    u = locations_unif(M,d)
    v = locations_unif(M,d)
    np.savetxt('True_locations.dat',np.hstack((u,v)),fmt='%.10f')
    print_indices(u,v,K,d,'indices.dat')
    if permute_dims:
        print_indices_perm(u,v,K,d,[0,1,2],'indices_PSCA3D_123.dat')
        print_indices_perm(u,v,K,d,[1,0,2],'indices_PSCA3D_213.dat')
        print_indices_perm(u,v,K,d,[2,0,1],'indices_PSCA3D_312.dat')
    C = cross_cov(u,v,method,rot)
    np.savetxt('True_cov.dat',C,fmt='%.10f')
    del u,v,C