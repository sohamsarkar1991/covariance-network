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

def print_locations(K,d,file = 'locations.dat'):
    u = locations(K,d)
    np.savetxt(file,u,fmt='%.10f')
    
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
        s1,t1 = u_[i,:]
        for j in range(i,D):
            s2,t2 = u_[j,:]
            C[i,j] = method(s1,s2) * method(t1,t2)
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
        s1,t1 = u_[i,:]
        s2,t2 = v_[i,:]
        C[i] = method(s1,s2) * method(t1,t2)
    return C

def sqrt_mat(C):
    lam, E = np.linalg.eigh(C)
    idx = lam>0.
    lam = np.sqrt(lam[idx])
    E = E[:,idx]
    return [lam,E]


def datagen_and_print(N,K,d,replicates=1,method=None,rot=None,M=10000):
    """
    Data generation from superpositions of separable processes.
    INPUT:
        N - number of observations to be generated
        K - grid size in 1D
        d - dimension
        replicates - number of replications
        method - process for each separable component
        rot - rotation matrix, to be applied
    OUTPUT:
        Generates a covariance by rotation of a separable covariance. 
        Writes the covariance on the file 'True_cov.dat'. Writes the 
        locations of the grid points on the file 'locations.dat'. 
        Generates N surfaces at the grid points and writes them on
        'Examplex.dat', for x=1,...,replicates.
    """
    print('Basis generation started')
    lam, E = sqrt_mat(cov_mat(K,d,method,rot))
    print('Basis generated')
    
    for repl in range(replicates):
        #np.random.seed(int(np.random.rand()*(2**32-1)))
        print('Replicate '+str(repl+1))
        filename = 'locations'+str(repl+1)+'.dat'
        print_locations(K,d,filename)
        filename = 'Example'+str(repl+1)+'.dat'
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
        np.savetxt('True_locations'+str(repl+1)+'.dat',np.hstack((u,v)),fmt='%.10f')
        print_indices(u,v,K,d,'indices'+str(repl+1)+'.dat')
        C = cross_cov(u,v,method,rot)
        np.savetxt('True_cov'+str(repl+1)+'.dat',C,fmt='%.10f')
        del C, u, v
    
def cov_data_gen(N,K,d,method=None,rot=None):
    """
    Data generation from superpositions of separable processes.
    INPUT:
        N - number of observations to be generated
        K - grid size in 1D
        d - dimension
        replicates - number of replications
        method - process for each separable component
        rot - rotation (in radian) to be applied
    OUTPUT:
        Generates a covariance by rotation of a separable covariance. 
        Writes the covariance on the file 'True_cov.dat'. Writes the 
        locations of the grid points on the file 'locations.dat'. 
        Generates N surfaces at the grid points and writes them on
        'Examplex.dat', for x=1,...,replicates.
    """
    
    C = cov_mat(K,d,method,rot)
    lam, E = sqrt_mat(C)
    #del C
    D = int(K**d)    
    X = np.zeros([N,D],dtype=float)
    
    for n in range(N):
        x = np.random.normal(loc=0.,scale=1.,size=len(lam))
        x = np.sum(E*lam*x,axis=1)
        X[n,:] = x
        
    return C,X
    
