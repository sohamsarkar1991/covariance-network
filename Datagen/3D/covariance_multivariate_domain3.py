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

def gneiting_cov(u,v,beta=0.,m=1.):
    """ the Gneiting (2002) model
    beta - separability parameter (in [0,1]; separable when beta=0)
    INPUT: u,v - points at which the covariance is evaluated
    beta - in [0,1]
    OUTPUT: C(u,v)
    """
    gamma=1
    alpha=1
    a=m**(2*alpha)
    const=m**(2*gamma) #c in Bagchi & Dette(2017)
    tau=1
    sigma2=1
    Dt = a*abs(u[0]-v[0])**(2*alpha)+1
    Cov = const*((np.linalg.norm(u[1:]-v[1:]))**(2*gamma))/(Dt**(beta*gamma))
    Cov = sigma2/(Dt**tau) * np.exp(-Cov)
    return Cov

def cressiehuang_cov(u,v,c0=1.,m=1.):
    """ the Cressie & Huang (1999) model
    c0 - separability parameter (in [1,Inf); separable when c0=1)
    INPUT: u,v - points at which the covariance is evaluated
            c0 - in [1,Inf)
    OUTPUT: C(u,v)
    """
    a0=m
    b0=m
    sigma2=1
    d=len(u)-1 # spatial dimension
    Dt = (a0**2)*((u[0]-v[0])**2)
    Cov = b0*np.linalg.norm(u[1:]-v[1:])*np.sqrt((Dt+1)/(Dt+c0))
    Cov = (sigma2*(c0**(d/2)))/(np.sqrt(Dt+1)*((Dt+c0)**(d/2))) * np.exp(-Cov)
    return Cov

def matern_cov(u,v,nu=0.5,rho=1.):
    """ the Matern covariance model
    
    """
    r = np.linalg.norm(u-v)/rho
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
    
    D = u_.shape[0]
    C = np.empty([D,D],dtype=float)

    for i in range(D):
        for j in range(i,D):
            C[i,j] = method(u_[i,:],u_[j,:])
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
        C[i] = method(u_[i,:],v_[i,:])
    return C

def sqrt_mat(C):
    lam, E = np.linalg.eigh(C)
    idx = lam>0.
    lam = np.sqrt(lam[idx])
    E = E[:,idx]
    return [lam,E]

def datagen_and_print(N,K,d,replicates=1,method=None,rot=None,M=10000,permute_dims=False):
    print('Basis generation started')
    lam, E = sqrt_mat(cov_mat(K,d,method,rot))
    print('Basis generated')
    
    for repl in range(replicates):
        #np.random.seed(int(np.random.rand()*(2**32-1)))
        print('Replicate {}'.format(repl+1))
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
        if permute_dims:
            print_indices_perm(u,v,K,d,[0,1,2],'indices_PSCA3D_123_'+str(repl+1)+'.dat')
            print_indices_perm(u,v,K,d,[1,0,2],'indices_PSCA3D_213_'+str(repl+1)+'.dat')
            print_indices_perm(u,v,K,d,[2,0,1],'indices_PSCA3D_312_'+str(repl+1)+'.dat')
        C = cross_cov(u,v,method,rot)
        np.savetxt('True_cov'+str(repl+1)+'.dat',C,fmt='%.10f')
        del u,v,C