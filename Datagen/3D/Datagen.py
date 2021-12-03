#!/usr/bin/env python
# coding: utf-8


import covariance_multivariate_domain3 as cmult
import numpy as np
np.random.seed(12345)

N = 512
K = 25
d = 3
M = 100000
permute_dims = True

method = lambda s,t: cmult.matern_cov(s,t,0.01)
#theta1 = np.pi/4
#theta2 = np.pi/4
#theta3 = np.pi/4
#Ox = np.array([[1.,0.,0.],[0.,np.cos(theta1),-np.sin(theta1)],[0.,np.sin(theta1),np.cos(theta1)]])
#Oy = np.array([[np.cos(theta2),0.,np.sin(theta2)],[0.,1.,0.],[-np.sin(theta2),0.,np.cos(theta2)]])
#Oz = np.array([[np.cos(theta3),-np.sin(theta3),0.],[np.sin(theta3),np.cos(theta3),0.],[0.,0.,1.]])
#O = np.matmul(Oz,np.matmul(Oy,Ox))
O = None

cmult.datagen_and_print(N,K,d,25,method,O,M,permute_dims)