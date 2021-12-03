#!/usr/bin/env python
# coding: utf-8

import covariance_multivariate_domain2 as cmult
import numpy as np
np.random.seed(12345)

N = 500
K = 25
print_cov = False
K_tr = 100
d = 2
M = 50000
method = lambda u,v: cmult.matern_cov(u,v,0.01)
#method = lambda u,v: gneiting_cov(u,v,beta=0.7)
#method = lambda u,v: cressiehuang_cov(u,v,c0=5.)
#theta = np.pi/4
#O = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
O = None

cmult.datagen_and_print(N,K,d,method,O,M,print_cov,K_tr)