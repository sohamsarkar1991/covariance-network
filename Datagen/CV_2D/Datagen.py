#!/usr/bin/env python
# coding: utf-8

import datagen_nonseparable2 as dnsep
import numpy as np
np.random.seed(12345)

N = 500
K = 10
print_cov = False
K_tr = 100
d = 2
M = 50000
method = lambda s,t: dnsep.iBM(s,t)
theta = np.pi/4
O = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
#O = None

dnsep.datagen_and_print(N,K,d,method,O,M,print_cov,K_tr)
