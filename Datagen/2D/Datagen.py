#!/usr/bin/env python
# coding: utf-8

import datagen_nonseparable2 as dnsep
import numpy as np
np.random.seed(12345)

N = 512
K = 80
d = 2
M = 50000
replicates = 25
method = lambda s,t: dnsep.iBM(s,t)
#theta = np.pi/4
#O = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
O = None

dnsep.datagen_and_print(N,K,d,replicates,method,O,M)