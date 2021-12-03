"""
Parameters used to fit the CovNet models
"""

import numpy as np
import torch
import Important_functions as Ifn
import Other_functions as Ofn

act_fn = torch.nn.Sigmoid()
init = torch.nn.init.xavier_normal_

optimizer = torch.optim.Adam
lr=0.01
split = lambda D: Ifn.batch_CV(D,batch_size=500)
scheduler = None
epochs_shallow = 10000
burn_in_shallow = 7500
interval_shallow = 50
epochs_deepshared = 10000
burn_in_deepshared = 7500
interval_deepshared = 50
epochs_deep = 5000
burn_in_deep = 3500
interval_deep = 50
folds = 5
directory = '/CovNet/2D/Datagen/' #should be the directory where the data are located
replicates = range(25)