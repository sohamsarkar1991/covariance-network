"""
Parameters used to fit the CovNet models
"""

import numpy as np
import torch
import Important_functions_fMRI as Ifn
import Other_functions as Ofn

act_fn = torch.nn.Sigmoid()
init = torch.nn.init.xavier_normal_

optimizer = torch.optim.Adam
lr=0.01
#split = lambda D: Ifn.split_CV(D,folds=20)
split = lambda D: Ifn.batch_CV(D,batch_size=500)
#lambdalr = lambda t:1/np.sqrt(t+1)
#scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer,lambdalr)
scheduler = None
epochs_shallow = 1000
burn_in_shallow = 700
interval_shallow = 10
epochs_deepshared = 1000
burn_in_deepshared = 700
interval_deepshared = 10
epochs_deep = 500
burn_in_deep = 350
interval_deep = 10
folds = 5
directory = '/home/ssarkar/CovNet/fMRI_3D/Data/'