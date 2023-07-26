#!/usr/bin/env python
# coding: utf-8

##### Important functions for fitting the CovNet model #####


import torch
import numpy as np

##### Train-validation splitting #####
##### Using mini batches #####

def batch_CV(D,batch_size=10): # points randomly shuffled each time, iterated through all points
    if batch_size==None or batch_size>D:
        batch_size = D
    folds = int(D/batch_size)
    Q = []
    indices = np.random.choice(D,D,replace=False)
    for fold in range(folds):
        Q_tr = np.sort(indices[fold*batch_size:(fold+1)*batch_size])
        #Q_va = np.sort(np.random.choice(indices[~Q_tr],batch_size,replace=False))
        #Q.append((Q_tr,Q_va))
        #del Q_tr,Q_va
        Q.append(Q_tr)
        del Q_tr
    return Q

##### Loss functions #####

def loss_COV(x,x_hat):
    """
    |||\widehat{C}_N - \widehat{\widehat{C}}_N|||_2^2
    \widehat{C}_N - empirical covaraince of X_1,\ldots,X_N
    \widehat{\widehat{C}}_N - empirical covariance of X^{NN}_1,\ldots,X^{NN}_N
    """
    D = x.shape[1]
    #x_hat = x_hat - torch.mean(x_hat,dim=0,keepdim=True)
    l1 = torch.matmul(x,x.T)/D
    l2 = torch.matmul(x_hat,x_hat.T)/D
    l3 = torch.matmul(x,x_hat.T)/D
    l = torch.mean(l1**2) + torch.mean(l2**2) - 2*torch.mean(l3**2) + torch.mean(l1)**2 + torch.mean(l2)**2 - 2*torch.mean(l3)**2
    return l

def loss_MSE(x,x_hat):
    """
    MSE loss -
    N^{-1}\sum_{n=1}^N \|X_n-X^{NN}_n\|^2
    """
    x_hat = x_hat - torch.mean(x_hat,dim=0,keepdim=True)
    l = torch.mean((x-x_hat)**2)
    return l

##### Early stopping routine #####
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=1e-4, filename='checkpoint_cnet.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            filename : file on which the best model will be stored
                            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.epoch = None
        self.early_stop = False
        self.delta = delta
        self.checkpoint = filename

    def __call__(self, val_loss, model, epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(model)
        elif score >= self.best_score or 1.-score/self.best_score < self.delta:
            #If validation error starts to increase or the decrease is less than the threshold
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.load_checkpoint(model)
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.epoch = epoch
            self.counter = 0

    def save_checkpoint(self, model):
        '''
        Save model when validation loss increase or doesn't decrease more than a certain level
        '''
        torch.save(model.state_dict(), self.checkpoint)
        
    def load_checkpoint(self, model):
        '''
        Reset the model to the saved checkpoint
        '''
        model.load_state_dict(torch.load(self.checkpoint))
        
##### Save and load best state_dict #####        
class BestState:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, filename='checkpoint_cnet.pt'):
        """
        Args:
            filename : file on which the best model will be stored
                            
        """
        self.best_score = None
        self.file = filename
        self.epoch = None

    def __call__(self, error, model,epoch):

        score = error

        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(model)
        elif score <= self.best_score:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        '''
        Save model
        '''
        torch.save(model.state_dict(), self.file)
        
    def load_checkpoint(self, model):
        '''
        Reset the model to the saved checkpoint
        '''
        model.load_state_dict(torch.load(self.file))


##### Optimization routine #####

def cnet_optim_best(x,u,model,loss_fn,optimizer,split,epochs=1000,burn_in=500,interval=1,checkpoint_file='Checkpoint.pt',loss_file='loss.txt'):
    """
    Optimization routine with on-the-go error computation. Returns the model state that produced the best error.
    INPUTS - x, u - data and locations
             model - the model to be fitted
             loss_fn - loss function (MSE/COV/COV2)
             optimizer - optimizer to be used. An element of class torch.optim
             split - training and validation splitter function
             epochs - number of epochs
             plot_filename - the filename in which the plots will be saved
             
    OUTPUTS - l_tr - training errors
              l_va - validation errors
    """
    D = u.shape[0]
    
    best_state = BestState(checkpoint_file)
    
    for epoch in range(burn_in):
        for Q_tr in split(D):
            loss = loss_fn(x[:,Q_tr],model(u[Q_tr,:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    for epoch in range(burn_in,epochs):
        train_losses = []
        for Q_tr in split(D):
            loss = loss_fn(x[:,Q_tr],model(u[Q_tr,:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        l_tr = np.mean(train_losses)
        with torch.no_grad():
            l_va = loss_fn(x,model(u)).item()
            del loss
        f = open(loss_file,"a")
        f.write("{:.10f}\t{:.10f}\n" .format(l_tr,l_va))
        f.close()
        if (epoch-burn_in)%interval == interval-1:
            best_state(l_va,model,epoch)

    best_state.load_checkpoint(model)
    epoch = best_state.epoch
    return epoch+1