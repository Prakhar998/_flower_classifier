#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Train a network
        Args:
            dataloader (dict) : dictionary with dataloaders
            dataset_sizes (dict) : dictionary of len of dataloaders
            model : model to be trained
            criteria : criterion
            optimizer : optimizer
            save path : file path to save the model to
            use_cuda (bool) : whether to use cuda or not (default=True)
            scheduler : learning rate scheduler
            num_epochs : num epochs to train for, (default = 25)
            plot (bool) : whether to plot or not

        Returns:
            model : trained model
'''
import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import copy 
import numpy as np
import matplotlib.pyplot as plt




def train_model(dataloaders, dataset_sizes, model, criteria, optimizer, scheduler, save_path, use_cuda=True, num_epochs=25, plot=False):
    if use_cuda:
        model.cuda()
    # track time elapsed
    since = time.time()
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    print("[+] Training Model...")
    best_acc = 0
    # save our best model
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        running_correct = 0

        ########// TRAIN //###########

        model.train()
        for batch_idx, (data, target) in enumerate(dataloaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # zero gradient
            optimizer.zero_grad()
            # forward pass
            outputs = model(data)
            # get loss
            loss = criteria(outputs, target)
            # backprop
            loss.backward()
            # update params
            optimizer.step()
            # update train loss
            train_loss += loss.item()*data.size(0)
            
            # statistics
            _,preds = torch.max(outputs, 1)
            running_correct += torch.mean((preds == target.data).type(torch.FloatTensor))
        epoch_loss = train_loss / dataset_sizes['train']
        epoch_acc = running_correct.double() / dataset_sizes['train']
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        ##########// VALIDATE //############    

        model.eval()
        for batch_idx, (data, target) in enumerate(dataloaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            valid_loss += loss.item()*data.size(0)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss/dataset_sizes['train'],
            valid_loss/dataset_sizes['valid']
            ))
        ## save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if plot:
        plt.plot(counter,loss_history)
        plt.show()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model