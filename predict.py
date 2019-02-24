#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import torch
import numpy
import torchvision
from utils import imshow_

def predict(dataloader, model):
    '''
    Make a prediction
    Args:
        dataloader: dataloader to predict from
        model : trained model

    Returns:
        preds (np array): array of predicted classes
        labels (np array): array of actual classes
        probs (np array): array of probabilites for predictions
        images (np array): array of images
            
    '''
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images.cuda()

    # compute predicted outputs by passing inputs to the model
    model.eval() # eval mode
    # get sample outputs
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    labels = labels.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    probs = outputs.cpu().detach().numpy()
    # return preds, labels, images
    return preds,labels,probs, images.cpu()


def plot_result(preds, labels, images):
  '''
  Display 5 images along with labels and predictions
          Args:
              preds (np array): array of predicted classes
              labels (np array): array of actual classes
              images (np array): array of images

  '''
  # set the title of the plot
  title = ""
  for ii in range(5):
      title = title + 3 * "-" + " {}({}) ".format(preds[ii], labels[ii]) + 3 * "-"
  imshow_(torchvision.utils.make_grid(images[:5]),ax =(16,6), title=title )