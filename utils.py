#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt


def imshow_(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))
    else:
        fig, ax = plt.subplots(figsize=ax)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.set_title("{}".format(title))
    ax.grid(False)
    plt.axis('off')
    ax.imshow(image)
    
    return ax