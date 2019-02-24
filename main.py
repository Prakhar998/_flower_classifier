#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
from torchvision import models, transforms
from torch import nn, optim
from datasets_and_dataloaders import *
import json
from utils import imshow_
from train import train_model
from predict import *

# how many samples per batch to load
batch_size = 32
# percentage of training set to use as validation
valid_size = 0.15

# create transforms
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
   ])

train_transforms = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10), 
      transforms.CenterCrop(350),
      transforms.ColorJitter(brightness = .1, saturation=.1),
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])    
  ])

# create dict of transforms
transforms_ = {"train": train_transforms, "test":test_transforms}

# create dict of directories
train_dir = 'train/'
valid_dir = 'valid/'
directories = {"train":train_dir, "test":valid_dir}

# create dict of classes/labels
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
num_classes = len(cat_to_name)
print(f"Number of classes: {num_classes}")

# create datasets
datasets_ = create_datasets(directories, transforms_, split=True, split_amount=.15)
train_data_idxed = [{'name':cat_to_name[str(int(x))], 'idx':i} for i,x in enumerate(datasets_['test'].classes)]

# create dataloaders
dataloaders = create_dataloaders(datasets_, batch_size)
train_dataloader = dataloaders['train']
valid_dataloader = dataloaders['valid']
test_dataloader = dataloaders['test']

# create dataset_sizes
dataset_sizes = {"train": len(train_dataloader), "valid": len(valid_dataloader), "test": len(test_dataloader)}

# create network
model = models.resnet152(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
# resnet156: replace fc layer with our own
in_features = model.fc.in_features 
out_features = num_classes
model.fc = nn.Linear(in_features, out_features)

# define criterion, scheduler and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = 0.0003)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1,last_epoch=-1)
epochs = 5
save_path = "resnet152-v1.pt"
# train the model
trained_model = train_model(dataloaders = dataloaders,
                            dataset_sizes = dataset_sizes,
                            model=model,
                            criteria=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            save_path=save_path,
                            num_epochs=epochs,
                            )

