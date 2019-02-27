## Flower Classifier

This was the capstone project of the Udacity PyTorch Scholarship Challenge. The challenge was to create a flower classifier using transfer learning. Given a user input of an image of a flower, the flower type was predicted. 

### Data: Oxford Flower Dataset

[Oxford flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) 102 classes of flowers common to the UK. Statistics for the dataset classes and their distribution can be found [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html).

### Data Preprocessing

I chose the following transforms:

```python
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(10), 
transforms.CenterCrop(350),
transforms.ColorJitter(brightness = .1, saturation=.1),
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]) 
```

For augmentation: 

RandomHorizontalFlip to flip the image horizontally.

ColorJitter to account for possible differences in lighting, contrast, and saturation within the photos.

RandomRotation to add variability to the photos so the network is able to generalize and adapt to various positions and angles of the flowers within the photos.

For the pretrained model:

I resized to 224 and normalized using the standard ImageNet values because that's what a pretrained (ImageNet) network would require. Lastly, I converted the array to a tensor to be fed into the network.

### Transfer Learning: Resnet152

 I chose Resnet152 because of its depth and performance; I tried shallower networks and did not achieve as high of accuracy. Initially, I froze all layers, replacing and training only the classifier. For the classifier, I opted for a single fc layer. With this configuration, I was able to achieve 92% accuracy after just 10 epochs.

### Fine-tuning

To achieve greater accuracy in the challenge, I used a stepLR, and I unfroze the 4th layer in addition to the fc layer. Although it's not displayed in the notebook I've uploaded to GitHub, I was able to achieve 96% accuracy through fine-tuning.

### Final Thoughts

Unfreezing more layers and training at a progressively higher learning-rate might help increase this accuracy even more. I could also continue to experiment with different augmentation techniques, as some classes were very imbalanced (containing only a few images per class). 
