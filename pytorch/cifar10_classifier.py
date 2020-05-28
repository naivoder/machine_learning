"""
this file uses pytorch and the torchvision package to train an image classifier on the cifar10 dataset
classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’
image size: 3x32x32

"""

# general process:
# ----------------
# 1) load and normalize datasets
# 2) define convolutional neural network
# 3) define loss function
# 4) train network on training data
# 5) test network on test data

import torch
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

### load and normalize cifar10
#-----------------------------

# output of torchvision datasets are PILImages, range [0, 1]
# transform them to tensors, normalized range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# training dataset
training_set = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)

# test dataset
test_set = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

# function to display images
def show_image(img):
    # unnormalize
    img = img / 2 + 0.5
    # convert tensor to numpy array
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

# select random training images for inspection
# will return 'batch_size' number of images
data_iter = iter(training_loader)
images, labels = data_iter.next()

# show images, print labels
show_image(tv.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

### define convolutional neural network
#--------------------------------------
