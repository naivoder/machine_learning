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

# set device type
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

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

import matplotlib.pyplot as plt
import numpy as np

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
# data_iter = iter(training_loader)
# images, labels = data_iter.next()
#
# show images, print labels
# show_image(tv.utils.make_grid(images))
# print('Labels:', ' '.join('%s' % classes[labels[j]] for j in range(4)))

### define convolutional neural network
#--------------------------------------

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channels
        # 6 output channels
        # 5 x 5 square convolution
        self.conv_1 = nn.Conv2d(3, 6, 5)
        # better to define pooling here than in forward()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        # affine operation: y = Wx + b
        self.fconn_1 = nn.Linear(16 * 5 * 5, 120)
        self.fconn_2 = nn.Linear(120, 84)
        self.fconn_3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        # flatten into 1d tensor
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fconn_1(x))
        x = F.relu(self.fconn_2(x))
        x = self.fconn_3(x)
        return x

    def num_flat_features(self, x):
        # omit batch dimension
        size = x.size()[1:]
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features

net = Net()
# if cuda available, use gpu for training
net.to(device)

### define loss function and optimizer
#-------------------------------------

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

### train network on training data (cpu first)
#---------------------------------------------

# loop over data iterator, feed inputs to network, optimize
print('Training two epochs...')
# define number of loops through dataset
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # get inputs and labels, use gpu if available
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero parameter gradient
        optimizer.zero_grad()

        # forward -> backward -> optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics every 2000 batches
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[Epoch: %d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print("Training Completed.")

# save trained model (not necessary here)
# https://pytorch.org/docs/stable/notes/serialization.html
PATH = './cifar_net.path'
torch.save(net.state_dict(), PATH)

### test network on test data
#----------------------------

# select random test images for inspection
# will return 'batch_size' number of images
data_iter = iter(training_loader)
images, labels = data_iter.next()
images, labels = images.to(device), labels.to(device)

# show images, print labels
# show_image(tv.utils.make_grid(images))
print('Ground Truth:', ' '.join('%s' % classes[labels[j]] for j in range(4)))

# load saved model (not necessary here)
# included for instruction
# PATH = './cifar_net.path'
# net = Net()
# net.load_state_dict(torch.load(PATH))
# net.to(device)

# output are energies for 10 classes, 0-1
outputs = net(images)
# see if network learned anything...
stuff, prediction = torch.max(outputs, 1)
# seems correct...
print('Predicted:', ' '.join('%s' % classes[prediction[j]] for j in range(4)))

# check performance on whole test set
correct = 0
total = 0
# don't keep track of tensor operations
with torch.no_grad():
    for data in test_loader:
        # get inputs and labels
        images, labels = data[0].to(device), data[1].to(device)
        output = net(images)
        # why output.data here, but not 159?
        stuff, prediction = torch.max(output.data, 1)
        total += labels.size(0)
        # handy trick to get total number of correct guesses
        correct += (prediction == labels).sum().item()
# ~ 50%, better than random chance (10% for 10 classes)
print('Accuracy on 10,000 test images: %d%%' % (100 * correct / total))

# inspect for classes that performed well/poorly
class_correct = [0. for i in range(10)]
class_total = [0. for i in range(10)]
with torch.no_grad():
    for data in test_loader:
        # get inputs and labels
        images, labels = data[0].to(device), data[1].to(device)
        output = net(images)
        stuff, prediction = torch.max(output, 1)
        # same handy trick but this time with squeeze...
        correct = (prediction == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy (%5s) : %2d%%' % (classes[i], 100 * class_correct[i] / class_total[i]))
