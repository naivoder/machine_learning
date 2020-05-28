"""
this file defines a simple neural network with pytorch, built from the tutorials at pytorch.org
you are only required to define the forward() function, the backward() function is autmatically defined using autograd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel
        # 6 output channels
        # 3 x 3 square convolution
        self.conv_layer1 = nn.Conv2d(1, 6, 3)
        self.conv_layer2 = nn.Conv2d(6, 16, 3)
        # affine operation: y = Wx + b
        self.fully_connect1 = nn.Linear(16 * 6 * 6, 120)
        self.fully_connect2 = nn.Linear(120, 84)
        self.fully_connect3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over 2x2 window
        x = F.max_pool2d(F.relu(self.conv_layer1(x)), (2, 2))
        # conv_layer2 is square, can only specify single number...
        x = F.max_pool2d(F.relu(self.conv_layer2(x)), 2)
        # flatten into 1d tensor
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fully_connect1(x))
        x = F.relu(self.fully_connect2(x))
        x = self.fully_connect3(x)
        return x

    def num_flat_features(self, x):
        # omit batch dimension
        size = x.size()[1:]
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features

if __name__ == "__main__":
    net = Net()
    print(net)
    # learnable parameters
    params = list(net.parameters())
    print(len(params))
    # conv_layer1's weight
    print(params[0].size())
    # random 32x32 input
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
    # zero gradient buffers of all parameters
    net.zero_grad()
    out.backward(torch.randn(1, 10))
    # !! torch only supports mini batches
    # e.g. nn.conv2d expects (nSamples, nChannels, H, W)
    # to run a single sample: input.unsqueeze(0)
    # --> adds fake batch dimension
    output = net(input)
    target = torch.randn(10)
    # -1 'infers' correct dimension
    target = target.view(1, -1)
    # define loss function
    criteria = nn.MSELoss()
    loss = criteria(output, target)
    print(loss)
    # follow loss backwards for graph of computations
    # MSELoss
    print(loss.grad_fn)
    # Linear
    print(loss.grad_fn.next_functions[0][0])
    # ReLU
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
    # to backpropagate the error, run loss.backward()
    # must first clear existing gradient
    net.zero_grad()
    print("conv_layer1 bias gradient before backpropation")
    print(net.conv_layer1.bias.grad)
    loss.backward()
    print("conv_layer1 bias gradient after backpropation")
    print(net.conv_layer1.bias.grad)
    # documentation on other modules and loss functions
    # https://pytorch.org/docs/stable/nn.html
    # update weights using Stochastic Gradient Descent (SGD)
    # weight = weight - learning_rate * gradient
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(learning_rate * f.grad.data)
    # use torch.optim package for update rules 
    # optimzier = optim.CHOICE(net.parameters(), lr=0.1)
    # in your training loop:
    # -> zero the gradient buffers
    # optimizer.zero_grad()
    # output = net(input)
    # loss = criterion(output, target)
    # loss.backward()
    # -> update
    # optimizer.step()
