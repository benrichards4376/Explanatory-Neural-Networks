#!/usr/bin/env python
# coding: utf-8

# In[22]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Function

class CustomAutogradFunction(Function):
    @staticmethod
    def forward(ctx, input):
        
        # Store input for use in the backward pass
        ctx.save_for_backward(input)
        
        # You can perform your custom operation here if needed
        output = input  # For demonstration, it just passes input as output
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Print information during the backward pass
        print("Backward pass: Gradients shape:", grad_output.shape)
        print("Gradients values:", grad_output)
        
        # Retrieve the input from the forward pass
        input, = ctx.saved_tensors
        
        # You can perform your custom gradient calculation here if needed
        grad_input = grad_output  # For demonstration, it just passes grad_output as grad_input
        
        return grad_input


# In[23]:


# Define a simple neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        self.custom_layer = CustomAutogradFunction.apply
        
        self.explainable = [self.fc1.weight.data[:], self.fc1.bias.data[:]]
        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.custom_layer(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# In[24]:


# Hyperparameters
input_size = 784  # MNIST images are 28x28 pixels
hidden_size = 128
output_size = 10  # 10 classes (digits 0-9)
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# In[25]:


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the model and optimizer
model = NeuralNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# In[26]:




# Training Loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)  # Flatten the input
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

print('Training finished.')
torch.save(model, 'numbers.pth')


# In[15]:


print(model.explainable)


# In[10]:


print(model.fc1.weight.data[:])


# In[ ]:




