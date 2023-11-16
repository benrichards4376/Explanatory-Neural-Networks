#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        #ctx.save_for_backward(input)
        
        # You can perform your custom operation here if needed
        #output = input  # For demonstration, it just passes input as output
        
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Print information during the backward pass
        #print("Backward pass: Gradients shape:", grad_output.shape)
        #print("Gradients values:", grad_output)
        #custom_gradients = input.clone().detach()
        #print("Custom Gradients:", custom_gradients)
        # Retrieve the input from the forward pass
        #input, = ctx.saved_tensors
        # You can perform your custom gradient calculation here if needed
        #grad_input = grad_output  # For demonstration, it just passes grad_output as grad_input
        # Calculate custom gradients (modify as needed)
        #custom_gradients = input.clone().detach()
        # Return custom gradients for backpropagation
        #return custom_gradients * grad_output
        
        input, = ctx.saved_tensors
        
        # the gradient of the loss with respect to the output of the custom autograd function is grad_output
        if(grad_output[0,0].item()!= 0):
            print("Output Gradient:", grad_output[0,0].item())
        
        return grad_output


# In[2]:


# Define a simple neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.custom_layer = CustomAutogradFunction.apply
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        #x = CustomAutogradFunction.apply(x)
        x = self.fc1(x)
        x = self.custom_layer(x)        
        x = self.relu(x)
        x = self.fc2(x)
        return x


# In[3]:


# Hyperparameters
input_size = 784  # MNIST images are 28x28 pixels
hidden_size = 128
output_size = 10  # 10 classes (digits 0-9)
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# In[4]:


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the model and optimizer
model = NeuralNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# In[5]:


# Training Loop
i = 0
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)  # Flatten the input
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        #print(model.fc1.weight.grad.data[:])
        
        fc1_weights_neuron1_gradients = model.fc1.weight.grad[0].detach().numpy()
        fc1_weights_neuron1 = model.fc1.weight[0].detach().numpy()
        
        if(fc1_weights_neuron1_gradients[0] != 0):
            print("Neuron 0 in Layer 1 at iteration ", i,"Gradient: ", fc1_weights_neuron1_gradients[0], "Weights: ",  fc1_weights_neuron1[0])
        #print("Default Gradient: ", fc1_weights_neuron1_gradients[0])
        i = i + 1
print('Training finished.')


# In[6]:


fc1_weights_neuron1 = model.fc1.weight[0].detach().numpy()
print(fc1_weights_neuron1[0])


# In[7]:


#This is the gradient of the loss with respect to the weights of the fc1 layer in the neural network
fc1_weights_neuron1_gradients = model.fc1.weight.grad[0].detach().numpy()
print(fc1_weights_neuron1_gradients[0])


# In[8]:


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)  # Set batch_size to 1 for individual testing

# Test the model on the test dataset and print the labels every 100 tests
model.eval()  # Set the model to evaluation mode

total = 0
correct = 0

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader, 1):
        # Flatten the input images
        images = images.view(-1, 28 * 28)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += 1

        if i % 1000 == 0:
            print(f'Test {i}: Actual Label: {labels.item()}, Predicted Number: {predicted.item()}')

# Calculate and print the total accuracy
accuracy = 100 * correct / total
print(f'Total Accuracy: {accuracy:.2f}%')


# In[ ]:





# In[ ]:




