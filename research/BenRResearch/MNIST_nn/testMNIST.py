import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the pre-trained model
model = torch.load('numbers.pth')

# Define a transformation to preprocess the input data (similar to training)
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
