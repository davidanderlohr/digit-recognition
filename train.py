import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Use hardware accelerator if available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Layer: 28*28 pixels as input, 128 output features
        self.fc1 = nn.Linear(28 * 28, 128)
        # Hidden Layer 1
        self.fc2 = nn.Linear(128, 64)
        # Hidden Layer 2
        self.fc3 = nn.Linear(64, 32)
        # Output Layer, 10 output features, representing digits 0-9
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        # Flatten image to 1D shape
        x = x.view(-1, 28 * 28)
        # Pass through the layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        # Softmax so that output is always between 0 and 1 and all outputs sum to 1
        output = torch.log_softmax(x, dim=1)
        return output
    

net = Net().to(device)

# Choose loss function: Combination of LogSoftmax and Negative Log Likelihood Loss
criterion = nn.CrossEntropyLoss()

# Choose optimizer: Adaptive Moment Estimation, learning rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
num_epochs = 200

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100}")
            running_loss = 0.0


print('Finished Training')


# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

# Save the model
torch.save(net.state_dict(), "model.pth")