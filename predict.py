from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch
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

# Load the saved model
net = Net()
net.load_state_dict(torch.load('model.pth'))
net.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Apply the transformations
    img = transform(img)

    # Add a batch dimension (since the model expects a batch of images)
    img = img.unsqueeze(0)
    
    return img

def predict_image(image_tensor):
    # Disable gradient calculation
    with torch.no_grad():
        # Pass the image through the network
        output = net(image_tensor).to(device)
        
        # Get the predicted class
        _, predicted = torch.max(output.data, 1)
        
    return predicted.item()


# Example usage
image_path = 'path_to_image.png'  # Replace with your image path
image_tensor = preprocess_image(image_path)
prediction = predict_image(image_tensor)

print(f'Predicted digit: {prediction}')