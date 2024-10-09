import torch 
import torch.nn as nn


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_chars=7, num_classes=62):
        super(CNN, self).__init__()
        self.num_chars = num_chars
        self.num_classes = num_classes
        
        # First convolutional layer: 1 input channel, 16 output channels, 5x5 kernel, stride 1, padding 2
        self.conv1 = nn.Sequential(         
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),                              
            nn.ReLU(),                      # Activation function
            nn.MaxPool2d(kernel_size=2),    # Max pooling with 2x2 kernel
        )
        
        # Second convolutional layer: 16 input channels, 32 output channels, 5x5 kernel, stride 1, padding 2
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),     
            nn.ReLU(),                      # Activation function
            nn.MaxPool2d(2),                # Max pooling with 2x2 kernel
        )
        
        # Calculate the size of the flattened feature map
        self.flattened_size = 32 * 7 * (7 * self.num_chars)
        
        # Fully connected layer: input size is the flattened feature map, output size is num_chars * num_classes
        self.out = nn.Linear(self.flattened_size, self.num_chars * self.num_classes)

    def forward(self, x):
        x = self.conv1(x)                   # Pass input through the first convolutional layer
        x = self.conv2(x)                   # Pass input through the second convolutional layer
        x = x.view(x.size(0), -1)           # Flatten the feature map
        output = self.out(x)                # Pass the flattened feature map through the fully connected layer
        output = output.view(-1, self.num_chars, self.num_classes)  # Reshape the output
        return output