import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_chars=7, num_classes=62, input_size=(1, 28, 28 * 7)):
        super(CNN, self).__init__()
        self.num_chars = num_chars
        self.num_classes = num_classes

        print(f"Initializing CNN with input_size: {input_size}")

        # Simplified convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        

        # Dynamically compute the flattened size
        self.flattened_size = self._get_flattened_size(input_size)

        # Fully connected layer
        self.out = nn.Linear(self.flattened_size, self.num_chars * self.num_classes)

    def _get_flattened_size(self, input_size):
        # Create a dummy input tensor with the specified input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, x):
        x = self.conv1(x)                   # Pass input through the first convolutional layer
        x = self.conv2(x)                   # Pass input through the second convolutional layer
        x = x.view(x.size(0), -1)           # Flatten the feature map
        output = self.out(x)                # Pass through the fully connected layer
        output = output.view(-1, self.num_chars, self.num_classes)  # Reshape the output
        return output

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # Using Kaiming He initialization for ReLU activations
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
