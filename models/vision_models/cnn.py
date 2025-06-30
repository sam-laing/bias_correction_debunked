import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, input_size=(32, 32), channel_size=3):
        """   
        input is svhn (32x32x3) or cifar10/100 (32x32x3)

        default input size as argument

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 10.
        """
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.channel_size = channel_size

        # Define the CNN layers
        self.conv1 = nn.Conv2d(channel_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * (input_size[0] // 8) * (input_size[1] // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channel_size, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def make_cnn(num_classes=10):
    """
    Factory function to create a CNN model.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.

    Returns:
        CNN: An instance of the CNN model.
    """
    return CNN(num_classes=num_classes)

