import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Define a torchvision transform pipeline once
transform = T.Compose(
    [
        T.ToPILImage(),  # Convert NumPy array to PIL image
        T.Resize((84, 84)),  # Resize to 84x84
        T.Grayscale(),  # Convert to grayscale
        T.ToTensor(),  # Convert to tensor
    ]
)


def preprocess(state: np.ndarray) -> np.ndarray:
    """Convert RGB frame to 84x84 grayscale tensor."""
    tensor = transform(state)
    return tensor.numpy()[0]  # Convert back to NumPy array if needed
