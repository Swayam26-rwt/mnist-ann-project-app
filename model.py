import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input: 28x28 = 784
        # Hidden 1: 64
        # Hidden 2: 64
        # Hidden 3: 32
        # Output: 10
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def forward_features(self, x):
        """
        Returns the intermediate layer activations as well as the final output.
        Useful for layer-wise feature visualization.
        """
        x = x.view(-1, 28 * 28)
        features = {}
        
        # Layer 1
        x1 = self.fc1(x)
        a1 = F.relu(x1)
        features['layer1'] = a1
        
        # Layer 2
        x2 = self.fc2(a1)
        a2 = F.relu(x2)
        features['layer2'] = a2
        
        # Layer 3
        x3 = self.fc3(a2)
        a3 = F.relu(x3)
        features['layer3'] = a3
        
        # Output
        out = self.fc4(a3)
        return out, features
