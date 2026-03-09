import torch
import torch.nn as nn

#  __   __                  ____          _         _   _
#  \ \ / /__  _   _ _ __   / ___|___   __| | ___   | | | | ___ _ __ ___
#   \ V / _ \| | | | '__| | |   / _ \ / _` |/ _ \  | |_| |/ _ \ '__/ _ \
#    | | (_) | |_| | |    | |__| (_) | (_| |  __/  |  _  |  __/ | |  __/
#    |_|\___/ \__,_|_|     \____\___/ \__,_|\___|  |_| |_|\___|_|  \___|
class SmallCNN(nn.Module):
    """
    Simple CNN classifier. Design your own architecture.
    """
    def __init__(self, w=28, h=28, num_classes=10): # ⚠️ DO NOT change this line
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, num_classes),
        )





    """
    Forward signature MUST be forward(x):
        x: (B, 28, 28)
    """
    def forward(self, x): # ⚠️ DO NOT change this line
        x = x.unsqueeze(1)

        x = self.conv_layer(x)
        logits = self.fc_layers(x)

        return logits
