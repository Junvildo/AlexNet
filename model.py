from torch import nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass

def conv2d(ch_in, ch_out, kz, s=1, p=0):
    return spectral_norm(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kz, stride=s, padding=p))

class AlexNet(nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()

        self.block = seq(
            conv2d(ch_in=3, ch_out=96, kz=11, s=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv2d(ch_in=96, ch_out=256, kz=5, s=1, p=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv2d(ch_in=256, ch_out=384, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=384, ch_out=384, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=384, ch_out=256, kz=3, s=1, p=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=1000), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=num_class), nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)