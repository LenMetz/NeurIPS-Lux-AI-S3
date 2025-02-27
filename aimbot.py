import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class ActionOracle(torch.nn.Module):
    def __init__(self,n_maps, n_params):
        super().__init__()
        self.n_maps = n_maps
        self.n_params = n_params
        self.cnn = nn.Sequential(
                nn.Conv2d(self.n_maps, 16, kernel_size=1, padding=1),
                nn.ReLU(),
                #nn.MaxPool2d(2),
                nn.Conv2d(16, 16, kernel_size=5, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                #nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(24*24*4, 128),
                nn.ReLU(),
            )
        self.ff = nn.Sequential(
                nn.Linear(128+self.n_params,64),
                nn.ReLU(),
                nn.Linear(64, 5),
                nn.ReLU(),
                nn.Softmax(),
        )
        self.cnn.apply(weights_init)
        self.ff.apply(weights_init)
        
    def forward(self, maps, params):
        cnn_out = self.cnn(maps)
        return self.ff(torch.cat((cnn_out, params),dim=-1))