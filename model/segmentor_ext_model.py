import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Model
class ModelExtension(nn.Module):
    def __init__(self, nc=1):
        # nc is the number of input channels
        super(ModelExtension, self).__init__()
        self.main = nn.Sequential(
        # Normalize Input.
        nn.BatchNorm2d(nc),
        nn.LeakyReLU(0.2, inplace=True),

        # State: 1 x 38 x 512
        nn.Conv2d(nc, 8, (7, 8), (1,4), (0,2), bias=True),
        nn.BatchNorm2d(8),
        nn.LeakyReLU(0.2, inplace=True),
        
        # State: 8 x 32 x 128
        nn.Conv2d(8, 16, (4, 8), (2,4), (1,2), bias=True),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.2, inplace=True),
        
        # State: 16 x 16 x 32
        nn.Conv2d(16, 32, (4, 8), (2,4), (1,2), bias=True),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        
        # State: 32 x 8 x 8
        nn.Conv2d(32, 64, (4,4), (2,2), (1,1), bias=True),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        
        # State: 64 x 4 x 4
        nn.Conv2d(64, 128, (4,4), (1,1), (0,0), bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        )
       
        # State: 128 x 1 x 1
        self.fc1 = nn.Linear(128, 1, bias=True)
        self.sig = nn.Sigmoid()        

    # Forward Pass.
    def forward(self, input):
        output = self.main(input.permute(0,2,1,3)) # 38 x 1 x 512 output size.
        output = output.view(-1, 128*1*1)
        output = 100.0*self.sig(self.fc1(output))
        return output.unsqueeze(1)
