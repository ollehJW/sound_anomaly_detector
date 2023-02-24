import torch
from torch import nn
from torch.nn import functional as F

class Baseline(nn.Module):
    def __init__(self, input_dims):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.bn1 = nn.BatchNorm1d(128, affine=True)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128, affine=True)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128, affine=True)
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128, affine=True)
        self.fc5 = nn.Linear(128, 8)
        self.bn5 = nn.BatchNorm1d(8, affine=True)
        self.fc6 = nn.Linear(8, 128)
        self.bn6 = nn.BatchNorm1d(128, affine=True)
        self.fc7 = nn.Linear(128, 128)
        self.bn7 = nn.BatchNorm1d(128, affine=True)
        self.fc8 = nn.Linear(128, 128)
        self.bn8 = nn.BatchNorm1d(128, affine=True)
        self.fc9 = nn.Linear(128, 128)
        self.bn9 = nn.BatchNorm1d(128, affine=True)
        self.fc10 = nn.Linear(128, input_dims)
        

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        out = F.relu(self.bn4(self.fc4(out)))
        out = F.relu(self.bn5(self.fc5(out)))
        out = F.relu(self.bn6(self.fc6(out)))
        out = F.relu(self.bn7(self.fc7(out)))
        out = F.relu(self.bn8(self.fc8(out)))
        out = F.relu(self.bn9(self.fc9(out)))
        out = self.fc10(out)
        return out