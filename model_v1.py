import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Định nghĩa mô hình 1D CNN
class CNN1D(nn.Module):
    def __init__(self, block_size=16, num_classes=101):
        super(CNN1D, self).__init__()
        
        # Lớp CNN
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Tính chiều dài đầu ra
        conv_output_length = block_size  # Với padding=1, chiều dài giữ nguyên

        # Lớp Fully Connected chung
        self.fc1 = nn.Linear(64 * conv_output_length, 512)  # 64 kênh * block_size
        self.fc2 = nn.Linear(512, 64)

        # Hai nhánh đầu ra cho i_x và i_y
        self.fc3_x = nn.Linear(64, num_classes)  # Đầu ra cho i_x (101 lớp)
        self.fc3_y = nn.Linear(64, num_classes)  # Đầu ra cho i_y (101 lớp)

    def forward(self, x):
        # Input: (batch_size, 8, block_size)
        x = F.relu(self.conv1(x))  # (batch_size, 16, block_size)
        x = F.relu(self.conv2(x))  # (batch_size, 32, block_size)
        x = F.relu(self.conv3(x))  # (batch_size, 64, block_size)
        
        # Làm phẳng
        x = x.view(x.size(0), -1)  # (batch_size, 64*block_size)
        
        # Qua các lớp FC chung
        x = F.relu(self.fc1(x))    # (batch_size, 512)
        x = F.relu(self.fc2(x))    # (batch_size, 64)
        
        # Hai nhánh đầu ra
        i_x = self.fc3_x(x)        # (batch_size, num_classes)
        i_y = self.fc3_y(x)        # (batch_size, num_classes)
        
        return i_x, i_y
    
    