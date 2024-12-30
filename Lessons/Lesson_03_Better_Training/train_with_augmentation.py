import os
from pathlib import Path

MODELS_PATH = Path(os.getenv('MODELS_PATH'))
DATA_PATH = Path(os.getenv('DATA_PATH'))
TORCH_HOME = Path(os.getenv('TORCH_HOME'))

print(f'{MODELS_PATH=}')
print(f'{DATA_PATH=}')
print(f'{TORCH_HOME=}')

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from sklearn.metrics import accuracy_score

from introdl.utils import get_device, create_CIFAR10_loaders
from introdl.idlmam import train_network

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    train_loader, valid_loader, test_loader = create_CIFAR10_loaders(data_dir=DATA_PATH, use_augmentation=True, num_workers=4)

    model = SimpleCNN()  # create a new instance of the model
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())  # default lr=0.001, weight_decay=0.01

    device = get_device()
    print(f'Device: {device}')

    ckpt_file = MODELS_PATH / 'L03_CIFAR10_SimpleCNN_AdamW_augment2.pt'
    epochs = 30

    score_funcs = {'ACC': accuracy_score}

    train_network(model,
        loss_func,
        train_loader,
        device=device,
        val_loader=valid_loader,
        epochs=epochs,
        optimizer=optimizer,
        score_funcs=score_funcs,
        checkpoint_file=ckpt_file)
