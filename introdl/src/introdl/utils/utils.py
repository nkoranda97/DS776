import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import sys
import os
import random
import numpy as np
import pandas as pd
import inspect
from torchinfo import summary

###########################################################
# Utility Functions
###########################################################

def detect_jupyter_environment():
    if 'google.colab' in sys.modules:
        return "colab"
    elif 'VSCODE_PID' in os.environ:
        return "vscode"
    elif 'SMC' in os.environ:
        return "cocalc"
    elif 'JPY_PARENT_PID' in os.environ:
        return "jupyterlab"
    else:
        return "Unknown"
    
def get_device():
    """
    Returns the appropriate device ('cuda', 'mps', or 'cpu') depending on availability.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def load_results(ckpt_file, device=torch.device('cpu')):
    """
    Load the results from a checkpoint file.

    Parameters:
    - ckpt_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the checkpoint onto. Defaults to 'cpu'.

    Returns:
    - results: The loaded results from the checkpoint file.
    """
    checkpoint_dict = torch.load(ckpt_file, map_location=device, weights_only=False) # moves tensors onto 'device'
    return pd.DataFrame(checkpoint_dict['results'])

def load_model(model, checkpoint_file, device=torch.device('cpu')):
    """
    Load the model from a checkpoint file.

    Parameters:
    - model: The model to load. It can be either a class or an instance of the model.
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the model onto. Defaults to 'cpu'.

    Returns:
    - model: The loaded model from the checkpoint file.
    """
    if inspect.isclass(model):
        model = model()
    elif not isinstance(model, nn.Module):
        raise ValueError("The model must be a class or an instance of nn.Module.")
        
    checkpoint_dict = torch.load(checkpoint_file, weights_only=False)
    model.load_state_dict(checkpoint_dict['model_state_dict']) 
    return model.to(device)

def summarizer(model, input_size, device=torch.device('cpu'), col_width=20):
    """
    Summarizes the given model by displaying the input size, output size, and number of parameters.

    Parameters:
    - model: The model to summarize.
    - input_size (tuple): The input size of the model.
    - device (torch.device, optional): The device to summarize the model on. Defaults to 'cpu'.
    - col_width (int, optional): The width of each column in the summary table. Defaults to 20.
    """
    model = model.to(device)
    print(summary(model, input_size=input_size, col_width=col_width, col_names=["input_size", "output_size", "num_params"]))


def create_CIFAR10_loaders(transform_train=None, transform_test=None, transform_valid=None,
                           valid_prop=0.2, batch_size=64, seed=42, data_dir='./data', 
                           downsample_prop=1.0, num_workers=1, use_augmentation=False):
    """
    Create data loaders for the CIFAR10 dataset.

    Args:
        transform_train (torchvision.transforms.Compose, optional): Transformations for the training set. Defaults to standard training transforms if None.
        transform_test (torchvision.transforms.Compose, optional): Transformations for the test set. Defaults to standard test transforms if None.
        transform_valid (torchvision.transforms.Compose, optional): Transformations for the validation set. Defaults to None.
        valid_prop (float or None): Proportion of the training set to use for validation. If 0.0 or None, no validation split is made.
        batch_size (int): Batch size for the data loaders.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory to download/load CIFAR10 data.
        downsample_prop (float): Proportion of the dataset to keep if less than 1. Defaults to 1.0.
        num_workers (int): Number of worker processes to use for data loading.
        use_augmentation (bool): Whether to apply data augmentation to the training set. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: Data loaders for training and test datasets, and validation if valid_prop is set.
    """

    # Set default transforms if none are supplied
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    if transform_train is None:
        if use_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)    
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    # Set validation transform; if None, use transform_test
    if transform_valid is None:
        transform_valid = transform_test

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load the full training and test datasets
    train_dataset_full = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Downsample datasets if required
    if downsample_prop < 1.0:
        downsample_size_train = int(downsample_prop * len(train_dataset_full))
        train_dataset_full, _ = random_split(train_dataset_full, [downsample_size_train, len(train_dataset_full) - downsample_size_train], generator=torch.Generator().manual_seed(seed))
        
        downsample_size_test = int(downsample_prop * len(test_dataset))
        test_dataset, _ = random_split(test_dataset, [downsample_size_test, len(test_dataset) - downsample_size_test], generator=torch.Generator().manual_seed(seed))

    # Split the dataset into training and validation sets if valid_prop is provided
    if valid_prop and valid_prop > 0.0:
        train_size = int((1 - valid_prop) * len(train_dataset_full))
        valid_size = len(train_dataset_full) - train_size
        train_dataset, valid_dataset = random_split(train_dataset_full, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))

        # Apply validation transform to validation dataset
        valid_dataset.dataset.transform = transform_valid

        # Create data loaders for training, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, valid_loader, test_loader
    else:
        # If no validation set is needed
        train_loader = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader
