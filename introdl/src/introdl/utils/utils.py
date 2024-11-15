import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms.v2 as transforms
import sys
import os
import random
import numpy as np
import pandas as pd
import inspect
from torchinfo import summary
import traceback

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
        
    checkpoint_dict = torch.load(checkpoint_file, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint_dict['model_state_dict']) 
    return model.to(device)

'''
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
'''

def summarizer(model, input_size, device=torch.device('cpu'), col_width=20, verbose=False):
    """
    Summarizes the given model by displaying the input size, output size, and number of parameters.

    Parameters:
    - model: The model to summarize.
    - input_size (tuple): The input size of the model.
    - device (torch.device, optional): The device to summarize the model on. Defaults to 'cpu'.
    - col_width (int, optional): The width of each column in the summary table. Defaults to 20.
    - verbose (bool, optional): If True, display the full error stack trace; otherwise, show only a simplified error message. Defaults to False.
    """
    model = model.to(device)
    try:
        print(summary(model, input_size=input_size, col_width=col_width, col_names=["input_size", "output_size", "num_params"]))
    except RuntimeError as e:
        if verbose:
            # Print the full stack trace and original error message
            traceback.print_exc()
            print(f"Original Error: {e}")
        else:
            # Display simplified error message with additional message for verbose option
            error_message = str(e).splitlines()[-1].replace("See above stack traces for more details.", "").strip()
            error_message = error_message.replace("Failed to run torchinfo.", "Failed to run all model layers.")
            error_message += " Run again with verbose=True to see stack trace."
            print(f"Error: {error_message}")


'''
def create_CIFAR10_loaders(transform_train=None, transform_test=None, transform_valid=None,
                           valid_prop=0.2, batch_size=64, seed=42, data_dir='./data', 
                           downsample_prop=1.0, num_workers=1, use_augmentation=False):
    """
    Create data loaders for the CIFAR10 dataset.

    Args:
        transform_train (torchvision.transforms.v2.Compose, optional): Transformations for the training set. Defaults to standard training transforms if None.
        transform_test (torchvision.transforms.v2.Compose, optional): Transformations for the test set. Defaults to standard test transforms if None.
        transform_valid (torchvision.transforms.v2.Compose, optional): Transformations for the validation set. Defaults to None.
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
    print("inside create_CIFAR10_loaders")
    print(f"{use_augmentation=}")

    # Set default transforms if none are supplied
    mean = (0.4914, 0.4822, 0.4465) 
    std = (0.2023, 0.1994, 0.2010)

    if transform_train is None:
        if use_augmentation:
            print('choosing augmentation')
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(mean=mean, std=std), 
                transforms.ToPureTensor()   
            ])
        else:
            print('choosing standard')
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=mean, std=std),
                transforms.ToPureTensor()   
            ])
    
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
            transforms.ToPureTensor()   
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
    print('after instantiation')
    print(f"{train_dataset_full.transform=}")

    # Downsample datasets if required
    if downsample_prop < 1.0:
        print('downsampling')
        downsample_size_train = int(downsample_prop * len(train_dataset_full))
        train_dataset_full, _ = random_split(train_dataset_full, [downsample_size_train, len(train_dataset_full) - downsample_size_train], generator=torch.Generator().manual_seed(seed))
        
        downsample_size_test = int(downsample_prop * len(test_dataset))
        test_dataset, _ = random_split(test_dataset, [downsample_size_test, len(test_dataset) - downsample_size_test], generator=torch.Generator().manual_seed(seed))

    # Split the dataset into training and validation sets if valid_prop is provided
    if valid_prop and valid_prop > 0.0:
        train_size = int((1 - valid_prop) * len(train_dataset_full))
        valid_size = len(train_dataset_full) - train_size
        train_dataset, valid_dataset = random_split(train_dataset_full, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))
        print('after split')
        print(f"{train_dataset.dataset.transform=}")
        print(f"{valid_dataset.dataset.transform=}")

        # Apply validation transform to validation dataset
        train_dataset.dataset.transform = transform_train
        valid_dataset.dataset.transform = transform_valid

        # Create data loaders for training, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(train_loader.dataset.dataset.transform)
        print(valid_loader.dataset.dataset.transform)

        return train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset
    else:
        # If no validation set is needed
        train_loader = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader
'''

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def create_CIFAR10_loaders(transform_train=None, transform_test=None, transform_valid=None,
                           valid_prop=0.2, batch_size=64, seed=42, data_dir='./data', 
                           downsample_prop=1.0, num_workers=1, persistent_workers = True, 
                           use_augmentation=False):
    """
    Create data loaders for the CIFAR10 dataset.

    Args:
        transform_train (torchvision.transforms.v2.Compose, optional): Transformations for the training set. Defaults to standard training transforms if None.
        transform_test (torchvision.transforms.v2.Compose, optional): Transformations for the test set. Defaults to standard test transforms if None.
        transform_valid (torchvision.transforms.v2.Compose, optional): Transformations for the validation set. Defaults to None.
        valid_prop (float or None): Proportion of the training set to use for validation. If 0.0 or None, no validation split is made.
        batch_size (int): Batch size for the data loaders.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory to download/load CIFAR10 data.
        downsample_prop (float): Proportion of the dataset to keep if less than 1. Defaults to 1.0.
        num_workers (int): Number of worker processes to use for data loading.
        use_augmentation (bool): Whether to apply data augmentation to the training set. Defaults to False.

    Returns:
        tuple: Train loader, test loader, and optionally valid loader, along with the datasets.
    """

    # Set default transforms if none are supplied
    mean = (0.4914, 0.4822, 0.4465) 
    std = (0.2023, 0.1994, 0.2010)

    if transform_train is None:
        if use_augmentation:
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(mean=mean, std=std), 
                transforms.ToPureTensor()   
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=mean, std=std),
                transforms.ToPureTensor()   
            ])
    
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
            transforms.ToPureTensor()   
        ])
        
    # Set validation transform; if None, use transform_test
    if transform_valid is None:
        transform_valid = transform_test

    # Load the full training and test datasets
    train_dataset_full = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Generate indices for training and validation if needed
    train_indices, valid_indices = None, None
    if valid_prop and 0 < valid_prop < 1.0:
        total_indices = list(range(len(train_dataset_full)))
        train_indices, valid_indices = train_test_split(
            total_indices,
            test_size=valid_prop,
            random_state=seed,
            shuffle=True
        )

    # Downsample datasets if required
    if downsample_prop < 1.0:
        train_indices = train_indices[:int(downsample_prop * len(train_indices))] if train_indices else None
        valid_indices = valid_indices[:int(downsample_prop * len(valid_indices))] if valid_indices else None

    # Create Subset datasets for training and optionally validation
    train_dataset = Subset(train_dataset_full, train_indices) if train_indices else train_dataset_full
    valid_dataset = Subset(CIFAR10(root=data_dir, train=True, download=True, transform=transform_valid), valid_indices) if valid_indices else None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, persistent_workers=persistent_workers)
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                  num_workers=num_workers, persistent_workers=persistent_workers)

    if valid_loader:
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader
