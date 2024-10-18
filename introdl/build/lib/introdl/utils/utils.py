import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import numpy as np
import pandas as pd
import inspect

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
    checkpoint_dict = torch.load(ckpt_file, map_location=device) # moves tensors onto 'device'
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
        
    checkpoint_dict = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint_dict['model_state_dict']) 
    return model.to(device)
