import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, SelectMultiple, Dropdown, Button, HBox, Text, VBox, Output
from IPython.display import display, clear_output
import torchvision.utils as vutils
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.pyplot import get_cmap
import matplotlib.gridspec as gridspec
from scipy.special import erf


########################################################
# Visualization Related Functions
########################################################

def show_image_grid(nrows, ncols, dataset, mean=None, std=None, cmap='Greys'):
    """
    Display a grid of images from a PyTorch dataset.

    Args:
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
        dataset (Dataset): PyTorch dataset containing images.
        mean (list or tuple, optional): Mean for denormalization.
        std (list or tuple, optional): Standard deviation for denormalization.
        cmap (str, optional): Colormap for 1-channel images. Default is 'Greys'.
    """
    # Randomly select nrows * ncols images
    indices = torch.randint(0, len(dataset), (nrows * ncols,))
    images = torch.stack([dataset[i][0] for i in indices])

    # Denormalize images if mean and std are provided
    if mean is not None and std is not None:
        images = torch.stack([denormalize_image(img, mean, std) for img in images])

    # Make a grid of images
    grid = vutils.make_grid(images, nrow=ncols, padding=2)

    # Check if images are 1-channel and extract the first channel only
    if grid.shape[0] == 1 or grid.shape[0] == 3 and torch.all(grid[0] == grid[1]) and torch.all(grid[0] == grid[2]):
        # If grid has 3 channels but all are the same, convert to 1-channel
        grid = grid[0].unsqueeze(0)

    # Convert the tensor to a numpy array for display
    npimg = grid.numpy()

    # Display the grid of images
    plt.figure(figsize=(ncols, nrows))
    # Check if the image is 1-channel or 3-channel
    if npimg.shape[0] == 1:  # 1-channel (grayscale)
        plt.imshow(npimg[0], cmap=cmap)
    else:  # 3-channel (RGB or other multi-channel)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def visualize2DSoftmax(X, y, model, title=None):
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20), np.linspace(y_min, y_max, num=20), indexing='ij')
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    with torch.no_grad():
        logits = model(torch.tensor(xy_v, dtype=torch.float32))
        y_hat = F.softmax(logits, dim=1).numpy()

    plt.figure(figsize=(5,5))
    cs = plt.contourf(xv, yv, y_hat[:,0].reshape(20,20), levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu)
    ax = plt.gca()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)
    if title is not None:
        ax.set_title(title)

# Helper function to register hooks and collect feature maps
def _get_conv_feature_maps(model, input_image):
    """
    Registers hooks to all convolutional layers in the model and captures the output feature maps.

    Args:
        model (torch.nn.Module): The trained CNN model.
        input_image (torch.Tensor): The input image tensor.

    Returns:
        List[torch.Tensor]: A list of feature maps captured from each convolutional layer.
    """
    conv_outputs = []
    
    def hook_fn(module, input, output):
        conv_outputs.append(output)
    
    hooks = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(input_image)
    
    for hook in hooks:
        hook.remove()

    return conv_outputs

# Function to apply colormap to individual feature maps with absolute scaling
def _apply_colormap_to_feature_map(feature_map, cmap_name='PuOr', vmin=-1, vmax=1):
    """
    Apply colormap (e.g., PuOr) to a single-channel feature map and convert it to a 3-channel RGB image.
    
    Args:
        feature_map (torch.Tensor): The feature map with shape (H, W).
        cmap_name (str): The name of the colormap to use.
        vmin (float): Minimum value for colormap normalization.
        vmax (float): Maximum value for colormap normalization.

    Returns:
        torch.Tensor: The colored feature map in RGB format.
    """
    feature_map_np = feature_map.cpu().numpy()
    cmap = get_cmap(cmap_name)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # Centered colormap
    colored_map = cmap(norm(feature_map_np))[:, :, :3]  # Apply colormap and remove alpha channel
    colored_map_rgb_tensor = torch.tensor(colored_map).permute(2, 0, 1)  # Convert to (3, H, W)
    return colored_map_rgb_tensor

# Function to calculate optimal grid size based on image size, padding, and aspect ratio
def _calculate_grid_dimensions(num_feature_maps, aspect_ratio, H, W, padding):
    """
    Calculate the optimal number of columns and rows for the feature map grid to achieve the desired aspect ratio.
    
    Args:
        num_feature_maps (int): The number of feature maps to display.
        aspect_ratio (float): The desired aspect ratio.
        H (int): The height of the feature maps.
        W (int): The width of the feature maps.
        padding (int): The padding between the feature maps.
        
    Returns:
        (int, int): The number of columns and rows for the grid.
    """
    total_width = W + 2 * padding
    total_height = H + 2 * padding
    adjusted_aspect_ratio = aspect_ratio * (total_height / total_width)
    num_columns = int(np.ceil(np.sqrt(num_feature_maps * adjusted_aspect_ratio)))
    num_rows = int(np.ceil(num_feature_maps / num_columns))
    return num_columns, num_rows

# Function to denormalize the image using specified mean and std
def denormalize_image(image_tensor, mean, std):
    """
    Denormalizes an image tensor using the specified mean and std.

    Args:
        image_tensor (torch.Tensor): The normalized image tensor.
        mean (tuple): The mean values for each channel.
        std (tuple): The standard deviation values for each channel.
    
    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image_tensor * std + mean

# Function to visualize feature maps with the title at the top and the original image over the feature maps
def visualize_feature_maps(dataset, model, target_index, mean, std, activation=None, pooling=None, layer=1, padding=1, scale=1.0, cmap_name='PuOr'):
    """
    Visualizes the feature maps from a CNN model for a target image in the dataset as a single image with colored borders between maps using a colormap.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset containing the images.
        model (torch.nn.Module): The trained CNN model.
        target_index (int): The index of the target image in the dataset.
        mean (tuple): The mean values for each channel.
        std (tuple): The standard deviation values for each channel.
        activation (str, optional): The activation function to apply to the feature maps. Options: None (default), 'relu', 'tanh'.
        pooling (str, optional): Whether to apply max pooling to the feature maps. Options: None (default), 'average', 'max'.
        layer (int, optional): The index of the convolutional layer to visualize (1-based indexing). Defaults to 1.
        padding (int, optional): Padding between feature maps in the grid. Defaults to 1 for one-pixel black borders.
        scale (float, optional): Scaling factor for the figure size. Defaults to 1.0.
        cmap_name (str, optional): Colormap name to use for feature maps. Defaults to 'PuOr'.
    """
    model.eval()
    device = next(model.parameters()).device
    image, label = dataset[target_index]
    image = image.unsqueeze(0).to(device)
    
    # Get all feature maps from convolutional layers
    conv_outputs = _get_conv_feature_maps(model, image)
    layer_idx = layer - 1  # Convert to 0-based index
    
    if layer_idx >= len(conv_outputs) or layer_idx < 0:
        raise ValueError(f"Layer index out of range. The model has {len(conv_outputs)} convolutional layers.")
    
    feature_maps = conv_outputs[layer_idx].squeeze(0)
    
    if pooling == 'average':
        pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        feature_maps = pool(feature_maps)
    elif pooling == 'max':
        pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        feature_maps = pool(feature_maps)

    if activation == 'relu':
        feature_maps = torch.relu(feature_maps)
    elif activation == 'tanh':
        feature_maps = torch.tanh(feature_maps)
   
    # Compute the absolute maximum across all feature maps for consistent colormap scaling
    abs_max = torch.max(torch.abs(feature_maps)).item()
    vmin, vmax = -abs_max, abs_max

    # Create the feature maps grid image using updated grid dimensions
    H, W = feature_maps.shape[1:3]  # Use actual height and width of feature maps
    num_columns, num_rows = _calculate_grid_dimensions(num_feature_maps=len(feature_maps), 
                                                      aspect_ratio=16/6.5, H=H, W=W, padding=padding)
    colored_feature_maps = [_apply_colormap_to_feature_map(feature_maps[i], cmap_name=cmap_name, vmin=vmin, vmax=vmax) for i in range(feature_maps.shape[0])]
    colored_feature_maps_tensor = torch.stack(colored_feature_maps)
    grid_image = vutils.make_grid(colored_feature_maps_tensor, nrow=num_columns, padding=padding, normalize=False, pad_value=0)

    figsize = (16 * scale, 9 * scale)

    # Set up a grid layout with the title at the top, then the original image, then feature maps below
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 2, 6.5])  # Title, original image, feature maps

    # Title at the top
    ax_title = fig.add_subplot(gs[0])
    ax_title.text(0.5, 0.5, f'Feature Maps for Layer {layer}', fontsize=18, verticalalignment='center', horizontalalignment='center')
    ax_title.axis('off')

    # Denormalize and display the original image in the middle
    ax_image = fig.add_subplot(gs[1])
    original_image = image.squeeze(0).cpu()
    original_image = denormalize_image(original_image, mean, std)
    original_image_np = np.clip(original_image.permute(1, 2, 0).numpy(), 0, 1)  # Clip values to be in range [0, 1]
    if original_image_np.shape[2] == 1:
        ax_image.imshow(original_image_np.squeeze(), cmap='Grays')
    else:
        ax_image.imshow(original_image_np)
    ax_image.axis('off')

    # Feature maps grid at the bottom
    ax_maps = fig.add_subplot(gs[2])
    grid_image_np = grid_image.permute(1, 2, 0).cpu().numpy()
    ax_maps.imshow(grid_image_np)
    ax_maps.axis('off')

    # Adjust padding to give proper spacing between the original image and feature maps, while removing left and right padding
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

    plt.show()

def feature_maps_widget(model, dataset, initial_target_index=0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), scale=0.35):
    """
    Creates and displays an interactive widget for visualizing feature maps with the ability to adjust
    layer index, activation type, and pooling type.

    Args:
        model (torch.nn.Module): The trained CNN model.
        dataset (torch.utils.data.Dataset): The dataset containing the images.
        initial_target_index (int, optional): Initial index of the image to visualize. Defaults to 0.
        mean (tuple, optional): Mean for image normalization. Defaults to (0.485, 0.456, 0.406).
        std (tuple, optional): Standard deviation for image normalization. Defaults to (0.229, 0.224, 0.225).
        scale (float, optional): Scale factor for the figure size. Defaults to 0.35.
    """
    def _update_visualization(change=None):
        """
        Function to update the visualization based on widget values.
        Called when any widget value changes.
        """
        clear_output(wait=True)
        display(ui_and_visualization)  # Display the UI and visualization container again
        visualize_feature_maps(
            dataset=dataset,
            model=model,
            target_index=target_index_widget.value,
            mean=mean,
            std=std,
            layer=layer_widget.value,
            activation=activation_widget.value,
            pooling=pooling_widget.value,
            scale=scale  # Use the fixed scale value
        )

    # Define a fixed width for the entire UI and visualization
    fixed_width = '500px'  # Adjust this width as necessary to match your desired size
    left_control_width = '300px'  # Width of the control widgets
    right_control_width = '200px'  # Width of the control widgets
    menu_width = '500px'  # Width of the menu widgets

    # Widgets for interactive control
    layer_widget = widgets.IntSlider(
        value=1,  # Default layer index
        min=1,    # Minimum layer index
        max=len([l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]),  # Number of conv layers
        step=1,
        description='Layer Index',
        layout=widgets.Layout(width=left_control_width)  # Set consistent width
    )

    # layer_widget = widgets.Dropdown(
    #     options=[i for i, l in enumerate(model.cnn_layers) if isinstance(l, torch.nn.Conv2d)],
    #     value=1,
    #     description='Layer Index',
    #     layout=widgets.Layout(width=control_width)  # Set consistent width
    # )

    activation_widget = widgets.Dropdown(
        options=[None, 'relu', 'tanh'],
        value=None,
        description='Activation',
        layout=widgets.Layout(width=right_control_width)  # Set consistent width
    )

    pooling_widget = widgets.Dropdown(
        options=[None, 'max', 'average'],
        value=None,
        description='Pooling',
        layout=widgets.Layout(width=right_control_width)  # Set consistent width
    )

    target_index_widget = widgets.IntSlider(
        min=0,
        max=len(dataset) - 1,
        step=1,
        value=initial_target_index,
        description='Image Index',
        layout=widgets.Layout(width=left_control_width)  # Set consistent width
    )

    # Arrange widgets in two columns
    left_column = VBox([layer_widget, target_index_widget], layout=widgets.Layout(width=left_control_width))
    right_column = VBox([activation_widget, pooling_widget], layout=widgets.Layout(width=right_control_width))

    # Combine columns into one horizontal box with fixed width
    ui = HBox([left_column, right_column], layout=widgets.Layout(width=menu_width))

    # Combine UI and visualization into a single vertical box with fixed width
    ui_and_visualization = VBox([ui], layout=widgets.Layout(width=fixed_width))

    # Display the arranged widgets after setting up the layout
    display(ui_and_visualization)

    # Attach update function to widget value changes
    layer_widget.observe(_update_visualization, names='value')
    activation_widget.observe(_update_visualization, names='value')
    pooling_widget.observe(_update_visualization, names='value')
    target_index_widget.observe(_update_visualization, names='value')

    # Initial visualization
    update_visualization()

# Function to retrieve filters for all convolutional layers
def _get_conv_filters(model):
    """
    Retrieves the filters (weights) for each convolutional layer in the model.

    Args:
        model (torch.nn.Module): The trained CNN model.

    Returns:
        List[torch.Tensor]: A list of filter tensors for each convolutional layer.
    """
    conv_filters = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_filters.append(layer.weight.detach().cpu())
    return conv_filters

# Function to normalize the filters for visualization
def _normalize_filters(filters):
    """
    Normalizes the filters to the range [0, 1] for better visualization.
    
    Args:
        filters (torch.Tensor): The filters tensor.

    Returns:
        torch.Tensor: The normalized filters tensor.
    """
    min_val = filters.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    max_val = filters.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    return (filters - min_val) / (max_val - min_val)

# Function to visualize the filters of a specified convolutional layer
def visualize_filters(model, layer_idx=1, padding=1, scale=1.0):
    """
    Visualizes the filters from a specified convolutional layer in the model.

    Args:
        model (torch.nn.Module): The trained CNN model.
        layer_idx (int, optional): The index of the convolutional layer to visualize (1-based indexing). Defaults to 1.
        padding (int, optional): Padding between filters in the grid. Defaults to 1.
        scale (float, optional): Scaling factor for the figure size. Defaults to 1.0.
    """
    conv_filters = _get_conv_filters(model)
    layer_idx = layer_idx - 1  # Convert to 0-based index
    
    if layer_idx >= len(conv_filters) or layer_idx < 0:
        raise ValueError(f"Layer index out of range. The model has {len(conv_filters)} convolutional layers.")
    
    filters = conv_filters[layer_idx]
    num_filters = filters.shape[0]
    in_channels = filters.shape[1]
    kernel_size = filters.shape[2:]

    # Normalize filters for better visualization
    filters = _normalize_filters(filters)

    # Determine the number of rows and columns for the subplot grid
    fig, axs = plt.subplots(num_filters, in_channels, figsize=(in_channels * 2, num_filters * 2))

    for i in range(num_filters):
        for j in range(in_channels):
            # Handle single row or single column cases
            if num_filters == 1 and in_channels == 1:
                ax = axs
            elif num_filters == 1:
                ax = axs[j]
            elif in_channels == 1:
                ax = axs[i]
            else:
                ax = axs[i, j]
            
            ax.imshow(filters[i, j], cmap='gray', aspect='auto')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def training_plots_widget(df=None, checkpoint_file=None, figsize=(10, 6)):
    """
    Visualize training results using an interactive widget.

    This function displays training results based on the input provided. It accepts either a 
    DataFrame containing the results or a path to a PyTorch model checkpoint file. If a model 
    checkpoint file is provided, the function loads the file and extracts the 'results' DataFrame 
    stored within it for visualization.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        A DataFrame containing training results. This DataFrame should have columns such as 
        'epoch', 'train loss', 'val loss', 'train acc', 'val acc', etc. If provided, the widget 
        will use this DataFrame for visualization.

    checkpoint_file : str, optional
        A path to a PyTorch model checkpoint file (.pt) that contains a 'results' DataFrame. 
        If provided, the function loads the file and extracts the DataFrame for visualization. 
        The file should have been saved using the format:
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results
        }, checkpoint_file)

    figsize : tuple, optional, default=(10, 6)
        The size of the figure to be created for the plots. It should be specified as a tuple 
        (width, height).

    Returns
    -------
    None
        The function displays an interactive widget within a Jupyter Notebook for visualizing 
        training results. It provides options to select the x-axis, y-axis metrics, and save 
        the plotted figures.
    
    Examples
    --------
    # Visualize using a DataFrame
    training_plots_widget(df=results_Fashion_MNIST_LeNet5)

    # Visualize using a model checkpoint file
    training_plots_widget(checkpoint_file='./models/latest_model.pt')
    """

    # Output widget to display messages or errors
    output_messages = Output()
    display(output_messages)  # Display output widget at the start to capture messages

    # Function to load checkpoint and extract the results DataFrame
    def load_checkpoint(file_path):
        with output_messages:
            clear_output(wait=True)
            print(f"Loading checkpoint from: {file_path}")
        try:
            checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            results = checkpoint.get('results', None)
            if results is None:
                with output_messages:
                    print(f"No 'results' DataFrame found in checkpoint: {file_path}")
                return None
            with output_messages:
                print(f"Loaded results DataFrame with columns: {results.keys()}")
            return pd.DataFrame(results)
        except Exception as e:
            with output_messages:
                print(f"Error loading checkpoint: {e}")
            return None
    
    # Function to initialize the widget with the provided or loaded DataFrame
    def _initialize_widget(df):
        clear_output(wait=True)  # Clear previous outputs
        
        # Display the loaded DataFrame columns
        with output_messages:
            print(f"DataFrame loaded with columns: {df.columns.tolist()}")
        
        # Dropdown to choose the x-axis
        x_axis_dropdown = Dropdown(options=['epoch', 'total time'], value='epoch', description='X-axis:')
        
        # SelectMultiple to choose the y-axis columns
        y_axis_select = SelectMultiple(options=df.columns, value=['train loss'], description='Y-axis:')
        
        # Button to prompt for saving the figure
        save_button = Button(description="Save Figure", button_style="success")
        
        # Text box to enter filename
        filename_input = Text(description="Filename:", value="training_results.png", placeholder="Enter filename")
        
        # Button to actually save the figure after filename is provided
        confirm_button = Button(description="Confirm Save", button_style="info")
        
        # Hide filename input and confirm button initially
        filename_input.layout.display = 'none'
        confirm_button.layout.display = 'none'
        
        # Function to plot data based on selected x-axis and y-axis using Seaborn
        def _plot_data(x_axis, y_axis):
            plt.figure(figsize=figsize)
            
            # Define linestyles inside the function
            linestyles = ['-', '--', '-.', ':']
            
            # Plot each selected y-axis column
            for idx, col in enumerate(y_axis):
                sns.lineplot(data=df, x=x_axis, y=col, linestyle=linestyles[idx % len(linestyles)], label=col)
            
            plt.xlabel(x_axis)
            plt.ylabel('Values')
            plt.title('Training Results')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # Function to save the figure as a PNG file using Seaborn
        def _save_figure(_):
            # Show filename input and confirm button when save is requested
            filename_input.layout.display = 'block'
            confirm_button.layout.display = 'block'
        
        # Function to actually save the figure with the given filename
        def _confirm_save(_):
            try:
                fig, ax = plt.subplots(figsize=figsize)
                
                # Define linestyles inside the function
                linestyles = ['-', '--', '-.', ':']
                
                # Plot each selected y-axis column with different linestyles
                for idx, col in enumerate(y_axis_select.value):
                    sns.lineplot(data=df, x=x_axis_dropdown.value, y=col, linestyle=linestyles[idx % len(linestyles)], ax=ax, label=col)
                
                ax.set_xlabel(x_axis_dropdown.value)
                ax.set_ylabel('Values')
                ax.set_title('Training Results')
                ax.legend()
                ax.grid(True)
                
                # Save the figure with the user-specified filename
                filename = filename_input.value
                if not filename.endswith(".png"):
                    filename += ".png"
                
                fig.savefig(filename)
                print(f"Figure saved successfully as {filename}.")
                plt.close(fig)
                
                # Hide the filename input and confirm button after saving
                filename_input.layout.display = 'none'
                confirm_button.layout.display = 'none'
            except Exception as e:
                print(f"Error saving figure: {e}")
        
        # Link the save button to the save_figure function
        save_button.on_click(_save_figure)
        
        # Link the confirm button to the confirm_save function
        confirm_button.on_click(_confirm_save)
        
        # Interactive widget to update the plot
        interact(_plot_data, x_axis=x_axis_dropdown, y_axis=y_axis_select)
        
        # Display the interactive widgets and save options
        display(VBox([HBox([save_button]), filename_input, confirm_button]))
    
    # Check the input type
    if df is not None:
        # If a DataFrame is provided, directly initialize the widget
        _initialize_widget(df)
    elif checkpoint_file is not None:
        # If a model file is provided, load it and initialize the widget
        if os.path.exists(checkpoint_file):
            df_loaded = load_checkpoint(checkpoint_file)
            if df_loaded is not None:
                _initialize_widget(df_loaded)
        else:
            with output_messages:
                print(f"Model file not found: {checkpoint_file}")
    else:
        # If neither df nor checkpoint_file is provided, display an error message
        with output_messages:
            print("Please provide either a DataFrame or a path to a checkpoint file.")

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import VBox, HBox, Dropdown, SelectMultiple, Button, Text, RadioButtons, Output, Layout
from IPython.display import display, clear_output

def training_plots_widget2(df=None, checkpoint_file=None, figsize=(5, 4)):
    """
    Visualize training results using an interactive widget with two plots and options to save.
    """
    
    # Output widget to display the plots
    output_plot = Output(layout=Layout(height='400px', width='100%'))  # Taller output for the plot area
    output_messages = Output(layout=Layout(height='50px'))  # Messages area

    # Function to load checkpoint and extract the results DataFrame
    def load_checkpoint(file_path):
        try:
            checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            results = checkpoint.get('results', None)
            if results is None:
                return None
            return pd.DataFrame(results)
        except Exception as e:
            return None
    
    # Function to initialize the widget with the provided or loaded DataFrame
    def _initialize_widget(df):
        clear_output(wait=True)  # Clear previous outputs
        
        col_width = '200px'

        # Dropdowns for left plot
        x_axis_left_dropdown = Dropdown(
            options=['epoch', 'total time'] if 'epoch' in df.columns else df.columns,
            value='epoch' if 'epoch' in df.columns else df.columns[0],
            description='X-axis (Left):',
            layout=Layout(width=col_width)
        )
        
        y_axis_left_select = SelectMultiple(
            options=df.columns,
            value=['train loss'] if 'train loss' in df.columns else [df.columns[0]],
            description='Y-axis (Left):',
            layout=Layout(width=col_width, height='150px')  # Increased height to ensure visibility
        )
        
        # Dropdowns for right plot
        x_axis_right_dropdown = Dropdown(
            options=['epoch', 'total time'] if 'epoch' in df.columns else df.columns,
            value='epoch' if 'epoch' in df.columns else df.columns[0],
            description='X-axis (Right):',
            layout=Layout(width=col_width)
        )
        
        y_axis_right_select = SelectMultiple(
            options=df.columns,
            value=['test loss'] if 'test loss' in df.columns else [df.columns[0]],
            description='Y-axis (Right):',
            layout=Layout(width=col_width, height='150px')  # Increased height for better visibility
        )
        
        # Radio buttons for selecting which plot(s) to save
        save_options = RadioButtons(
            options=['Left Plot', 'Right Plot', 'Both'],
            description='Save:',
            disabled=False,
            layout=Layout(width=col_width)
        )
        
        # Text box to enter filename
        filename_input = Text(description="Filename:", value="training_results.png", layout=Layout(width='300px'))
        
        # Save button
        save_button = Button(description="Save Figure", button_style="success", layout=Layout(width=col_width))

        # Function to plot data based on selected x-axis and y-axis using Seaborn
        def _plot_data():
            with output_plot:
                clear_output(wait=True)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
                
                # Define linestyles inside the function
                linestyles = ['-', '--', '-.', ':']
                
                # Plot left side
                for idx, col in enumerate(y_axis_left_select.value):
                    sns.lineplot(data=df, x=x_axis_left_dropdown.value, y=col, linestyle=linestyles[idx % len(linestyles)], ax=ax1, label=col)
                ax1.set_xlabel(x_axis_left_dropdown.value)
                ax1.set_ylabel('Values')
                ax1.set_title('Left Plot')
                ax1.legend()
                ax1.grid(True)
                
                # Plot right side
                for idx, col in enumerate(y_axis_right_select.value):
                    sns.lineplot(data=df, x=x_axis_right_dropdown.value, y=col, linestyle=linestyles[idx % len(linestyles)], ax=ax2, label=col)
                ax2.set_xlabel(x_axis_right_dropdown.value)
                ax2.set_ylabel('Values')
                ax2.set_title('Right Plot')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()  # Tighten layout for the widget plot
                plt.show()

        # Function to save the figure with the selected options
        def _save_figure(_):
            with output_messages:
                try:
                    # Define linestyles here for saving the figure
                    linestyles = ['-', '--', '-.', ':']
                    
                    save_choice = save_options.value
                    filename = filename_input.value
                    if not filename.endswith(".png"):
                        filename += ".png"
                    
                    if save_choice == 'Left Plot':
                        fig, ax = plt.subplots(figsize=figsize)
                        for idx, col in enumerate(y_axis_left_select.value):
                            sns.lineplot(data=df, x=x_axis_left_dropdown.value, y=col, linestyle=linestyles[idx % len(linestyles)], ax=ax, label=col)
                        ax.set_xlabel(x_axis_left_dropdown.value)
                        ax.set_ylabel('Values')
                        ax.set_title('Left Plot')
                        ax.legend()
                        ax.grid(True)
                        plt.tight_layout()  # Tighten layout before saving
                        fig.savefig(filename, bbox_inches='tight')  # Remove extra white space
                        print(f"Left plot saved as {filename}")
                        plt.close(fig)
                        
                    elif save_choice == 'Right Plot':
                        fig, ax = plt.subplots(figsize=figsize)
                        for idx, col in enumerate(y_axis_right_select.value):
                            sns.lineplot(data=df, x=x_axis_right_dropdown.value, y=col, linestyle=linestyles[idx % len(linestyles)], ax=ax, label=col)
                        ax.set_xlabel(x_axis_right_dropdown.value)
                        ax.set_ylabel('Values')
                        ax.set_title('Right Plot')
                        ax.legend()
                        ax.grid(True)
                        plt.tight_layout()  # Tighten layout before saving
                        fig.savefig(filename, bbox_inches='tight')  # Remove extra white space
                        print(f"Right plot saved as {filename}")
                        plt.close(fig)
                        
                    else:  # Both plots
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
                        for idx, col in enumerate(y_axis_left_select.value):
                            sns.lineplot(data=df, x=x_axis_left_dropdown.value, y=col, linestyle=linestyles[idx % len(linestyles)], ax=ax1, label=col)
                        ax1.set_xlabel(x_axis_left_dropdown.value)
                        ax1.set_ylabel('Values')
                        ax1.set_title('Left Plot')
                        ax1.legend()
                        ax1.grid(True)

                        for idx, col in enumerate(y_axis_right_select.value):
                            sns.lineplot(data=df, x=x_axis_right_dropdown.value, y=col, linestyle=linestyles[idx % len(linestyles)], ax=ax2, label=col)
                        ax2.set_xlabel(x_axis_right_dropdown.value)
                        ax2.set_ylabel('Values')
                        ax2.set_title('Right Plot')
                        ax2.legend()
                        ax2.grid(True)
                        
                        plt.tight_layout()  # Tighten layout before saving
                        fig.savefig(filename, bbox_inches='tight')  # Remove extra white space
                        print(f"Both plots saved as {filename}")
                        plt.close(fig)

                except Exception as e:
                    print(f"Error saving figure: {e}")
        
        # Link the save button to the save_figure function
        save_button.on_click(_save_figure)

        # Event listeners for dropdowns to trigger re-plot
        x_axis_left_dropdown.observe(lambda change: _plot_data(), 'value')
        y_axis_left_select.observe(lambda change: _plot_data(), 'value')
        x_axis_right_dropdown.observe(lambda change: _plot_data(), 'value')
        y_axis_right_select.observe(lambda change: _plot_data(), 'value')

        # Layout for dropdowns and buttons
        plot_controls = HBox([VBox([x_axis_left_dropdown, y_axis_left_select]), 
                              VBox([x_axis_right_dropdown, y_axis_right_select]), 
                              VBox([save_options, filename_input, save_button])])

        # Display the layout: top for plots, bottom for controls
        display(VBox([output_plot, plot_controls, output_messages]))

        # Initial plot display
        _plot_data()
    
    # Check the input type
    if df is not None:
        _initialize_widget(df)
    elif checkpoint_file is not None:
        if os.path.exists(checkpoint_file):
            df_loaded = load_checkpoint(checkpoint_file)
            if df_loaded is not None:
                _initialize_widget(df_loaded)
        else:
            print(f"Model file not found: {checkpoint_file}")
    else:
        print("Please provide either a DataFrame or a path to a checkpoint file.")


def activations_widget():
    # Define the activation functions
    activation_functions = {
        "sigmoid": lambda x: (1 / (1 + np.exp(-x)), lambda y: y * (1 - y)),
        "ReLU": lambda x: (np.maximum(0, x), lambda y: np.where(x > 0, 1, 0)),
        "tanh": lambda x: (np.tanh(x), lambda y: 1 - y ** 2),
        "LeakyReLU": lambda x: (np.where(x > 0, x, 0.1 * x), lambda y: np.where(x > 0, 1, 0.1 * np.ones_like(x))),
        "GELU": lambda x: (x * 0.5 * (1 + erf(x / np.sqrt(2))), lambda y: 0.5 * (1 + erf(x / np.sqrt(2))) + (x * np.exp(-0.5 * x ** 2)) / np.sqrt(2 * np.pi)),
        "swish": lambda x: (x / (1 + np.exp(-x)), lambda y: (1 + np.exp(-x) + x * np.exp(-x)) / (1 + np.exp(-x)) ** 2),
    }

    # Define the function to plot selected activation functions
    def _plot_activation_functions(selected_activations):
        x = np.linspace(-6, 6, 100)
        data = {
            "x": [],
            "value": [],
            "derivative": [],
            "activation": [],
        }

        for activation in selected_activations:
            func = activation_functions[activation]
            y, dy_func = func(x)
            dy = dy_func(y)

            # Append data for plotting
            data["x"].extend(x)
            data["value"].extend(y)
            data["derivative"].extend(dy)
            data["activation"].extend([activation] * len(x))

        # Create DataFrame from the collected data
        df = pd.DataFrame(data)

        # Set up the plotting style and layout
        sns.set(style="whitegrid", palette="muted")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # Adjusted aspect ratio

        # Plot activation functions
        sns.lineplot(
            data=df, 
            x="x", 
            y="value", 
            hue="activation", 
            style="activation", 
            ax=axes[0]
        )
        axes[0].set_title("Activation Functions")
        axes[0].legend(title='Activation')

        # Plot derivatives of activation functions
        sns.lineplot(
            data=df, 
            x="x", 
            y="derivative", 
            hue="activation", 
            style="activation", 
            ax=axes[1]
        )
        axes[1].set_title("Derivatives")
        axes[1].legend(title="Activation'")

        plt.show()

    # Create a widget for selecting activation functions
    activation_selector = widgets.SelectMultiple(
        options=list(activation_functions.keys()),
        value=["sigmoid"],  # Default value
        description="Activations",
        disabled=False
    )

    # Use the interactive function to connect the selector widget to the plot function
    interactive_plot = interactive(_plot_activation_functions, selected_activations=activation_selector)
    output = interactive_plot.children[-1]
    output.layout.height = '500px'
    display(interactive_plot)


def visualize_feature_maps(dataset, model, target_index, mean, std, layer=1, activation=None, pooling=None, original_image_size=(5, 5), feature_maps_size=(10, 10), cmap_name='PuOr'):
    """
    Visualizes the feature maps from a CNN model for a target image in the dataset using GridSpec to control the layout.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset containing the images.
        model (torch.nn.Module): The trained CNN model.
        target_index (int): The index of the target image in the dataset.
        mean (tuple): The mean values for each channel.
        std (tuple): The standard deviation values for each channel.
        layer (int, optional): The index of the convolutional layer to visualize. Defaults to 1.
        activation (str, optional): Activation function to apply ('relu', 'tanh', or None). Defaults to None.
        pooling (str, optional): Whether to apply max pooling or average pooling. Options: None, 'max', 'average'. Defaults to None.
        original_image_size (tuple, optional): Figure size for the original image. Defaults to (5, 5).
        feature_maps_size (tuple, optional): Figure size for the feature maps. Defaults to (10, 10).
        cmap_name (str, optional): Colormap name to use for feature maps. Defaults to 'PuOr'.
    """
    model.eval()
    device = next(model.parameters()).device
    image, label = dataset[target_index]
    image = image.unsqueeze(0).to(device)

    # Hook function to extract feature maps
    conv_outputs = []
    
    def hook_fn(module, input, output):
        conv_outputs.append(output)

    # Register hooks to all convolutional layers
    conv_layers = [layer_module for layer_module in model.modules() if isinstance(layer_module, torch.nn.Conv2d)]
    if layer > len(conv_layers):
        raise ValueError(f"The model has only {len(conv_layers)} convolutional layers, but layer {layer} was requested.")

    # Attach the hook to the specific layer
    hooks = [conv_layers[layer - 1].register_forward_hook(hook_fn)]

    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

    if not conv_outputs:
        raise RuntimeError(f"No feature maps were captured. Make sure the requested layer {layer} is valid.")

    # Get the feature maps from the specified layer
    feature_maps = conv_outputs[0].squeeze(0)  # Get the feature maps for the layer

    # Apply the requested activation function
    if activation == 'relu':
        feature_maps = torch.relu(feature_maps)
    elif activation == 'tanh':
        feature_maps = torch.tanh(feature_maps)

    # Apply pooling if requested
    if pooling == 'max':
        feature_maps = torch.nn.functional.max_pool2d(feature_maps, kernel_size=2, stride=2)
    elif pooling == 'average':
        feature_maps = torch.nn.functional.avg_pool2d(feature_maps, kernel_size=2, stride=2)

    # Normalize feature maps to [0, 1] range for consistent visualization
    abs_max = torch.max(torch.abs(feature_maps)).item()
    vmin, vmax = -abs_max, abs_max

    # Apply colormap to feature maps
    def apply_colormap(feature_map, cmap_name):
        feature_map_np = feature_map.cpu().numpy()
        cmap = get_cmap(cmap_name)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        colored_map = cmap(norm(feature_map_np))[:, :, :3]  # Apply colormap and remove alpha channel
        return torch.tensor(colored_map).permute(2, 0, 1)  # Convert to (3, H, W)

    colored_feature_maps = [apply_colormap(feature_maps[i], cmap_name) for i in range(feature_maps.shape[0])]

    # Create the grid of feature maps for display
    grid_image = vutils.make_grid(colored_feature_maps, nrow=int(len(colored_feature_maps) ** 0.5), padding=1, normalize=False)

    # Denormalize the original image for display
    def denormalize_image(image_tensor, mean, std):
        num_channels = image_tensor.shape[0]  # Get the number of channels (1 for MNIST, 3 for RGB)
        mean = torch.tensor(mean).view(num_channels, 1, 1).to(image_tensor.device)
        std = torch.tensor(std).view(num_channels, 1, 1).to(image_tensor.device)
        return image_tensor * std + mean

    original_image = denormalize_image(image.squeeze(0).cpu(), mean, std)

    # Calculate the relative sizes of the original image and feature maps
    original_image_width, original_image_height = original_image_size
    feature_maps_width, feature_maps_height = feature_maps_size

    total_width = original_image_width + feature_maps_width
    total_height = max(original_image_height, feature_maps_height)

    # Create a GridSpec layout with relative sizes
    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[original_image_width, feature_maps_width])

    # Plot the original image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original_image.permute(1, 2, 0).clip(0, 1) if original_image.shape[0] == 3 else original_image.squeeze(0), cmap='gray')
    ax1.axis('off')
    ax1.set_title(f"Original Image", fontsize=14)

    # Plot the feature maps
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(grid_image.permute(1, 2, 0).cpu().numpy())
    ax2.axis('off')
    ax2.set_title(f"Feature Maps for Layer {layer}", fontsize=14)

    plt.tight_layout()
    plt.show()


def feature_maps_widget(model, dataset, initial_target_index=0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), original_image_size=(5, 5), feature_maps_size=(10, 10)):
    """
    Creates and displays an interactive widget for visualizing feature maps with the ability to adjust
    layer index, activation type, pooling type, and figure sizes for original image and feature maps.

    Args:
        model (torch.nn.Module): The trained CNN model.
        dataset (torch.utils.data.Dataset): The dataset containing the images.
        initial_target_index (int, optional): Initial index of the image to visualize. Defaults to 0.
        mean (tuple, optional): Mean for image normalization. Defaults to (0.485, 0.456, 0.406).
        std (tuple, optional): Standard deviation for image normalization. Defaults to (0.229, 0.224, 0.225).
        original_image_size (tuple, optional): Size for the original image. Defaults to (5, 5).
        feature_maps_size (tuple, optional): Size for the feature maps. Defaults to (10, 10).
    """
    # Extract the convolutional layers from the model
    conv_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
    num_conv_layers = len(conv_layers)
    
    if num_conv_layers == 0:
        raise ValueError("The model does not contain any convolutional layers.")

    # Adjust slider width based on the number of layers
    slider_width = '200px' if num_conv_layers <= 2 else '400px'

    def update_visualization(change=None):
        """
        Function to update the visualization based on widget values.
        Called when any widget value changes.
        """
        clear_output(wait=True)
        display(ui_and_visualization)  # Display the UI and visualization container again
        visualize_feature_maps(
            dataset=dataset,
            model=model,
            target_index=target_index_widget.value,
            mean=mean,
            std=std,
            layer=layer_widget.value,
            activation=activation_widget.value,
            pooling=pooling_widget.value,
            original_image_size=original_image_size,  # Pass custom size for the original image
            feature_maps_size=feature_maps_size  # Pass custom size for the feature maps
        )

    # Define a fixed width for the entire UI and visualization
    fixed_width = '800px'  # Adjust this width as necessary to match your desired size

    # Widgets for interactive control
    layer_widget = widgets.IntSlider(
        value=1,  # Default layer index
        min=1,    # Minimum layer index
        max=num_conv_layers,  # Set max to the number of convolutional layers
        step=1,
        description='Layer Index',
        layout=widgets.Layout(width=slider_width)  # Adjust slider width based on number of layers
    )

    activation_widget = widgets.Dropdown(
        options=[None, 'relu', 'tanh'],
        value=None,
        description='Activation',
        layout=widgets.Layout(width=fixed_width)  # Set consistent width
    )

    pooling_widget = widgets.Dropdown(
        options=[None, 'max', 'average'],
        value=None,
        description='Pooling',
        layout=widgets.Layout(width=fixed_width)  # Set consistent width
    )

    target_index_widget = widgets.IntSlider(
        min=0,
        max=len(dataset) - 1,
        step=1,
        value=initial_target_index,
        description='Image Index',
        layout=widgets.Layout(width=fixed_width)  # Set consistent width
    )

    # Arrange widgets in two columns
    left_column = VBox([layer_widget, target_index_widget], layout=widgets.Layout(width=fixed_width))
    right_column = VBox([activation_widget, pooling_widget], layout=widgets.Layout(width=fixed_width))

    # Combine columns into one horizontal box with fixed width
    ui = HBox([left_column, right_column], layout=widgets.Layout(width=fixed_width))

    # Combine UI and visualization into a single vertical box with fixed width
    ui_and_visualization = VBox([ui], layout=widgets.Layout(width=fixed_width))

    # Display the arranged widgets after setting up the layout
    display(ui_and_visualization)

    # Attach update function to widget value changes
    layer_widget.observe(update_visualization, names='value')
    activation_widget.observe(update_visualization, names='value')
    pooling_widget.observe(update_visualization, names='value')
    target_index_widget.observe(update_visualization, names='value')

    # Initial visualization
    update_visualization()