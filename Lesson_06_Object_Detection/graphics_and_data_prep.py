import os
import re
import cv2
import random
import shutil
import zipfile
import numpy as np
import torch
from pathlib import Path
from matplotlib.patches import Rectangle
from torchvision import ops

import urllib.request
import matplotlib.pyplot as plt

def prepare_penn_fudan_dataset(target_dir: Path, seed=42):
    # Define paths for the dataset and processed directories
    dataset_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    dataset_zip = target_dir / "PennFudanPed.zip"
    dataset_dir = target_dir / "PennFudanPed"
    
    # Check if the dataset has already been processed
    processed_dir = dataset_dir / "train" / "images"
    if processed_dir.exists():
        print("Processed PennFudan dataset already exists. Skipping preparation.")
        return

    # Step 1: Download and unzip the dataset if not already downloaded
    if not dataset_zip.exists():
        urllib.request.urlretrieve(dataset_url, dataset_zip)
    
    # Unzip dataset if not already unzipped
    if not dataset_dir.exists():
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    
    # Define paths for images, annotations, masks, and labels
    images_dir = dataset_dir / "PNGImages"
    annotations_dir = dataset_dir / "Annotation"
    masks_dir = dataset_dir / "PedMasks"
    labels_dir = dataset_dir / "labels"
    
    # Create labels directory for YOLO annotations
    labels_dir.mkdir(exist_ok=True)
    
    # Step 2: Convert annotations to YOLO format and store in labels directory
    def convert_annotation_to_yolo(annotation_file, image_width, image_height):
        yolo_annotations = []
        
        with annotation_file.open('r') as file:
            lines = file.readlines()
            
            for line in lines:
                if "Bounding box" in line:
                    # Extract coordinates using regular expressions
                    match = re.search(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", line)
                    if match:
                        Xmin, Ymin, Xmax, Ymax = map(int, match.groups())
                        
                        # Convert to YOLO format
                        x_center = (Xmin + Xmax) / 2 / image_width
                        y_center = (Ymin + Ymax) / 2 / image_height
                        width = (Xmax - Xmin) / image_width
                        height = (Ymax - Ymin) / image_height
                        yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}\n")
                        
        return yolo_annotations

    # Process all annotation files and save in the labels folder
    for annotation_file in annotations_dir.glob("*.txt"):
        image_filename = annotation_file.stem + ".png"
        image_path = images_dir / image_filename
        
        if image_path.exists():
            # Read image dimensions
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            
            # Convert annotation to YOLO format
            yolo_data = convert_annotation_to_yolo(annotation_file, w, h)
            
            # Save YOLO annotation file
            yolo_label_path = labels_dir / annotation_file.name
            with yolo_label_path.open('w') as f:
                f.writelines(yolo_data)

    # Step 3: Split dataset into train and val subsets
    random.seed(seed)
    image_files = sorted([f for f in images_dir.glob("*.png")])
    random.shuffle(image_files)
    
    split_index = int(0.8 * len(image_files))
    train_files = {f.stem for f in image_files[:split_index]}  # Use base filenames for faster lookups
    val_files = {f.stem for f in image_files[split_index:]}
    
    # Create new train/val directory structure with subfolders
    for split in ["train", "val"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "annotations").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "masks").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Helper function to move files
    def move_files(files, src_dir, train_dir, val_dir, is_mask=False):
        for file in files:
            base_name = file.stem
            if is_mask:
                base_name = base_name.replace("_mask", "")  # Remove "_mask" to match with image names
            
            if base_name in train_files:
                dst_dir = train_dir
            elif base_name in val_files:
                dst_dir = val_dir
            else:
                continue  # Skip files not in train or val sets
            
            src_file = src_dir / file.name
            dst_file = dst_dir / file.name
            
            # Move the file if it exists
            if src_file.exists():
                shutil.move(str(src_file), str(dst_file))

    # Move images, annotations, masks, and labels based on the split
    move_files(image_files, images_dir, dataset_dir / "train" / "images", dataset_dir / "val" / "images")
    move_files(annotations_dir.glob("*.txt"), annotations_dir, dataset_dir / "train" / "annotations", dataset_dir / "val" / "annotations")
    move_files(masks_dir.glob("*.png"), masks_dir, dataset_dir / "train" / "masks", dataset_dir / "val" / "masks", is_mask=True)
    move_files(labels_dir.glob("*.txt"), labels_dir, dataset_dir / "train" / "labels", dataset_dir / "val" / "labels")
    
    # Step 4: Remove original directories
    shutil.rmtree(images_dir)
    shutil.rmtree(annotations_dir)
    shutil.rmtree(masks_dir)
    shutil.rmtree(labels_dir)

    print("Dataset preparation complete, and original directories removed!")


    # Helper function to display boxes on an axis
def display_boxes(ax, boxes, color, label):
    """
    Display bounding boxes on an axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to display the bounding boxes.
    boxes (numpy.ndarray): The bounding boxes to be displayed. Each box should have four coordinates (x, y, xmax, ymax).
    color (str): The color of the bounding boxes.
    label (str): The label for the bounding boxes.

    Returns:
    None
    """
    boxes = np.atleast_2d(boxes)  # Ensure boxes are 2D
    for box in boxes:
        if len(box) == 4:  # Check if box has four coordinates
            x, y, xmax, ymax = box
            width, height = xmax - x, ymax - y
            rect = Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none', label=label)
            ax.add_patch(rect)

# Function to denormalize the image tensor for display
def denormalize_torch(image_tensor, mean, std):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return image_tensor * std[:, None, None] + mean[:, None, None]

# Combined function to display images with ground truth and optional predictions
def display_images_and_boxes(dataset, num_samples=3, model=None, confidence_threshold=0.5, iou_threshold=0.4, indices=None, figsize=(6,6), 
                             denormalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Display random samples from the dataset with ground truth boxes. If a model is provided, it also displays predictions.

    Parameters:
    - dataset: The dataset containing the images and ground truth annotations.
    - num_samples: The number of random samples to display (default: 3).
    - model: The object detection model. If None, only ground truth boxes are displayed.
    - confidence_threshold: The confidence threshold for filtering predictions (default: 0.5).
    - iou_threshold: The IoU threshold for Non-Max Suppression (default: 0.4).
    - indices (list, optional): The indices of the samples to display. If None, random samples will be selected.
    - figsize (tuple, optional): The size of the figure (default: (6,6)).
    - denormalize (bool): Whether to denormalize the image for display.
    """
    if indices is None:
        indices = random.sample(range(len(dataset)), num_samples)
    
    device = None
    if model is not None:
        model.eval()
        device = next(model.parameters()).device

    for idx in indices:
        # Load the image and target directly from the dataset
        image, target = dataset[idx]
        
        # Denormalize the image if needed
        if denormalize:
            image = denormalize_torch(image, mean, std)
        
        img_np = image.permute(1, 2, 0).cpu().numpy() if device else image.permute(1, 2, 0).numpy()
        
        # Prepare the figure
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img_np)
        
        # Check if ground truth boxes are available
        if "boxes" in target and target["boxes"].numel() > 0:
            gt_boxes = target["boxes"].cpu().numpy() if device else target["boxes"].numpy()
            display_boxes(ax, gt_boxes, color="blue", label="Ground Truth")
        else:
            print(f"No ground truth boxes found for index {idx}")

        if model is not None:
            # Move image to device if model is specified
            image = image.to(device)
            with torch.no_grad():
                output = model([image])[0]
            
            # Filter predictions by confidence threshold
            pred_boxes = output["boxes"].cpu().detach().numpy()
            pred_scores = output["scores"].cpu().detach().numpy()
            
            # Apply confidence threshold
            relevant_boxes = pred_boxes[pred_scores > confidence_threshold]
            
            # Apply Non-Maximum Suppression if needed
            if iou_threshold < 1.0 and len(relevant_boxes) > 0:
                keep = ops.nms(torch.tensor(relevant_boxes), torch.tensor(pred_scores[pred_scores > confidence_threshold]), iou_threshold)
                relevant_boxes = relevant_boxes[keep]
            
            # Display predicted boxes in orange
            display_boxes(ax, relevant_boxes, color="orange", label="Prediction")
            
            # Set up legend with custom labels for ground truth and predictions
            handles = [
                Rectangle((0, 0), 1, 1, edgecolor="blue", facecolor='none', label="Ground Truth"),
                Rectangle((0, 0), 1, 1, edgecolor="orange", facecolor='none', label="Prediction")
            ]
        else:
            # Only ground truth legend
            handles = [Rectangle((0, 0), 1, 1, edgecolor="blue", facecolor='none', label="Ground Truth")]
        
        plt.legend(handles=handles)
        plt.axis('off')
        plt.show()

