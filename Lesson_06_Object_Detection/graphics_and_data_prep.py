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
import matplotlib.pyplot as plt
import requests
from torchvision import ops
import ipywidgets as widgets
from IPython.display import display

def download_pennfudanped(target_dir: Path) -> None:
    """
    Downloads and unzips the PennFudanPed dataset into the specified directory.
    Args:
        target_dir (Path): The directory where the dataset will be downloaded and unzipped.
    Returns:
        None
    """
    dataset_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = target_dir / "PennFudanPed.zip"
    dataset_path = target_dir / "PennFudanPed"

    if dataset_path.exists():
        print("PennFudanPed dataset already exists.")
        return
    
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading PennFudanPed dataset...")
    response = requests.get(dataset_url, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Download complete.")

    print("Unzipping the dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print("Unzipping complete.")
    zip_path.unlink()
    print("Zip file deleted. PennFudanPed dataset is ready.")

def prepare_penn_fudan_yolo(target_dir: Path, seed=42):
    # Define paths for the dataset and processed directories
    dataset_dir = target_dir / "PennFudanPed"
    yolo_dataset_dir = target_dir / "PennFudanPedYOLO"

    # Step 1: Check if PennFudanPedYOLO already exists
    if yolo_dataset_dir.exists():
        print("PennFudanPedYOLO already exists. Stopping preparation.")
        return

    # Step 2: Download and unzip the dataset if necessary
    download_pennfudanped(target_dir)

    # Step 3: Copy the extracted dataset to PennFudanPedYOLO
    shutil.copytree(dataset_dir, yolo_dataset_dir)
    dataset_dir = yolo_dataset_dir

    # Define paths for images, annotations, masks, and labels
    images_dir = dataset_dir / "PNGImages"
    annotations_dir = dataset_dir / "Annotation"
    masks_dir = dataset_dir / "PedMasks"
    labels_dir = dataset_dir / "labels"
    
    # Create labels directory for YOLO annotations
    labels_dir.mkdir(exist_ok=True)
    
    # Step 4: Convert annotations to YOLO format and store in labels directory
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

    # Step 5: Split dataset into train and val subsets
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
    
    # Step 6: Remove original directories
    shutil.rmtree(images_dir)
    shutil.rmtree(annotations_dir)
    shutil.rmtree(masks_dir)
    shutil.rmtree(labels_dir)

    print("Dataset preparation complete, and original directories removed!")

'''
def prepare_penn_fudan_yolo(target_dir: Path, seed=42):
    # Define paths for the dataset and processed directories
    dataset_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    dataset_zip = target_dir / "PennFudanPed.zip"
    dataset_dir = target_dir / "PennFudanPed"
    yolo_dataset_dir = target_dir / "PennFudanPedYOLO"

    # Step 1: Check if PennFudanPedYOLO already exists
    if yolo_dataset_dir.exists():
        print("PennFudanPedYOLO already exists. Stopping preparation.")
        return

    # Step 2: Download and unzip the dataset if not already downloaded
    if not dataset_zip.exists():
        urllib.request.urlretrieve(dataset_url, dataset_zip)
    
    # Unzip dataset if not already unzipped
    if not dataset_dir.exists():
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

    # Step 3: Copy the extracted dataset to PennFudanPedYOLO
    shutil.copytree(dataset_dir, yolo_dataset_dir)
    dataset_dir = yolo_dataset_dir

    # Define paths for images, annotations, masks, and labels
    images_dir = dataset_dir / "PNGImages"
    annotations_dir = dataset_dir / "Annotation"
    masks_dir = dataset_dir / "PedMasks"
    labels_dir = dataset_dir / "labels"
    
    # Create labels directory for YOLO annotations
    labels_dir.mkdir(exist_ok=True)
    
    # Step 4: Convert annotations to YOLO format and store in labels directory
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

    # Step 5: Split dataset into train and val subsets
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
    
    # Step 6: Remove original directories
    shutil.rmtree(images_dir)
    shutil.rmtree(annotations_dir)
    shutil.rmtree(masks_dir)
    shutil.rmtree(labels_dir)

    print("Dataset preparation complete, and original directories removed!")

def prepare_penn_fudan_orig(target_dir: Path, seed=42):
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
'''

def overlay_masks(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    """
    Overlay a binary mask outline on an image.

    Parameters:
    - image (numpy.ndarray): The image on which to overlay the mask.
    - mask (numpy.ndarray): The binary mask (1 for mask, 0 for background).
    - color (tuple): The color for the mask outline in (B, G, R) format.
    - thickness (int): The thickness of the mask outline (default: 1).

    Returns:
    - image (numpy.ndarray): The image with the mask outline overlaid.
    """
    mask = (mask > 0.5).astype(np.uint8)
    if len(mask.shape) > 2:
        mask = mask.squeeze()  # Ensure single-channel mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)
    return image

def denormalize_image(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """
    Denormalizes an image by applying the given mean and standard deviation.
    Parameters:
    image (np.ndarray): The normalized image array.
    mean (list[float]): The mean values used for normalization.
    std (list[float]): The standard deviation values used for normalization.
    Returns:
    np.ndarray: The denormalized image array, with values clipped between 0 and 1.
    """
    mean = np.array(mean)
    std = np.array(std)
    return (image * std + mean).clip(0, 1)

def display_images_and_masks(dataset, num_samples=3, model=None, indices=None, figsize=(4, 4), overlay=True, 
                             denormalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Display random samples from the dataset with ground truth masks. If a model is provided, it also displays predicted masks.

    Parameters:
    - dataset: The dataset containing the images and ground truth masks.
    - num_samples: The number of random samples to display (default: 3).
    - model: The segmentation model. If None, only ground truth masks are displayed.
    - indices (list, optional): The indices of the samples to display. If None, random samples will be selected.
    - figsize (tuple, optional): The size of the display figure (default: (6, 6)).
    - overlay (bool, optional): If True, overlays the mask on the image. If False, displays image and mask side-by-side.
    - denormalize (bool, optional): If True, denormalizes the images before displaying (default: False).
    """
    if indices is None:
        indices = random.sample(range(len(dataset)), num_samples)
    
    device = None
    if model is not None:
        model.eval()
        device = next(model.parameters()).device

    for idx in indices:
        # Load the image and ground truth mask
        image, target_mask = dataset[idx]
        
        # Move image to device for model prediction
        pred_mask = None
        if model is not None:
            image_for_model = image.to(device)
            with torch.no_grad():
                logits = model(image_for_model.unsqueeze(0))[0].squeeze().cpu()
                pred_mask = torch.sigmoid(logits.clone().detach()).numpy()

        # Convert image to numpy and optionally denormalize
        img_np = image.permute(1, 2, 0).cpu().numpy() if device else image.permute(1, 2, 0).numpy()
        if denormalize:
            img_np = denormalize_image(img_np, mean, std)
        img_np = (img_np * 255).astype(np.uint8)  # Scale for OpenCV display

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gt_mask = target_mask.cpu().numpy() if device else target_mask.numpy()

        if overlay:
            # Overlay ground truth mask in blue
            img_overlay = overlay_masks(img_bgr.copy(), gt_mask, color=(255, 0, 0))
            
            # Overlay predicted mask in orange if model is provided
            if pred_mask is not None:
                pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize predicted mask
                img_overlay = overlay_masks(img_overlay, pred_mask, color=(0, 165, 255))

            img_rgb_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=figsize)
            plt.imshow(img_rgb_overlay)
            plt.axis('off')
            plt.show()

        else:
            # Prepare separate images for side-by-side display
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_rgb = (gt_mask.squeeze() * 255).astype(np.uint8)
            mask_rgb = np.stack([mask_rgb] * 3, axis=-1)  # Convert to 3-channel grayscale
            
            images = [img_rgb, mask_rgb]
            titles = ['Image', 'Ground Truth Mask']
            
            # Add predicted mask if model is provided
            if pred_mask is not None:
                pred_rgb = (pred_mask * 255).astype(np.uint8)  # Display as binary mask in grayscale
                pred_rgb = np.stack([pred_rgb] * 3, axis=-1)  # Convert to 3-channel grayscale
                images.append(pred_rgb)
                titles.append('Predicted Mask')
            
            plt.figure(figsize=(figsize[0] * len(images), figsize[1]))
            for i, (title, img) in enumerate(zip(titles, images)):
                plt.subplot(1, len(images), i + 1)
                plt.imshow(img)
                plt.title(title)
                plt.axis('off')
            plt.show()


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
def denormalize_torch(image_tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """
    Denormalizes a PyTorch tensor image using the provided mean and standard deviation.

    Args:
        image_tensor (torch.Tensor): The normalized image tensor to be denormalized.
        mean (list[float]): The mean values used for normalization.
        std (list[float]): The standard deviation values used for normalization.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return image_tensor * std[:, None, None] + mean[:, None, None]

'''
# Combined function to display images with ground truth and optional predictions
def display_images_and_boxes(dataset, num_samples=3, model=None, confidence_threshold=0.5, iou_threshold=0.4, indices=None, 
                             figsize=(6, 6), denormalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                             show_confidence=False, show_prediction=False):
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
    - show_confidence (bool): Whether to show confidence scores on predicted boxes.
    - show_prediction (bool): Whether to show predicted class labels on boxes.
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
            pred_labels = output["labels"].cpu().detach().numpy()
            
            # Apply confidence threshold
            relevant_boxes = pred_boxes[pred_scores > confidence_threshold]
            relevant_scores = pred_scores[pred_scores > confidence_threshold]
            relevant_labels = pred_labels[pred_scores > confidence_threshold]

            # Apply Non-Maximum Suppression if needed
            if iou_threshold < 1.0 and len(relevant_boxes) > 0:
                keep = ops.nms(torch.tensor(relevant_boxes), torch.tensor(relevant_scores), iou_threshold)
                relevant_boxes = relevant_boxes[keep]
                relevant_scores = relevant_scores[keep]
                relevant_labels = relevant_labels[keep]

            # Ensure they are arrays even if there's only one relevant box
            if relevant_boxes.ndim == 1:
                relevant_boxes = np.expand_dims(relevant_boxes, axis=0)
                relevant_scores = np.expand_dims(relevant_scores, axis=0)
                relevant_labels = np.expand_dims(relevant_labels, axis=0)
            
            # Display predicted boxes in orange
            display_boxes(ax, relevant_boxes, color="orange", label="Prediction")
            
            # Show confidence scores and predicted class labels if required
            if show_confidence or show_prediction:
                class_names = getattr(dataset, 'classes', None)
                for box, score, label in zip(relevant_boxes, relevant_scores, relevant_labels):
                    x, y, _, _ = box
                    text = ""
                    if show_confidence:
                        text += f"{score:.2f}"
                    if show_prediction:
                        if class_names:
                            text += f", {class_names[label]}"
                        else:
                            text += f", {label}"
                    ax.text(x, y, text, color="white", fontsize=8, bbox=dict(facecolor="orange", alpha=0.5))
            
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
'''

'''
def display_images_and_boxes(dataset, num_samples=3, model=None, confidence_threshold=0.5, iou_threshold=0.4, indices=None, 
                             figsize=(6, 6), denormalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                             show_confidence=False, show_prediction=False, filter_mode='nms'):
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
    - show_confidence (bool): Whether to show confidence scores on predicted boxes.
    - show_prediction (bool): Whether to show predicted class labels on boxes.
    - filter_mode (str): The mode for filtering predictions ('nms' for Non-Max Suppression, 'gt' for ground truth filtering).
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
            pred_labels = output["labels"].cpu().detach().numpy()
            
            # Apply confidence threshold
            relevant_boxes = pred_boxes[pred_scores > confidence_threshold]
            relevant_scores = pred_scores[pred_scores > confidence_threshold]
            relevant_labels = pred_labels[pred_scores > confidence_threshold]

            if filter_mode == 'nms':
                # Apply Non-Maximum Suppression if needed
                if iou_threshold < 1.0 and len(relevant_boxes) > 0:
                    keep = ops.nms(torch.tensor(relevant_boxes), torch.tensor(relevant_scores), iou_threshold)
                    relevant_boxes = relevant_boxes[keep]
                    relevant_scores = relevant_scores[keep]
                    relevant_labels = relevant_labels[keep]
            elif filter_mode == 'gt':
                # Filter out predictions with IoU below threshold with any ground truth box
                if len(relevant_boxes) > 0 and len(gt_boxes) > 0:
                    ious = ops.box_iou(torch.tensor(relevant_boxes), torch.tensor(gt_boxes)).numpy()
                    keep = np.any(ious > iou_threshold, axis=1)
                    relevant_boxes = relevant_boxes[keep]
                    relevant_scores = relevant_scores[keep]
                    relevant_labels = relevant_labels[keep]

            # Ensure they are arrays even if there's only one relevant box
            if relevant_boxes.ndim == 1:
                relevant_boxes = np.expand_dims(relevant_boxes, axis=0)
                relevant_scores = np.expand_dims(relevant_scores, axis=0)
                relevant_labels = np.expand_dims(relevant_labels, axis=0)
            
            # Display predicted boxes in orange
            display_boxes(ax, relevant_boxes, color="orange", label="Prediction")
            
            # Show confidence scores and predicted class labels if required
            if show_confidence or show_prediction:
                class_names = getattr(dataset, 'classes', None)
                for box, score, label in zip(relevant_boxes, relevant_scores, relevant_labels):
                    x, y, _, _ = box
                    text = ""
                    if show_confidence:
                        text += f"{score:.2f}"
                    if show_prediction:
                        if class_names:
                            text += f", {class_names[label]}"
                        else:
                            text += f", {label}"
                    ax.text(x, y, text, color="white", fontsize=8, bbox=dict(facecolor="orange", alpha=0.5))
            
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
'''

def predict_boxes(dataset, index, model, mean=None, std=None):
    """
    Predict bounding boxes and labels for a given image in the dataset.

    Parameters:
    - dataset: The dataset containing the images and ground truth annotations.
    - index: The index of the image to predict on.
    - model: The object detection model.
    - mean: The mean values for normalization (default: None).
    - std: The standard deviation values for normalization (default: None).

    Returns:
    - image: The denormalized image.
    - gt_boxes: The ground truth bounding boxes.
    - gt_labels: The ground truth labels.
    - pred_boxes: The predicted bounding boxes.
    - pred_scores: The predicted scores.
    - pred_labels: The predicted labels.
    """
    image, target = dataset[index]

    if mean is not None and std is not None:
        image = denormalize_torch(image, mean, std)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        output = model([image.to(device)])[0]

    gt_boxes = target["boxes"].cpu().numpy() if "boxes" in target else None
    gt_labels = target["labels"].cpu().numpy() if "labels" in target else None
    pred_boxes = output["boxes"].cpu().detach().numpy()
    pred_scores = output["scores"].cpu().detach().numpy()
    pred_labels = output["labels"].cpu().detach().numpy()

    return image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels


def display_images_and_boxes(dataset, num_samples=3, model=None, confidence_threshold=0.5, iou_threshold=0.4, indices=None, 
                             figsize=(6, 6), denormalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                             show_confidence=False, show_prediction=False, filter_mode='nms'):
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
    - show_confidence (bool): Whether to show confidence scores on predicted boxes.
    - show_prediction (bool): Whether to show predicted class labels on boxes.
    - filter_mode (str): The mode for filtering predictions ('nms' for Non-Max Suppression, 'gt' for ground truth filtering).
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
                output = predict_boxes(dataset, idx, model, mean, std)
            
            # Filter predictions by confidence threshold
            pred_boxes = output["boxes"].cpu().detach().numpy()
            pred_scores = output["scores"].cpu().detach().numpy()
            pred_labels = output["labels"].cpu().detach().numpy()
            
            # Apply confidence threshold
            relevant_boxes = pred_boxes[pred_scores > confidence_threshold]
            relevant_scores = pred_scores[pred_scores > confidence_threshold]
            relevant_labels = pred_labels[pred_scores > confidence_threshold]

            if filter_mode == 'nms':
                # Apply Non-Maximum Suppression if needed
                if iou_threshold < 1.0 and len(relevant_boxes) > 0:
                    keep = ops.nms(torch.tensor(relevant_boxes), torch.tensor(relevant_scores), iou_threshold)
                    relevant_boxes = relevant_boxes[keep]
                    relevant_scores = relevant_scores[keep]
                    relevant_labels = relevant_labels[keep]
            elif filter_mode == 'gt':
                # Filter out predictions with IoU below threshold with any ground truth box
                if len(relevant_boxes) > 0 and len(gt_boxes) > 0:
                    ious = ops.box_iou(torch.tensor(relevant_boxes), torch.tensor(gt_boxes)).numpy()
                    keep = np.any(ious > iou_threshold, axis=1)
                    relevant_boxes = relevant_boxes[keep]
                    relevant_scores = relevant_scores[keep]
                    relevant_labels = relevant_labels[keep]

            # Ensure they are arrays even if there's only one relevant box
            if relevant_boxes.ndim == 1:
                relevant_boxes = np.expand_dims(relevant_boxes, axis=0)
                relevant_scores = np.expand_dims(relevant_scores, axis=0)
                relevant_labels = np.expand_dims(relevant_labels, axis=0)
            
            # Display predicted boxes in orange
            display_boxes(ax, relevant_boxes, color="orange", label="Prediction")
            
            # Show confidence scores and predicted class labels if required
            if show_confidence or show_prediction:
                class_names = getattr(dataset, 'classes', None)
                for box, score, label in zip(relevant_boxes, relevant_scores, relevant_labels):
                    x, y, _, _ = box
                    text = ""
                    if show_confidence:
                        text += f"{score:.2f}"
                    if show_prediction:
                        if class_names:
                            text += f", {class_names[label]}"
                        else:
                            text += f", {label}"
                    ax.text(x, y, text, color="white", fontsize=8, bbox=dict(facecolor="orange", alpha=0.5))
            
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

        import matplotlib.pyplot as plt

def display_image_with_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels):
    """
    Display an image with ground truth and predicted bounding boxes.

    Parameters:
    - image: The image to display.
    - gt_boxes: The ground truth bounding boxes.
    - gt_labels: The ground truth labels.
    - pred_boxes: The predicted bounding boxes.
    - pred_scores: The predicted scores.
    - pred_labels: The predicted labels.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    # Display ground truth boxes
    if gt_boxes is not None:
        for box, label in zip(gt_boxes, gt_labels):
            x, y, w, h = box
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue')
            ax.add_patch(rect)
            ax.text(x, y, label, color='blue', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Display predicted boxes
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red')
        ax.add_patch(rect)
        ax.text(x, y, f'{label}: {score:.2f}', color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.show()

def display_image_with_dropdown(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels):
    """
    Display an image with ground truth and predicted bounding boxes, and a dropdown to select the class to display.

    Parameters:
    - image: The image to display.
    - gt_boxes: The ground truth bounding boxes.
    - gt_labels: The ground truth labels.
    - pred_boxes: The predicted bounding boxes.
    - pred_scores: The predicted scores.
    - pred_labels: The predicted labels.
    """
    class_names = np.unique(np.concatenate((gt_labels, pred_labels)))
    dropdown = widgets.Dropdown(options=class_names, description='Class:')
    output = widgets.Output()

    def on_dropdown_change(change):
        output.clear_output()
        selected_class = change.new
        selected_gt_boxes = gt_boxes[gt_labels == selected_class] if gt_boxes is not None else None
        selected_pred_boxes = pred_boxes[pred_labels == selected_class]
        selected_pred_scores = pred_scores[pred_labels == selected_class]
        selected_pred_labels = pred_labels[pred_labels == selected_class]
        with output:
            display_image_with_boxes(image, selected_gt_boxes, [selected_class] * len(selected_gt_boxes),
                                        selected_pred_boxes, selected_pred_scores, selected_pred_labels)

    dropdown.observe(on_dropdown_change, names='value')
    on_dropdown_change({'new': class_names[0]})

    display(widgets.VBox([dropdown, output]))

# Example usage:
image = np.random.rand(100, 100, 3)
gt_boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
gt_labels = np.array(['cat', 'dog'])
pred_boxes = np.array([[15, 15, 25, 25], [35, 35, 45, 45]])
pred_scores = np.array([0.9, 0.8])
pred_labels = np.array(['cat', 'dog'])
display_image_with_dropdown(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels)