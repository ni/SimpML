"""Vision utils."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from fastai.data.core import DataLoaders
from fastai.vision.data import get_image_files
from PIL import Image


def get_img(data: Dict, normalize: bool = False, resize_shape: Tuple = (256, 256)) -> np.ndarray:
    """Load, normalize, and resize an image from disk.

    Args:
        data (Dict): Dictionary containing the image path.
        normalize (bool, optional): Whether to normalize the image. Defaults to False.
        resize_shape (Tuple, optional): The dimensions to resize the image to.
        Defaults to (256, 256).

    Returns:
        np.ndarray: The loaded image array.
    """
    # Load the image
    img_array = np.array(Image.open(data["image_path"]).resize(resize_shape))

    # Normalize the image if required
    if normalize:
        img_array = img_array / 255.0

    # Convert to RGB
    img_array = np.array(Image.fromarray(img_array.astype("uint8")).convert("RGB"))

    return img_array


def find_img_params_and_classes(img_folder_path: str) -> Tuple:
    """Find the normalization requirement and max dimensions of images in a folder.

    Args:
        img_folder_path (str): Path to the image folder.

    Returns:
        Tuple: Whether to normalize, max dimensions, and class dictionary.
    """
    max_width = 0
    max_height = 0
    normalize = False
    classes = []

    # Iterate through all sub-folders and files
    for root, dirs, files in os.walk(img_folder_path):
        # Assuming the first level subfolders are class names
        if root == img_folder_path:
            classes.extend(dirs)

        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):  # Add more image formats if needed
                img_path = os.path.join(root, file)
                img = Image.open(img_path)

                # Update max dimensions
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)

                # Check for normalization
                if not normalize:
                    img_array = np.array(img)
                    if np.any(img_array > 255):
                        normalize = True

    return (
        normalize,
        (max_width, max_height),
        dict(zip(classes, [str(i) for i in range(len(classes))])),
    )


def get_data_from_path_as_df(params: Tuple) -> List:
    """Get image data as a list of dictionaries from a given path.

    Args:
        params (Tuple): Path and a dictionary to map labels to numerical values.

    Returns:
        List: A list of dictionaries containing image data.
    """
    path, dict_values_to_labels = params
    # Define path
    path = Path(path)

    # Get all image files
    files = get_image_files(path)

    # Initialize empty list
    data = []

    # Loop over all files
    for file in files:
        # Get label from parent folder name
        label = file.parent.name
        # Create dictionary and append to list
        data.append(
            {"file_name": file.name, "image_path": file, "label": dict_values_to_labels[label]}
        )

    return data


def get_label(data: Dict) -> Union[str, int]:
    """Get the label from a data dictionary.

    Args:
        data (Dict): Dictionary containing the label.

    Returns:
        Union[str, int]: The label of the image.
    """
    return data["label"]


def count_and_plot_images(root_dir: str, classes: Optional[list] = None) -> plt.Figure:
    """Count images in given directories and plot a bar graph."""
    if classes is None:
        classes = ["Pass", "Fail"]
    # Initialize a dictionary to hold counts
    counts = {}
    # Loop through the classes
    for class_name in classes:
        # Define path to the class directory
        class_dir = os.path.join(root_dir, class_name)

        # Count the number of images (files) in the directory
        num_images = len(
            [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        )

        # Add the count to our dictionary
        counts[class_name] = num_images

    # Display bar plot of counts
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title("Number of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")

    # Text on the top of each bar
    for i in range(len(counts)):
        plt.text(x=i - 0.1, y=list(counts.values())[i] + 1, s=list(counts.values())[i], size=10)

    plt.show()
    return fig


def plot_images(
    dls: DataLoaders, dict_values_to_labels: Dict, n_images: int = 12, n_images_in_row: int = 6
) -> Dict:
    """Plot a grid of sample images per class.

    Args:
        dls (DataLoaders): The data loaders object containing the dataset.
        dict_values_to_labels (Dict): Dictionary mapping class names to labels.
        n_images (int, optional): The total number of images to display for each
        class. Defaults to 12.
        n_images_in_row (int, optional): The number of images in a single row.
        Defaults to 6.

    Returns:
        Dict: A dictionary containing the selected elements per class.
    """
    images_per_class = {}
    n_rows = int(math.ceil(n_images / n_images_in_row))
    for class_name, label in list(dict_values_to_labels.items()):
        selected_elements: List = []
        while len(selected_elements) < n_images:
            # Randomly select an element
            element = random.choice(dls.train_ds)
            # If the label is TensorCategory(1), add it to the list
            if int(element[1]) == int(label):
                selected_elements.append(element)
        f = plt.figure(figsize=(n_images_in_row * 4, n_rows * 3))
        for num, img in enumerate(selected_elements):
            plt.subplot(2, 6, num + 1)
            plt.axis("off")
            plt.title(str(class_name))
            plt.imshow(img[0], cmap="Greys_r")
        f.suptitle(f"{n_images} samples form {class_name} images")
        images_per_class[class_name] = selected_elements
    return images_per_class


def empty_torch_cache() -> None:
    """Empty the PyTorch GPU cache to free up memory."""
    torch.cuda.empty_cache()
