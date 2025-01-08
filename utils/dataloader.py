"""
This module provides functionality for loading and processing low-light images for training.
It includes utilities for dataset creation and image preprocessing.
"""

import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import random

# Set random seed for reproducibility
random.seed(1143)


def populate_train_list(lowlight_images_path):
    """
    Creates a list of file paths for all images in the given directory and its subdirectories.
    
    Args:
        lowlight_images_path (str): Path to the directory containing low-light images
        
    Returns:
        list: Shuffled list of file paths
    """
    file_paths_and_names = []

    # Walk through directory tree and collect all file paths
    for dirpath, dirnames, filenames in os.walk(lowlight_images_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths_and_names.append(file_path)

    # Shuffle the file paths for randomization
    random.shuffle(file_paths_and_names)

    return file_paths_and_names


class lowlight_loader(data.Dataset):
    """
    Dataset class for loading and preprocessing low-light images.
    
    Args:
        lowlight_images_path (str): Path to the directory containing low-light images
    """
    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = 128  # Target size for image resizing
        
        print("Total training examples:", len(self.train_list))

        # Limit dataset size to 200 if more than 1000 images are available
        if len(self.train_list) > 1000:
            self.train_list = random.sample(self.train_list, 200)
        self.data_list = self.train_list

    def __getitem__(self, index):
        """
        Loads and preprocesses a single image from the dataset.
        
        Args:
            index (int): Index of the image to load
            
        Returns:
            torch.Tensor: Preprocessed image tensor in CHW format
        """
        data_lowlight_path = self.data_list[index]
		
        # Load and preprocess the image
        data_lowlight = Image.open(data_lowlight_path)
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.Resampling.LANCZOS)
        data_lowlight = (np.asarray(data_lowlight)/255.0)  # Normalize to [0,1]
        data_lowlight = torch.from_numpy(data_lowlight).float()

        return data_lowlight.permute(2,0,1)  # Convert to CHW format

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        
        Returns:
            int: Number of images
        """
        return len(self.data_list)





