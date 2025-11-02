"""
Testing script for IllumiCurveNet.

This script implements a testing pipeline for our Low Light Image Enhancement model allowing configurations through command line arguments.
"""

import os
import argparse
import glob
import numpy as np
import torch
import torchvision
import torch.optim
import model
from PIL import Image

def test(image_path, config):
    """
    Process and enhance a single low-light image using the IllumiCurveNet model.
    
    Args:
        image_path (str): Path to the input low-light image
        
    Returns:
        None: Saves the enhanced image to the result directory
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    # Load and preprocess the image
    data_lowlight = Image.open(image_path)

    # Resize to dimensions divisible by 8 (to avoid dimension mismatch errors in the model's concatenations)
    width = (data_lowlight.size[0] // 8) * 8
    height = (data_lowlight.size[1] // 8) * 8
    data_lowlight = data_lowlight.resize((width, height))

    data_lowlight = (np.asarray(data_lowlight)/255.0)  # Normalize to [0,1]
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)  # Change to channel-first format
    data_lowlight = data_lowlight.cuda().unsqueeze(0)  # Add batch dimension and move to GPU

    # Initialize and load the model
    IC_net = model.illumi_curve_net().cuda()
    IC_net.load_state_dict(torch.load(config.pretrain_snapshot))

    # Generate enhanced image
    enhanced_image,_ = IC_net(data_lowlight)

    # Create output path and save enhanced image
    result_path = image_path.replace('test_data','result')
    result_dir = os.path.dirname(result_path)

    os.makedirs(result_dir, exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description='IllumiCurveNet Testing Script')
    
    # Define command line arguments
    parser.add_argument('--lowlight_images_path', type=str, default="data/test_data/", help='Path to low-light testing images')
    parser.add_argument('--pretrain_snapshot', type=str, default= "snapshots/model-best.pth", help='Pretrained model snapshot')

    config = parser.parse_args()

    # Process all images in the specified directory
    with torch.no_grad():
        file_path = config.lowlight_images_path
        file_list = os.listdir(file_path)

        for file_name in file_list:
            test_list = glob.glob(file_path+file_name+"/*") 
            for image in test_list:
                test(image, config)
        
        print("Done")
    
