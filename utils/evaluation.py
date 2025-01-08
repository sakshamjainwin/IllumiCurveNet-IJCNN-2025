"""
Image Evaluation Module

This module provides functionality for evaluating image quality using various metrics.
It includes comparison metrics between two images (PSNR, LPIPS, SSIM, MAE) and
no-reference quality metrics (PI, NIQE).
"""

import os
import cv2
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import pyiqa

class ImageEvaluator:
        """
        A class for evaluating image quality using multiple metrics.
    
        This class provides methods to calculate various image quality metrics
        including PSNR, LPIPS, SSIM, MAE, PI, and NIQE.
        """

        def calculate_metrics(self, image1_path, image2_path):
            """
            Calculate comparison metrics between two images.
        
            Args:
                image1_path (str): Path to the first image
                image2_path (str): Path to the second image
            
            Returns:
                tuple: (PSNR, LPIPS, SSIM, MAE) values
            """
            # Load images using OpenCV
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)

            # Calculate Peak Signal-to-Noise Ratio
            psnr_value = peak_signal_noise_ratio(img1, img2)

            # Initialize LPIPS and SSIM metrics using GPU
            lpips_metric = pyiqa.create_metric('lpips').cuda()
            ssim_metric = pyiqa.create_metric('ssim').cuda()

            # Calculate LPIPS and SSIM scores
            lpips_value = lpips_metric(image1_path, image2_path)
            ssim_value = ssim_metric(image1_path, image2_path)

            # Calculate Mean Absolute Error
            mae_value = mean_squared_error(img1, img2)

            return psnr_value, lpips_value, ssim_value, mae_value

        def calculate_pi_niqe(self, image_path):
            """
            Calculate no-reference quality metrics (PI and NIQE) for a single image.
        
            Args:
                image_path (str): Path to the image
            
            Returns:
                tuple: (PI score, NIQE score)
            """
            # Initialize metrics using GPU
            niqe_metric = pyiqa.create_metric('niqe').cuda()
            pi_metric = pyiqa.create_metric('pi').cuda()

            # Calculate scores
            niqe_score = niqe_metric(image_path)
            pi_score = pi_metric(image_path)

            return pi_score, niqe_score

        def evaluate_images(self, folder1, folder2):
            """
            Evaluate and compare all images in two folders using multiple metrics.
        
            Args:
                folder1 (str): Path to the first folder containing images
                folder2 (str): Path to the second folder containing images
            """
            # Get list of image files from both folders
            images1 = os.listdir(folder1)
            images2 = os.listdir(folder2)

            # Verify that both folders contain the same image files
            assert images1 == images2, "Image files in folders do not match"

            # Initialize accumulators for metrics
            psnr_sum = 0
            lpips_sum = 0
            ssim_sum = 0
            mae_sum = 0
            pi_sum = 0
            niqe_sum = 0

            # Process each pair of images
            for image_name in images1:
                image1_path = os.path.join(folder1, image_name)
                image2_path = os.path.join(folder2, image_name)

                # Calculate metrics for current image pair
                psnr, lpips, ssim, mae = self.calculate_metrics(image1_path, image2_path)
                pi, niqe = self.calculate_pi_niqe(image2_path)

                # Accumulate metrics
                psnr_sum += psnr
                lpips_sum += lpips
                ssim_sum += ssim
                mae_sum += mae
                pi_sum += pi
                niqe_sum += niqe

            # Calculate average metrics
            num_images = len(images1)
            psnr_avg = psnr_sum / num_images
            lpips_avg = lpips_sum /num_images
            ssim_avg = ssim_sum / num_images
            mae_avg = mae_sum / num_images
            pi_avg = pi_sum / num_images
            niqe_avg = niqe_sum / num_images

            # Print results
            print(f'Average PSNR: {psnr_avg:.4f}')
            print(f'Average LPIPS: {lpips_avg:.4f}')
            print(f'Average SSIM: ', ssim_avg)
            print(f'Average MAE: {mae_avg:.4f}')
            print(f'Average Perceptual Index (PI): {pi_avg:.4f}')
            print(f'Average NIQE: {niqe_avg:.4f}')
