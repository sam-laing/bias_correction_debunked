#!/usr/bin/env python3
"""
Script to download and save Vision Transformer (ViT) pretrained weights on a compute cluster.
"""

import torch
import os
import logging
from torchvision.models import (
    vit_b_16, vit_b_32, vit_l_16, vit_l_32,
    ViT_B_16_Weights, ViT_B_32_Weights,
    ViT_L_16_Weights, ViT_L_32_Weights
)

# Configuration
MODEL_TYPE = "vit_b_16"  # Choose from: vit_b_16, vit_b_32, vit_l_16, vit_l_32
OUTPUT_DIR = "/fast/slaing/pretrained_weights/VIT_tiny_imagenet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "download_weights.log")),
        logging.StreamHandler()
    ]
)

def download_vit_weights(model_type, output_dir):
    """Download and save ViT weights with error handling."""
    try:
        logging.info(f"Initializing {model_type} with pretrained weights...")
        
        model_map = {
            "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
            "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
            "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1),
            "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1)
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model_class, weights = model_map[model_type]
        model = model_class(weights=weights)
        
        output_path = os.path.join(output_dir, f"{model_type}_weights.pth")
        logging.info(f"Saving weights to {output_path}")
        torch.save(model.state_dict(), output_path)
        
        # Verify the save
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # in MB
            logging.info(f"Weights saved successfully! File size: {file_size:.2f} MB")
        else:
            raise RuntimeError("Weight file not created successfully")
            
    except Exception as e:
        logging.error(f"Error downloading weights: {str(e)}")
        raise

if __name__ == "__main__":
    logging.info(f"Starting weight download for {MODEL_TYPE}")
    try:
        download_vit_weights(MODEL_TYPE, OUTPUT_DIR)
        logging.info("Process completed successfully")
    except Exception as e:
        logging.critical(f"Fatal error in main process: {str(e)}")
        exit(1)