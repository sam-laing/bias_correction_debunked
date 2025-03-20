import os
import scipy.io
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image

def download_file(url, filename):
    """Download a file with a progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        print(f"Downloading {filename}...")
        with open(filename, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    # URLs for the SVHN dataset
    train_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    extra_url = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"  # Optional extra training data
    
    base_dir = "/fast/slaing/data/vision/svhn"
    os.makedirs(base_dir, exist_ok=True)
    
    train_file = os.path.join(base_dir, "train_32x32.mat")
    test_file = os.path.join(base_dir, "test_32x32.mat")
    extra_file = os.path.join(base_dir, "extra_32x32.mat")
    
    # Check if files already exist
    train_exists = os.path.exists(train_file)
    test_exists = os.path.exists(test_file)
    extra_exists = os.path.exists(extra_file)
    
    if train_exists and test_exists:
        print("SVHN dataset files already exist. Skipping download.")
    else:
        # Download train set
        if not train_exists:
            if download_file(train_url, train_file):
                print("Train set downloaded successfully.")
            else:
                print("Failed to download train set.")
                return
        
        # Download test set
        if not test_exists:
            if download_file(test_url, test_file):
                print("Test set downloaded successfully.")
            else:
                print("Failed to download test set.")
                return
        
        # Download extra set (optional)
        if not extra_exists and input("Download extra training data? (y/n): ").lower() == 'y':
            if download_file(extra_url, extra_file):
                print("Extra set downloaded successfully.")
            else:
                print("Failed to download extra set.")
    
    # Verify the data by loading a small portion
    try:
        print("Verifying train data...")
        train_data = scipy.io.loadmat(train_file)
        print(f"Train data shape: {train_data['X'].shape}")
        print(f"Train labels shape: {train_data['y'].shape}")
        
        print("Verifying test data...")
        test_data = scipy.io.loadmat(test_file)
        print(f"Test data shape: {test_data['X'].shape}")
        print(f"Test labels shape: {test_data['y'].shape}")
        
        if extra_exists:
            print("Verifying extra data...")
            extra_data = scipy.io.loadmat(extra_file)
            print(f"Extra data shape: {extra_data['X'].shape}")
            print(f"Extra labels shape: {extra_data['y'].shape}")
        
        print("\nSVHN dataset has been successfully downloaded and verified.")
        print(f"Data located at: {base_dir}")
        
        # Display a sample image for visual verification
        sample_idx = np.random.randint(0, train_data['X'].shape[3])
        sample_img = train_data['X'][:, :, :, sample_idx]
        sample_label = train_data['y'][sample_idx][0]
        
        print(f"Sample image label: {sample_label}")
        print("You can manually display the sample image if PIL is available.")
        
    except Exception as e:
        print(f"Error verifying data: {e}")

if __name__ == "__main__":
    main()