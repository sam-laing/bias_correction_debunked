import os
import requests
from tqdm import tqdm
import zipfile
import shutil

def main():
    # URL for the Tiny ImageNet dataset
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_name = "tiny-imagenet-200.zip"
    extract_dir = "/fast/slaing/data/vision/tiny-imagenet-200"

    # Check if dataset directory already exists
    if os.path.exists(extract_dir) and os.path.isdir(extract_dir):
        print(f"Dataset directory {extract_dir} already exists. Skipping download.")
    else:
        # Create the directory to store the dataset
        os.makedirs(extract_dir, exist_ok=True)
        
        # Download the dataset with progress bar
        def download_file(url, filename):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                
                print(f"Downloading {dataset_name}...")
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
        
        download_success = download_file(url, dataset_name)
        
        if download_success:
            print("Download complete.")
            
            # Extract the dataset
            try:
                print(f"Extracting {dataset_name}...")
                with zipfile.ZipFile(dataset_name, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(extract_dir))
                print("Extraction complete.")
                
                # Clean up the zip file
                os.remove(dataset_name)
                print(f"Removed {dataset_name}.")
                
                print("Tiny ImageNet dataset has been downloaded and extracted successfully.")
            except zipfile.BadZipFile:
                print(f"Error: The file {dataset_name} is not a valid zip file.")
            except Exception as e:
                print(f"Error during extraction: {e}")
        else:
            print("Download failed. Please check your internet connection and try again.")


if __name__ == "__main__":
    main()