import os
import requests
from tqdm import tqdm
import tarfile
import shutil

def main():
    # URL for the CUB-200-2011 dataset
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

    dataset_name = "CUB_200_2011.tgz"
    extract_dir = "/fast/slaing/data/vision/CUB_200_2011"

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
                with tarfile.open(dataset_name, 'r:gz') as tar_ref:
                    tar_ref.extractall(os.path.dirname(extract_dir))
                print("Extraction complete.")
                
                # Clean up the tar file
                os.remove(dataset_name)
                print(f"Removed {dataset_name}.")
                
                print("CUB-200-2011 dataset has been downloaded and extracted successfully.")
            except tarfile.TarError:
                print(f"Error: The file {dataset_name} is not a valid tar file.")
            except Exception as e:
                print(f"Error during extraction: {e}")
        else:
            print("Download failed. Please check your internet connection and try again.")


if __name__ == "__main__":
    main()