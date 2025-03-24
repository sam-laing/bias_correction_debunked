import os
import requests
from tqdm import tqdm
import gzip
import shutil
import hashlib

def main():
    # URLs for the MNIST dataset files
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "training_images": "train-images-idx3-ubyte.gz",
        "training_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    # MD5 checksums for verification
    checksums = {
        "training_images": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "training_labels": "d53e105ee54ea40749a09fcbcd1e9432",
        "test_images": "9fb629c4189551a2d022fa330f9573f3",
        "test_labels": "ec29112dd5afa0611ce80d1b7f02629c"
    }
    
    # Directory to store the dataset
    extract_dir = "/fast/slaing/data/vision/mnist"
    
    # Check if dataset directory already exists
    if os.path.exists(extract_dir) and os.path.isdir(extract_dir) and len(os.listdir(extract_dir)) == 4:
        print(f"Dataset directory {extract_dir} already exists and seems complete. Skipping download.")
        return
    
    # Create the directory to store the dataset
    os.makedirs(extract_dir, exist_ok=True)
    
    # Function to download a file with progress bar
    def download_file(url, filename, md5=None):
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
            
            # Verify MD5 checksum if provided
            if md5:
                file_md5 = hashlib.md5(open(filename, 'rb').read()).hexdigest()
                if file_md5 != md5:
                    print(f"Warning: MD5 checksum mismatch for {filename}.")
                    print(f"Expected: {md5}")
                    print(f"Got: {file_md5}")
                    return False
                
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return False
    
    # Download and extract each file
    for file_key, file_name in files.items():
        file_url = base_url + file_name
        compressed_file = os.path.join(extract_dir, file_name)
        extracted_file = os.path.join(extract_dir, file_name[:-3])  # Remove .gz
        
        # Skip if the extracted file already exists
        if os.path.exists(extracted_file):
            print(f"File {extracted_file} already exists. Skipping.")
            continue
            
        # Download the compressed file
        download_success = download_file(file_url, compressed_file, checksums.get(file_key))
        
        if download_success:
            try:
                # Extract the gzip file
                print(f"Extracting {file_name}...")
                with gzip.open(compressed_file, 'rb') as f_in:
                    with open(extracted_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Extracted {file_name} to {extracted_file}")
                
                # Clean up compressed file
                os.remove(compressed_file)
                print(f"Removed {file_name}")
                
            except Exception as e:
                print(f"Error during extraction of {file_name}: {e}")
        else:
            print(f"Download failed for {file_name}. Please check your internet connection and try again.")
    
    print("MNIST dataset has been downloaded and extracted successfully.")

if __name__ == "__main__":
    main()