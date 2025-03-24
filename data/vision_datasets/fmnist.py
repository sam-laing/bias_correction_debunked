import os
import requests
from tqdm import tqdm
import gzip
import shutil
import hashlib

def main():
    # URLs for the Fashion-MNIST dataset files
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "training_images": "train-images-idx3-ubyte.gz",
        "training_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    # MD5 checksums for verification
    checksums = {
        "training_images": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "training_labels": "25c81989df183df01b3e8a0aad5dffbe",
        "test_images": "bef4ecab320f06d8554ea6380940ec79",
        "test_labels": "bb300cfdad3c16e7a12a480ee83cd310"
    }
    
    # Directory to store the dataset
    extract_dir = "/fast/slaing/data/vision/fashion-mnist"
    
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
    
    print("Fashion-MNIST dataset has been downloaded and extracted successfully.")
    
    # Print dataset information
    print("\nFashion-MNIST Dataset Information:")
    print("----------------------------------")
    print("Fashion-MNIST is a dataset of Zalando's article images, consisting of:")
    print("- 60,000 training examples")
    print("- 10,000 test examples")
    print("- 10 fashion categories (T-shirt/top, Trouser, Pullover, etc.)")
    print("- 28x28 grayscale images")
    print("\nClasses:")
    print("0: T-shirt/top")
    print("1: Trouser")
    print("2: Pullover")
    print("3: Dress")
    print("4: Coat")
    print("5: Sandal")
    print("6: Shirt")
    print("7: Sneaker")
    print("8: Bag")
    print("9: Ankle boot")

if __name__ == "__main__":
    main()