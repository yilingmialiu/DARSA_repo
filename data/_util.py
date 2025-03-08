import os
import torch
import urllib.request
import zipfile
import tarfile

def check_exits(root: str, file_name: str):
    """Check if file exists in root directory"""
    if not os.path.exists(os.path.join(root, file_name)):
        raise FileNotFoundError(f"{file_name} not found in {root}")

def download(root: str, filename: str, url: str, *args):
    """Download file from the internet"""
    os.makedirs(root, exist_ok=True)
    filepath = os.path.join(root, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {url} to {filepath}")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded {filename}")
            
            # Extract the downloaded file
            if filename.endswith('.zip'):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(root)
            elif filename.endswith('.tar'):
                print(f"Extracting {filename}...")
                with tarfile.open(filepath, 'r') as tar_ref:
                    tar_ref.extractall(root)
            print(f"Successfully extracted {filename}")
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise Exception(f"Error downloading {url}: {str(e)}")
