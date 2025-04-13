import gdown
import os

def download_model_if_not_exists(file_id, path):
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading model from: {url}")
        gdown.download(url, path, quiet=False)
