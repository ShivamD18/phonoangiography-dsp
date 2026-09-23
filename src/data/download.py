import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/raw")
URL = "https://archive.physionet.org/pn3/challenge/2016/training.zip"
ZIP_PATH = DATA_DIR / "training.zip"

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not ZIP_PATH.exists():
        print("Downloading PhysioNet CinC 2016 training set (~180 MB)...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(URL, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB chunk
        
        with open(ZIP_PATH, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        print("Download complete.")
    else:
        print(f"Archive already exists at {ZIP_PATH}")
        
    print("Extracting archive (this may take a minute)...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as archive:
        archive.extractall(DATA_DIR)
    print(f"Extraction complete! Files ready inside: {DATA_DIR}")

if __name__ == "__main__":
    download_and_extract()
