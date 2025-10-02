# ============================================================================
# CelebA API
# ============================================================================

# TODO: Implement the CelebA API function
# datasets: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

import os
import zipfile
import json

def ensure_kaggle_env():
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_username.strip() or not kaggle_key or not kaggle_key.strip():
        print("Kaggle API credentials not set. Please provide the following:")
        kaggle_username = input("Enter your Kaggle Username: ")
        kaggle_key = input("Enter your Kaggle API Key: ")

        with open(".env", "w") as f:
            f.write(f"KAGGLE_USERNAME={kaggle_username}\n")
            f.write(f"KAGGLE_KEY={kaggle_key}\n")

        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

        kaggle_config_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_config_dir, exist_ok=True)
        with open(os.path.join(kaggle_config_dir, "kaggle.json"), "w") as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)
        os.chmod(os.path.join(kaggle_config_dir, "kaggle.json"), 0o600)
    else:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

def celeba_api_function(dest_dir="../../datasets"):
    ensure_kaggle_env()

    from kaggle.api.kaggle_api_extended import KaggleApi

    os.makedirs(dest_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset = "jessicali9530/celeba-dataset"
    files_to_download = [
        "img_align_celeba.zip",
        "list_attr_celeba.csv",
        "list_eval_partition.csv",
        "identity_CelebA.txt"
    ]

    print(f"Downloading dataset files from Kaggle: {dataset}")
    for file in files_to_download:
        api.dataset_download_files(dataset, path=dest_dir, unzip=True)
        zip_path = os.path.join(dest_dir, file)
        if zipfile.is_zipfile(zip_path):
            print(f"Unzipping {file}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            os.remove(zip_path)
    print("Download and extraction complete.")