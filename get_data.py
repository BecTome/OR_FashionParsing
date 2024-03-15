import os
import shutil
import urllib.request
import zipfile

# Define the URLs
urls = {
    "train_data": "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip",
    "val_test_data": "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip",
    "train_instance_attr": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json",
    "val_instance_attr": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json"
}

# Directory to save downloaded files
base_dir = "./data"
images_dir = os.path.join(base_dir, "images")
annotations_dir = os.path.join(base_dir, "annotations")

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Download datasets
for name, url in urls.items():
    filename = os.path.join(base_dir, os.path.basename(url))
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, filename)

# Extract zip files
for zip_file in os.listdir(base_dir):
    if zip_file.endswith(".zip"):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(os.path.join(base_dir, zip_file), 'r') as zip_ref:
            zip_ref.extractall(images_dir)

# Remove zip files
for zip_file in os.listdir(base_dir):
    if zip_file.endswith(".zip"):
        os.remove(os.path.join(base_dir, zip_file))

# Move instance attribute files to annotations directory
for name in ["instances_attributes_train2020.json", "instances_attributes_val2020.json"]:
    src = os.path.join(base_dir, name)
    dst = os.path.join(annotations_dir, name)
    shutil.move(src, dst)

# Change train and test directories name to train2020 and val2020
try:
    os.rename(os.path.join(images_dir, "train"), os.path.join(images_dir, "train2020"))
    os.rename(os.path.join(images_dir, "test"), os.path.join(images_dir, "val2020"))
except FileNotFoundError:
    pass

print("Datasets downloaded and extracted successfully.")
