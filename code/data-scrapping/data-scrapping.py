import kaggle
import os

# Set the environment variable to point to your kaggle.json file
os.environ['KAGGLE_CONFIG_DIR'] = r'F:\AIBAS\Project AIBAS\.kaggle'

# Authenticate using kaggle.json
kaggle.api.authenticate()

# Download the dataset
dataset = "alexteboul/diabetes-health-indicators-dataset"
output_dir = './data/dataset'
os.makedirs(output_dir, exist_ok=True)
kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True)

print(f"Dataset downloaded and saved to {output_dir}")
