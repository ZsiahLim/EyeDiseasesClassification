import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Paths
data_path = "/Users/henrilix/PycharmProjects/EyeDiseaseDetection/ODIR-5K/full_df.csv"
image_root_dir = "/Users/henrilix/PycharmProjects/EyeDiseaseDetection/ODIR-5K"
output_dir = "/Users/henrilix/PycharmProjects/EyeDiseaseDetection/processed_ODIR"

# Load the dataset labels
data = pd.read_csv(data_path)

# Map labels to class names
class_mapping = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other Diseases"
}

# Create output directories
os.makedirs(output_dir, exist_ok=True)
for subset in ["train", "val", "test"]:
    for class_name in class_mapping.values():
        os.makedirs(os.path.join(output_dir, subset, class_name), exist_ok=True)

# Map multi-label targets to a dominant class (the first class in order of importance)
def get_dominant_label(row):
    for label, class_name in class_mapping.items():
        if row[label] == 1:  # Assuming binary indicator columns for each disease
            return class_name
    return None

# Assign dominant labels to the dataset
data["Dominant_Label"] = data.apply(get_dominant_label, axis=1)

# Filter out rows with no dominant label
data = data[data["Dominant_Label"].notnull()]

# Split the dataset into train, validation, and test sets
train_val, test = train_test_split(data, test_size=0.2, stratify=data["Dominant_Label"], random_state=42)
train, val = train_test_split(train_val, test_size=0.2, stratify=train_val["Dominant_Label"], random_state=42)

# Function to copy images to appropriate directories
def copy_images(df, subset):
    for _, row in df.iterrows():
        label = row["Dominant_Label"]
        source_path = os.path.join(image_root_dir, row["filename"])  # Assuming "filename" column has the image names
        dest_dir = os.path.join(output_dir, subset, label)
        if os.path.exists(source_path):  # Ensure the file exists before copying
            shutil.copy(source_path, dest_dir)

# Copy images for train, validation, and test sets
print("Processing training images...")
copy_images(train, "train")
print("Processing validation images...")
copy_images(val, "val")
print("Processing test images...")
copy_images(test, "test")

print("Dataset split and processed successfully!")
