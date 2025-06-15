import os
import random
import shutil
from math import floor
from pathlib import Path

# CONFIGURE THESE
input_dir = Path("structuredImagesCombinedClasses")
output_dir = Path("structuredImagesCombinedClassesSplit")
train_split = 0.8
val_split = 0.2  # Adjusted to add up to 1.0 with train

# Set a random seed for reproducibility
random.seed(42)

# Make sure output dirs exist
for split in ['train', 'val']:
    (output_dir / split).mkdir(parents=True, exist_ok=True)

# Go through each class folder
for class_dir in input_dir.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*"))
    if len(images) <= 2:
        print(f"Skipping '{class_dir.name}' (only {len(images)} images)")
        continue

    random.shuffle(images)

    num_images = len(images)
    num_train = floor(train_split * num_images)
    num_val = num_images - num_train  # Ensure all are used

    train_imgs = images[:num_train]
    val_imgs = images[num_train:]

    # Copy images to new structure
    for split, split_imgs in zip(['train', 'val'], [train_imgs, val_imgs]):
        split_class_dir = output_dir / split / class_dir.name
        split_class_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_imgs:
            shutil.copy(img_path, split_class_dir / img_path.name)

print("✅ Done splitting into train and val.")