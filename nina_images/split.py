import os
import random
import shutil
from math import floor
from pathlib import Path

# CONFIGURE THESE
input_dir = Path("structuredImages")
output_dir = Path("fullRawDataset")
train_split = 0.6
val_split = 0.15
test_split = 0.15

# Set a random seed for reproducibility
random.seed(42)

# Make sure output dirs exist
for split in ['train', 'val', 'test']:
    (output_dir / split).mkdir(parents=True, exist_ok=True)

# Go through each class folder
for class_dir in input_dir.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*"))
    if len(images) <= 3:
        print(f"Skipping '{class_dir.name}' (only {len(images)} images)")
        continue

    random.shuffle(images)

    num_images = len(images)
    num_train = floor(train_split * num_images)
    num_val = max(1, floor(val_split * num_images))
    num_test = max(1, floor(test_split * num_images))

    # Adjust in case rounding caused total > num_images
    total = num_train + num_val + num_test
    if total > num_images:
        num_train -= (total - num_images)

    train_imgs = images[:num_train]
    val_imgs = images[num_train:num_train + num_val]
    test_imgs = images[num_train + num_val:num_train + num_val + num_test]

    # Copy images to new structure
    for split, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
        split_class_dir = output_dir / split / class_dir.name
        split_class_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_imgs:
            shutil.copy(img_path, split_class_dir / img_path.name)

print("Done splitting dataset.")
