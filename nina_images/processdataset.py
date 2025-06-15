from pathlib import Path
from PIL import Image
import shutil
import random
from math import floor
from tqdm import tqdm 

# CONFIG
ROOT_DIR = Path("structuredImages")   # Full dataset path with class folders
TARGET_SIZE = (384, 384)
MIN_IMAGES_REQUIRED = 15
TEST_SPLIT_RATIO = 0.15
TEST_OUTPUT_DIR = Path("testset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Set seed for reproducibility
random.seed(42)

def resize_if_needed(image_path, target_size):
    try:
        with Image.open(image_path) as img:
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img.save(image_path)
                # print(f"Resized: {image_path}")
            else:
                pass  # Already correct size
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# # Step 1: Resize all images
# print("🔧 Resizing images...")
# for image_path in ROOT_DIR.rglob("*"):
#     if image_path.suffix.lower() in IMAGE_EXTENSIONS:
#         resize_if_needed(image_path, TARGET_SIZE)
# print("✅ Done resizing.")

# Step 1: Resize all images
print("🔧 Resizing images...")
all_image_paths = [p for p in ROOT_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

for image_path in tqdm(all_image_paths, desc="Resizing"):
    resize_if_needed(image_path, TARGET_SIZE)

print("✅ Done resizing.")

# Step 2: Remove classes with fewer than 15 images
print("\n🧹 Removing classes with fewer than 15 images...")
removed_classes = []
for class_dir in ROOT_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    if len(image_files) < MIN_IMAGES_REQUIRED:
        shutil.rmtree(class_dir)
        removed_classes.append((class_dir.name, len(image_files)))

print(f"✅ Removed {len(removed_classes)} classes:")
for name, count in removed_classes:
    print(f"  - {name} ({count} images)")

# Step 3: Move 15% of each remaining class into test set
print("\n✂️ Splitting out test set...")
for class_dir in ROOT_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

    if len(image_files) < MIN_IMAGES_REQUIRED:
        print(f"Skipping '{class_dir.name}' (only {len(image_files)} images)")
        continue

    num_test = max(1, floor(len(image_files) * TEST_SPLIT_RATIO))
    test_images = random.sample(image_files, num_test)

    target_class_dir = TEST_OUTPUT_DIR / class_dir.name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in test_images:
        shutil.move(str(img_path), target_class_dir / img_path.name)

    print(f"Moved {num_test} test images from '{class_dir.name}'")

print("✅ Test set split complete.")