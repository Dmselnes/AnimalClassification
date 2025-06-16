import os
import shutil
import random

# Paths
source_root = "structuredImagesCombinedClasses"  # where your 60 class folders are
target_root = "smallCombinedClassesCorrect"   # where to save the reduced dataset
max_images = 1000

os.makedirs(target_root, exist_ok=True)

for class_name in os.listdir(source_root):
    class_dir = os.path.join(source_root, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    # Make target class folder
    target_class_dir = os.path.join(target_root, class_name)
    os.makedirs(target_class_dir, exist_ok=True)

    # List and shuffle images
    all_images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(all_images)

    # Copy up to 1000 images
    for image_name in all_images[:max_images]:
        src = os.path.join(class_dir, image_name)
        dst = os.path.join(target_class_dir, image_name)
        shutil.copy2(src, dst)

print("Done! Dataset reduced and copied.")
