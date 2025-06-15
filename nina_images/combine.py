import os
import shutil
from tqdm import tqdm

# Set your two source folders and the target folderO
src1 = "Images"
src2 = "images_part_2"
dst = "allImages"

# Create the destination folder if it doesn't exist
os.makedirs(dst, exist_ok=True)

# Helper function to copy all files
def copy_images(src_folder, dst_folder):
    for filename in tqdm(os.listdir(src_folder), desc=f"Copying from {src_folder}"):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        # Optionally: handle duplicate filenames
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(dst_folder, f"{base}_{counter}{ext}")
                counter += 1

        shutil.copy2(src_path, dst_path)

# Copy both folders
copy_images(src1, dst)
copy_images(src2, dst)

print("✅ Done combining the folders.")
