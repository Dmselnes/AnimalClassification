import os
from pathlib import Path

# CONFIG
full_test_dir = Path("testsetCombinedClasses")  # Path to the full dataset's test set
check_dirs = [                            # Paths to check for overlaps
    Path("smallsplit"),
    Path("structuredImagesCombinedClasses")
]

# Collect all test filenames (without paths)
test_filenames = set()
for class_folder in full_test_dir.iterdir():
    if class_folder.is_dir():
        for img_file in class_folder.iterdir():
            test_filenames.add(img_file.name)

# Check for overlaps in train and val folders
overlaps = []

for check_dir in check_dirs:
    for class_folder in check_dir.iterdir():
        if class_folder.is_dir():
            for img_file in class_folder.iterdir():
                if img_file.name in test_filenames:
                    overlaps.append(img_file)

# Output
if overlaps:
    print(f"⚠️ Found {len(overlaps)} overlapping images:")
    for overlap in overlaps[:10]:  # show a sample
        print(overlap)
    if len(overlaps) > 10:
        print("...and more.")
else:
    print("✅ No overlapping images found!")