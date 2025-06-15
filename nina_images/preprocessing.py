from pathlib import Path
from PIL import Image
import shutil
import random
from math import floor
from tqdm import tqdm

# CONFIG
ROOT_DIR = Path("structuredImagesCombinedClasses")   # Full dataset
TARGET_SIZE = (384, 384)
TEST_SPLIT_RATIO = 0.15
TEST_OUTPUT_DIR = Path("testsetCombinedClasses")
SMALLSPLIT_DIR = Path("../smallsplit")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Set seed for reproducibility
random.seed(42)

# Load all used filenames in smallsplit
print("🔎 Indexing 'smallsplit' files...")
used_filenames = set()
for split in ["train", "val"]:
    split_path = SMALLSPLIT_DIR / split
    if split_path.exists():
        for path in split_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                used_filenames.add(path.name)

print(f"Found {len(used_filenames)} images already used in 'smallsplit'.")

# Norwegian to English category mapping
norwegian_to_english = {
    "Rev": "fox",
    "Rådyr": "roe deer",
    "Grevling": "mustelid",
    "Elg": "capreolinae",
    "Hare": "lepus",
    "Hjort": "deer",
    "Fugl": "bird",
    "Ekorn": "rodent",
    "Sau": "farm animal",
    "Gaupe": "feline",
    "Mår": "mustelid",
    "Katt": "feline",
    "Ulv": "wolf",
    "Smågnager": "rodent",
    "Svarttrost": "bird",
    "Storfe": "farm animal",
    "Storfugl": "bird",
    "Skogshøns": "bird",
    "Villsvin": "boar",
    "Ringdue": "bird",
    "Rugde": "bird",
    "Jerv": "mustelid",
    "Kjøttmeis": "bird",
    "Skjære": "bird",
    "Måltrost": "bird",
    "Trost sp.": "bird",
    "Bjørn": "bear",
    "Sørhare": "lepus",
    "Meis sp.": "bird",
    "Jerpe": "bird",
    "Nøtteskrike": "bird",
    "Ravn": "bird",
    "Rødvingetrost": "bird",
    "Gråtrost": "bird",
    "Orrfugl": "bird",
    "Annet mårdyr": "mustelid",
    "Trane": "bird",
    "Blåmeis": "bird",
    "Svartspett": "bird",
    "Duetrost": "bird",
    "Bokfink": "bird",
    "Dompap": "bird",
    "Kråke": "bird",
    "Svarthvit fluesnapper": "bird",
    "Røyskatt": "mustelid",
    "Ilder": "mustelid",
    "Flaggspett": "bird",
    "Rødstrupe": "bird",
    "Dåhjort": "deer",
    "Kattugle": "bird",
    "Musvåk": "bird",
    "Rovfugl": "bird",
    "Mink": "mustelid",
    "Spettmeis": "bird",
    "Snømus": "mustelid",
    "Nøttekråke": "bird",
    "Lavskrike": "bird",
    "Kanadagås": "bird",
    "Gråhegre": "bird",
    "Vandrefalk": "bird",
    "Lappugle": "bird",
    "Grønnspett": "bird",
    "Grønnfink": "bird",
    "Gråfluesnapper": "bird",
    "Annet pattedyr": "unknown"
}

def resize_if_needed(image_path, target_size):
    try:
        with Image.open(image_path) as img:
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img.save(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Step 1: Merge folders into English categories
print("🗂️ Grouping folders into English categories...")
unmapped_classes = []

for class_dir in ROOT_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    category_name = class_dir.name.encode('latin1').decode('utf-8')  # Fix encoding
    english_group = norwegian_to_english.get(category_name)

    if not english_group or english_group == "unknown":
        print(f"❌ Skipping and removing unmapped class: '{category_name}' ({class_dir.name})")
        shutil.rmtree(class_dir)
        unmapped_classes.append(category_name)
        continue

    target_dir = ROOT_DIR / english_group
    target_dir.mkdir(exist_ok=True)

    for image_file in class_dir.iterdir():
        if image_file.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.move(str(image_file), target_dir / image_file.name)

    shutil.rmtree(class_dir)

print(f"\n✅ Grouping complete. {len(unmapped_classes)} classes removed.\n")

# Step 2: Resize images
print("🔧 Resizing images...")
all_image_paths = [p for p in ROOT_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

for image_path in tqdm(all_image_paths, desc="Resizing"):
    resize_if_needed(image_path, TARGET_SIZE)

print("✅ Done resizing.\n")

# Step 3: Split test set (excluding smallsplit)
print("✂️ Splitting out test set (excluding overlapping images)...")
for class_dir in ROOT_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    image_files = [f for f in class_dir.iterdir()
                   if f.suffix.lower() in IMAGE_EXTENSIONS and f.name not in used_filenames]

    image_files = sorted(image_files)  # sort for deterministic sampling
    num_test = max(1, floor(len(image_files) * TEST_SPLIT_RATIO))
    test_images = random.sample(image_files, num_test)

    target_class_dir = TEST_OUTPUT_DIR / class_dir.name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in test_images:
        shutil.move(str(img_path), target_class_dir / img_path.name)

    print(f"Moved {num_test} test images from '{class_dir.name}' (out of {len(image_files)} eligible)")

print("✅ Test set split complete.")