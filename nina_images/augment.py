import os
import random
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# CONFIG
SOURCE_DIR = "smallsplit/train"       # Original dataset (class subfolders)
TARGET_DIR = "smallsplit/trainaug"    # Output augmented dataset
IMAGE_SIZE = (384, 384)
MAX_PER_CLASS = 1000
MAX_AUG_PER_IMAGE = 10

# Data augmentation pipeline
def augment_image(image):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[int(IMAGE_SIZE[0] * 0.9), int(IMAGE_SIZE[1] * 0.9), 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0, 255)

# Create output structure
os.makedirs(TARGET_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(images)
    num_needed = MAX_PER_CLASS - num_images

    target_class_dir = os.path.join(TARGET_DIR, class_name)
    os.makedirs(target_class_dir, exist_ok=True)

    # Copy original images
    for img_name in images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(target_class_dir, img_name)
        if not os.path.exists(dst):
            tf.keras.utils.save_img(dst, tf.keras.utils.load_img(src, target_size=IMAGE_SIZE))

    if num_needed <= 0:
        continue

    print(f"Augmenting {class_name} ({num_images} → {MAX_PER_CLASS})")

    # Track number of augmentations per image
    aug_count = defaultdict(int)
    total_augmented = 0

    with tqdm(total=num_needed) as pbar:
        while total_augmented < num_needed:
            img_name = random.choice(images)

            # Skip if this image has reached max augmentation
            if aug_count[img_name] >= MAX_AUG_PER_IMAGE:
                if all(aug_count[img] >= MAX_AUG_PER_IMAGE for img in images):
                    print(f"Reached augmentation cap for all images in class '{class_name}'")
                    break
                continue

            img_path = os.path.join(class_path, img_name)
            try:
                img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(img_path))
                img = augment_image(img)
                img = tf.cast(img, tf.uint8).numpy()
                out_path = os.path.join(target_class_dir, f"aug_{aug_count[img_name]}_{img_name}")
                Image.fromarray(img).save(out_path)
                aug_count[img_name] += 1
                total_augmented += 1
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

