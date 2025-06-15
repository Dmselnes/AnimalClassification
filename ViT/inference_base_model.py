from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os

# Path to the local folder containing images
image_folder = '../RandomImages'

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Iterate over each image file in the folder
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # Open the image from the local file
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print(f"Predicted class for {image_file}: {model.config.id2label[predicted_class_idx]}")
