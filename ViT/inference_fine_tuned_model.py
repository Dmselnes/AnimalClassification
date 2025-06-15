from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
from safetensors.torch import load_file
import torch

image_folder = 'LessImagesTest/test'
model_path = 'test/vit-base-patch16-224-in21k_lr5em05_seed42_ep50_bs64/model.safetensors'# Load the model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", from_tf=False, config=None)
num_finetuned_classes = 15
model.classifier = torch.nn.Linear(model.config.hidden_size, num_finetuned_classes)

# Define the class labels for the fine-tuned model
class_labels = ['Annet mÃ¥rdyr', 'BjÃ¸rn', 'BlÃ¥meis', 'Bokfink', 'Duetrost', 'Ekorn', 'Elg', 'Fugl', 'Gaupe', 'Grevling', 'GrÃ¥trost', 'Hare', 'Hjort', 'Jerpe', 'Jerv', 'Katt', 'KjÃ¸ttmeis', 'Meis sp.', 'MÃ¥ltrost', 'MÃ¥r', 'NÃ¸tteskrike', 'Orrfugl', 'Ravn', 'Rev', 'Ringdue', 'Rugde', 'RÃ¥dyr', 'RÃ¸dvingetrost', 'Sau', 'SkjÃ¦re', 'SkogshÃ¸ns', 'SmÃ¥gnager', 'Storfe', 'Storfugl', 'Svartspett', 'Svarttrost', 'SÃ¸rhare', 'Trane', 'Trost sp.', 'Ulv', 'Villsvin']

# Assign the labels to the model
model.config.id2label = {i: label for i, label in enumerate(class_labels)}
model.config.label2id = {label: i for i, label in enumerate(class_labels)}

# Load the model weights from the safetensors file
model_weights = load_file(model_path)
model.load_state_dict(model_weights)

# Get the list of all subfolders (representing the classes) inside the `test` folder
subfolders = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Counters for accuracy
total_images = 0
correct_predictions = 0

# Iterate over each class subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(image_folder, subfolder)
    
    # Get a list of all image files in the subfolder (class)
    image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # Iterate over each image file in the subfolder (class)
    for image_file in image_files:
        image_path = os.path.join(subfolder_path, image_file)
        
        # Open the image from the local file
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get the predicted class index
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
        
        # Print the predicted class and the actual class label (subfolder)
        print(f"Image: {image_file}, Predicted class: {predicted_label}, Actual label: {subfolder}")
        
        total_images += 1
        if predicted_label == subfolder:
            correct_predictions += 1

# Calculate and print accuracy
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"\nTotal images: {total_images}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
