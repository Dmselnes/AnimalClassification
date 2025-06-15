from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
from safetensors.torch import load_file
import torch
import time
import argparse
import unicodedata
# from fvcore.nn import FlopCountAnalysis, parameter_count_table  # Uncomment if using FLOPs

def normalize_str(s):
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII").lower()

# Argument parsing
parser = argparse.ArgumentParser(description="Run inference on ViT model with optional image limit.")
parser.add_argument("--max-images", type=int, default=100, help="Maximum number of images to process. Use -1 for no limit.")
args = parser.parse_args()
max_images = args.max_images

# Set device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
image_folder = 'nina_images/testset'
model_path = 'test-smallsplit/vit-base-patch16-224-in21k_lr3em04_seed42_ep30_bs64/model.safetensors'

# Load and prepare model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", from_tf=False, config=None)
num_finetuned_classes = 41
model.classifier = torch.nn.Linear(model.config.hidden_size, num_finetuned_classes)

# Define the class labels
class_labels = ['Annet mÃ¥rdyr', 'BjÃ¸rn', 'BlÃ¥meis', 'Bokfink', 'Duetrost', 'Ekorn', 'Elg', 'Fugl', 'Gaupe', 'GrÃ¥trost', 'Grevling', 'Hare', 'Hjort', 'Jerpe', 'Jerv', 'Katt', 'KjÃ¸ttmeis', 'MÃ¥ltrost', 'MÃ¥r', 'Meis sp.', 'NÃ¸tteskrike', 'Orrfugl', 'RÃ¥dyr', 'RÃ¸dvingetrost', 'Ravn', 'Rev', 'Ringdue', 'Rugde', 'SÃ¸rhare', 'Sau', 'SkjÃ¦re', 'SkogshÃ¸ns', 'SmÃ¥gnager', 'Storfe', 'Storfugl', 'Svartspett', 'Svarttrost', 'Trane', 'Trost sp.', 'Ulv', 'Villsvin']
model.config.id2label = {i: label for i, label in enumerate(class_labels)}
model.config.label2id = {label: i for i, label in enumerate(class_labels)}

# Load weights
model_weights = load_file(model_path)
model.load_state_dict(model_weights)
model.to(device)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())

# Processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Dataset
subfolders = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

# Metrics
total_images = 0
n_images = 0
correct_predictions = 0
start_time = time.time()

# Inference loop
for subfolder in subfolders:
    subfolder_path = os.path.join(image_folder, subfolder)
    image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_file in image_files:
        if max_images != -1 and total_images >= max_images:
            break

        image_path = os.path.join(subfolder_path, image_file)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        if total_images < 100:
            print(f"Image: {image_file}, Predicted: {predicted_label}, Actual: {subfolder}")

        # Progress reporting
        if total_images < 1000 and n_images > 100:
            print(f"Processed {total_images} images")
            n_images = 0
        if total_images > 1000 and n_images > 1000:
            print(f"Processed {total_images} images")
            n_images = 0

        total_images += 1
        n_images += 1

        if normalize_str(predicted_label) == normalize_str(subfolder):
            correct_predictions += 1

# Metrics summary
end_time = time.time()
inference_time = end_time - start_time
throughput = total_images / inference_time if inference_time > 0 else 0
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0

# Print results
print("\n=== Evaluation Summary ===", flush=True)
print(f"Total Images: {total_images}", flush=True)
print(f"Correct Predictions: {correct_predictions}", flush=True)
print(f"Accuracy: {accuracy:.2f}%", flush=True)
print(f"Throughput: {throughput:.2f} images/sec", flush=True)
print(f"Total Parameters: {total_params:,}", flush=True)
# print(f"FLOPs (approx): {flops.total():,.0f}")  # Uncomment if using FLOPs
