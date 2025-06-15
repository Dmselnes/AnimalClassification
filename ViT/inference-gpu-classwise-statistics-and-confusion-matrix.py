from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
from safetensors.torch import load_file
import torch
import time
import argparse
import unicodedata
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def normalize_str(s):
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII").lower().strip()

# Argument parsing
parser = argparse.ArgumentParser(description="Run inference with an image classification model and report per-class accuracy.")
parser.add_argument("--model-path", required=True, help="Path to the trained model in .safetensors format")
parser.add_argument("--model-name", required=True, help="Model name for loading processor and base config (e.g., google/vit-base-patch16-224)")
parser.add_argument("--image-folder", required=True, help="Path to the test image directory with subfolders per class")
parser.add_argument("--max-images", type=int, default=-1, help="Maximum number of images to process. Use -1 for no limit.")
args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load base model and override classifier
model = ViTForImageClassification.from_pretrained(args.model_name)
num_finetuned_classes = 41
model.classifier = torch.nn.Linear(model.config.hidden_size, num_finetuned_classes)

# Set labels (adjust if needed)
class_labels = ['Annet mÃ¥rdyr', 'BjÃ¸rn', 'BlÃ¥meis', 'Bokfink', 'Duetrost', 'Ekorn', 'Elg', 'Fugl', 'Gaupe', 'GrÃ¥trost', 'Grevling', 'Hare', 'Hjort', 'Jerpe', 'Jerv', 'Katt', 'KjÃ¸ttmeis', 'MÃ¥ltrost', 'MÃ¥r', 'Meis sp.', 'NÃ¸tteskrike', 'Orrfugl', 'RÃ¥dyr', 'RÃ¸dvingetrost', 'Ravn', 'Rev', 'Ringdue', 'Rugde', 'SÃ¸rhare', 'Sau', 'SkjÃ¦re', 'SkogshÃ¸ns', 'SmÃ¥gnager', 'Storfe', 'Storfugl', 'Svartspett', 'Svarttrost', 'Trane', 'Trost sp.', 'Ulv', 'Villsvin']

model.config.id2label = {i: label for i, label in enumerate(class_labels)}
model.config.label2id = {label: i for i, label in enumerate(class_labels)}

# Load trained weights
print(f"Loading model weights from {args.model_path}")
model_weights = load_file(args.model_path + "/model.safetensors")
model.load_state_dict(model_weights)
model.to(device)
model.eval()

# Processor
processor = ViTImageProcessor.from_pretrained(args.model_name)

# Metrics
total_images = 0
correct_predictions = 0
per_class_counts = defaultdict(int)
per_class_correct = defaultdict(int)

start_time = time.time()
n_images = 0

# Confusion matrix  
y_true = []
y_pred = []

# Walk through the image folder
subfolders = [f for f in os.listdir(args.image_folder) if os.path.isdir(os.path.join(args.image_folder, f))]

for subfolder in subfolders:
    subfolder_path = os.path.join(args.image_folder, subfolder)
    image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_file in image_files:
        if args.max_images != -1 and total_images >= args.max_images:
            break

        image_path = os.path.join(subfolder_path, image_file)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_idx]

        norm_actual = normalize_str(subfolder)
        norm_pred = normalize_str(predicted_label)
        # Confusion matrix
        y_true.append(norm_actual)
        y_pred.append(norm_pred)

        per_class_counts[norm_actual] += 1
        if norm_pred == norm_actual:
            correct_predictions += 1
            per_class_correct[norm_actual] += 1

        if total_images < 100:
            print(f"Image: {image_file}, Predicted: {predicted_label}, Actual: {subfolder}")

        n_images += 1
        if (total_images < 1000 and n_images >= 100) or (total_images >= 1000 and n_images >= 1000):
            print(f"Processed {total_images} images")
            n_images = 0

        total_images += 1

# Results
end_time = time.time()
inference_time = end_time - start_time
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
throughput = total_images / inference_time if inference_time > 0 else 0

print("\n=== Evaluation Summary ===")
print(f"Total Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Overall Accuracy: {accuracy:.2f}%")
print(f"Throughput: {throughput:.2f} images/sec")

print("\n=== Per-Class Accuracy & Distribution ===")
print(f"{'Class':30s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>10s} {'% of Total':>12s}")
print("-" * 80)
for label in sorted(per_class_counts.keys()):
    correct = per_class_correct[label]
    total = per_class_counts[label]
    acc = (correct / total) * 100 if total > 0 else 0
    pct = (total / total_images) * 100 if total_images > 0 else 0
    print(f"{label:30s} {acc:9.2f}% {correct:10d} {total:10d} {pct:11.2f}%")

print("\n=== Confusion Matrix: Raw Counts ===")
cm_raw = confusion_matrix(y_true, y_pred, labels=all_labels)
disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=all_labels)

fig_raw, ax_raw = plt.subplots(figsize=(20, 20))
disp_raw.plot(include_values=True, xticks_rotation='vertical', ax=ax_raw, cmap='Blues')
ax_raw.set_title("Raw Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_raw.png")
#plt.show()

print("\n=== Confusion Matrix: Normalized ===")
cm_norm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=all_labels)

fig_norm, ax_norm = plt.subplots(figsize=(20, 20))
disp_norm.plot(include_values=True, xticks_rotation='vertical', ax=ax_norm, cmap='Blues')
ax_norm.set_title("Normalized Confusion Matrix (Per-Class %)")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png")
#plt.show()
