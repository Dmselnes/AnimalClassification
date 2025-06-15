import os
import time
import random
import unicodedata
from collections import defaultdict
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoModelForImageClassification, AutoImageProcessor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def normalize_str(s):
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII").lower().strip()

def write_and_print(f, text):
    print(text, end="")
    f.write(text)

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def get_transforms(image_processor):
    size = (384, 384)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    return Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

def collect_images(image_folder):
    all_images = []
    for class_name in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_name)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(".jpg"):
                all_images.append({
                    "path": os.path.join(class_path, file),
                    "label": class_name
                })
    return all_images

def predict(model_path, image_folder, max_images=None):
    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    transform = get_transforms(image_processor)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    images = collect_images(image_folder)
    if max_images is not None and max_images > 0:
        images = random.sample(images, min(max_images, len(images)))

    total_images = 0
    correct_predictions = 0
    per_class_counts = defaultdict(int)
    per_class_correct = defaultdict(int)
    y_true = []
    y_pred = []

    n_images = 0
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for item in images:
            img = load_image(item["path"])
            input_tensor = transform(img).unsqueeze(0).to(device)
            true_label = normalize_str(item["label"])
            true_id = label2id[item["label"]]

            outputs = model(input_tensor)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = normalize_str(id2label[pred_id])

            y_true.append(true_label)
            y_pred.append(pred_label)

            per_class_counts[true_label] += 1
            if pred_label == true_label:
                correct_predictions += 1
                per_class_correct[true_label] += 1

            if total_images < 100:
                print(f"Image: {os.path.basename(item['path'])}, Predicted: {id2label[pred_id]}, Actual: {item['label']}")

            n_images += 1
            if (total_images < 1000 and n_images >= 100) or (total_images >= 1000 and n_images >= 1000):
                print(f"Processed {total_images} images")
                n_images = 0

            total_images += 1

    end_time = time.time()
    inference_time = end_time - start_time
    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    throughput = total_images / inference_time if inference_time > 0 else 0

    # Filenames
    safe_model_name = os.path.basename(model_path.rstrip("/")).replace("/", "_")
    summary_filename = f"{safe_model_name}_evaluation_summary.txt"
    raw_cm_filename = f"confusion_matrix_raw_{safe_model_name}.png"
    norm_cm_filename = f"confusion_matrix_normalized_{safe_model_name}.png"
    tar_filename = f"{safe_model_name}-inference-results.tar.gz"

    # Write to file
    with open(summary_filename, "w") as f:
        write_and_print(f, "\n=== Evaluation Summary ===\n")
        write_and_print(f, f"Total Images: {total_images}\n")
        write_and_print(f, f"Correct Predictions: {correct_predictions}\n")
        write_and_print(f, f"Overall Accuracy: {accuracy:.2f}%\n")
        write_and_print(f, f"Throughput: {throughput:.2f} images/sec\n\n")

        write_and_print(f, "=== Per-Class Accuracy & Distribution ===\n")
        write_and_print(f, f"{'Class':30s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>10s} {'% of Total':>12s}\n")
        write_and_print(f, "-" * 80 + "\n")
        for label in sorted(per_class_counts.keys()):
            correct = per_class_correct[label]
            total = per_class_counts[label]
            acc = (correct / total) * 100 if total > 0 else 0
            pct = (total / total_images) * 100 if total_images > 0 else 0
            write_and_print(f, f"{label:30s} {acc:9.2f}% {correct:10d} {total:10d} {pct:11.2f}%\n")

    # Confusion Matrices
    all_labels = sorted(set(y_true + y_pred))
    cm_raw = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize='true')

    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=all_labels)
    fig_raw, ax_raw = plt.subplots(figsize=(20, 20))
    disp_raw.plot(include_values=True, xticks_rotation='vertical', ax=ax_raw, cmap='Blues')
    ax_raw.set_title("Raw Confusion Matrix")
    plt.tight_layout()
    plt.savefig(raw_cm_filename)

    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=all_labels)
    fig_norm, ax_norm = plt.subplots(figsize=(20, 20))
    disp_norm.plot(include_values=True, xticks_rotation='vertical', ax=ax_norm, cmap='Blues')
    ax_norm.set_title("Normalized Confusion Matrix (Per-Class %)")
    plt.tight_layout()
    plt.savefig(norm_cm_filename)

    # Package results
    os.system(f"tar -czf {tar_filename} {summary_filename} {raw_cm_filename} {norm_cm_filename}")
    print(f"\n🗃️  Results saved to: {tar_filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CNN image classifier and save evaluation results.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--image-folder", type=str, required=True, help="Image folder with subdirectories for each class")
    parser.add_argument("--max-images", type=int, default=None, help="Limit the number of images used for evaluation")

    args = parser.parse_args()
    predict(args.model_path, args.image_folder, max_images=args.max_images)
