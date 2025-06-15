import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# 🔧 Customize this with your CSV paths
csv_files = {
    # "ViT": "../../Experiments/vit_results/google_vit-base-patch16-384_evaluation_summary.txt",
    # "Swin": "../../Experiments/swin_results/microsoft_swin-base-patch4-window12-384_evaluation_summary.txt",
    # "DeiT": "../../Experiments/deit_results/facebook_deit-base-patch16-384_evaluation_summary.txt",
    # "resnet-152 small": "../../Experiments/resnet-152_lr1em04_seed42_ep30_bs8-inference-results/resnet-152_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # "resnet-50 small": "../../Experiments/resnet-50_lr1em04_seed42_ep30_bs8-inference-results/resnet-50_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # "resnet-18 small": "../../Experiments/resnet-18_lr1em04_seed42_ep30_bs8-inference-results/resnet-18_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # "cvt small": "../../Experiments/cvt-13-384_lr1em04_seed42_ep30_bs8-inference-results/cvt-13-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # "vit": "../../Experiments/vit_results/google_vit-base-patch16-384_evaluation_summary.txt",
    # "swin": "../../Experiments/swin_results/microsoft_swin-base-patch4-window12-384_evaluation_summary.txt",
    # "deit": "../../Experiments/deit_results/facebook_deit-base-patch16-384_evaluation_summary.txt",
    # #"resnet-152 small": "../../Experiments/resnet-152_lr1em04_seed42_ep30_bs8-inference-results/resnet-152_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # #"resnet-50 small": "../../Experiments/resnet-50_lr1em04_seed42_ep30_bs8-inference-results/resnet-50_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # #"resnet-18 small": "../../Experiments/resnet-18_lr1em04_seed42_ep30_bs8-inference-results/resnet-18_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # "resnet-152": "../../Experiments/resnet-152_lr1em04_seed42_ep50_bs8-inference-results/resnet-152_lr1em04_seed42_ep50_bs8_evaluation_summary.txt",
    # "resnet-50": "../../Experiments/resnet-50_lr1em04_seed42_ep50_bs8-inference-results/resnet-50_lr1em04_seed42_ep50_bs8_evaluation_summary.txt",
    # "resnet-18": "../../Experiments/resnet-18_lr1em04_seed42_ep50_bs8-inference-results/resnet-18_lr1em04_seed42_ep50_bs8_evaluation_summary.txt",
    # #"cvt small": "../../Experiments/cvt-13-384_lr1em04_seed42_ep30_bs8-inference-results/cvt-13-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # "cvt": "../../Experiments/cvt-13-384_lr1em04_seed42_ep50_bs8-inference-results/cvt-13-384_lr1em04_seed42_ep50_bs8_evaluation_summary.txt",
    # #"convnext small": "../../Experiments/convnext-base-384_lr1em04_seed42_ep30_bs8-inference-results/convnext-base-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # "convnext": "../../Experiments/convnext-base-384_lr1em04_seed42_ep50_bs8-inference-results/convnext-base-384_lr1em04_seed42_ep50_bs8_evaluation_summary.txt",
    # #"beit small": "../../Experiments/beit-base-patch16-384_lr1em04_seed42_ep30_bs8-inference-results/beit-base-patch16-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    # "beit": "../../Experiments/beit-base-patch16-384_lr1em04_seed42_ep50_bs8-inference-results/beit-base-patch16-384_lr1em04_seed42_ep50_bs8_evaluation_summary.txt",
    "vit small": "../../Experiments/inference-result-small/vit-base-patch16-384_lr1em04_seed42_ep30_bs8-inference-results/vit-base-patch16-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    #"vit small imbalanced": "../../Experiments/inference-result-all-small/vit-base-patch16-384_lr1em04_seed42_ep30_bs8-inference-results/vit-base-patch16-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    #"vit small augmented": "../../Experiments/inference-result-all-small-balanced/vit-base-patch16-384_lr1em04_seed42_ep30_bs8-inference-results/vit-base-patch16-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    "deit small": "../../Experiments/inference-result-small/deit-base-patch16-384_lr1em04_seed42_ep30_bs8-inference-results/deit-base-patch16-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    #"deit small imbalanced&augmented": "../../Experiments/inference-result-all-small/deit-base-patch16-384_lr1em04_seed42_ep30_bs8-inference-results/deit-base-patch16-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    #"deit small augmented": "../../Experiments/inference-result-all-small-balanced/deit-base-patch16-384_lr1em04_seed42_ep30_bs8-inference-results/deit-base-patch16-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    "swin small": "../../Experiments/inference-result-small/swin-base-patch4-window12-384_lr1em04_seed42_ep30_bs8-inference-results/swin-base-patch4-window12-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    #"swin small imbalanced&augmented": "../../Experiments/inference-result-all-small/swin-base-patch4-window12-384_lr1em04_seed42_ep30_bs8-inference-results/swin-base-patch4-window12-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",
    #"swin small augmented": "../../Experiments/inference-result-all-small-balanced/swin-base-patch4-window12-384_lr1em04_seed42_ep30_bs8-inference-results/swin-base-patch4-window12-384_lr1em04_seed42_ep30_bs8_evaluation_summary.txt",

}

records = []

for model_name, filepath in csv_files.items():
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find where the per-class section starts
    start = None
    for i, line in enumerate(lines):
        if "=== Per-Class Accuracy" in line:
            start = i
            break
    if start is None:
        print(f"Skipping {filepath} — section not found.")
        continue

    # Read table lines after header
    header_line_index = start + 2
    data_lines = lines[header_line_index + 1:]

    for line in data_lines:
        if not line.strip() or line.startswith("---"):
            continue

        # Use regex to split columns by multiple spaces
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) < 5:
            continue

        class_name = parts[0]
        accuracy_str = parts[1]
        try:
            accuracy = float(accuracy_str.strip('%')) / 100.0
        except ValueError:
            continue

        records.append({
            "model": model_name,
            "class": class_name,
            "accuracy": accuracy
        })

# Create DataFrame
df = pd.DataFrame(records)

# Compute class-wise (mean per-class) accuracy per model
classwise_accuracy = df.groupby("model")["accuracy"].mean()

print("\n📈 Class-wise Accuracy (mean of per-class accuracies):")
for model, acc in classwise_accuracy.items():
    print(f"  {model:20}: {acc:.2%}")


# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="class", y="accuracy", hue="model")
plt.title("Per-class Accuracy per Model")
plt.ylabel("Accuracy")
plt.xlabel("Class")
plt.ylim(0.4, 1.01)
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.tight_layout()
# plt.savefig("classwise_accuracy_comparison.pdf", bbox_inches="tight")
plt.show()
