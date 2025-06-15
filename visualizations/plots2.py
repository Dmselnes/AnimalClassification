import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

# Connect to tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# 🔄 Toggle: Metric to plot
metric_to_plot = "eval_accuracy"  # "eval_accuracy" or "loss"

# ✅ Custom label mapping for datasets (used to clean up dataset names)
custom_labels = {
    "smallsplit/train": "Unbalanced",
    "smallCombinedClassesCorrectSplit/train": "Combined",
    "smallsplit/trainaug": "Augmented",
    "nina_images/structuredImagesCombinedClassesSplit/train": "Large",
    "nina_images/smallCombinedClassesCorrectSplit/train" : "Small",
    "unknown_dataset": "Unknown",
}

# ✅ Simplified dataset grouping for linestyle (Small vs Large)
dataset_group_labels = {
    "smallsplit/train": "Unbalanced",
    "smallCombinedClassesCorrectSplit/train": "Combined",
    "smallsplit/trainaug": "Augmented",
    "nina_images/structuredImagesCombinedClassesSplit/train": "Large",
    "nina_images/smallCombinedClassesCorrectSplit/train" : "Small",
    "unknown_dataset": "Unknown",
    # "smallsplit/train": "Small",
    # "smallCombinedClassesCorrectSplit/train": "Small",
    # "smallsplit/trainaug": "Small",
    # "nina_images/structuredImagesCombinedClassesSplit/train": "Large",
    # "nina_images/smallCombinedClassesCorrectSplit/train" : "Small",
    # "unknown_dataset": "Unknown"
}

# 🔵 Filter runs based on run_name
wanted_run_names = [
    # # Small not balanced:
    # "swin-base-patch4-window12-384_lr1em04_seed42_ep30_bs8", 
    # "vit-base-patch16-384_lr1em04_seed42_ep30_bs8",
    # "deit-base-patch16-384_lr1em04_seed42_ep30_bs8",

    # # Small combined classes:
    # "unleashed-eel-979",
    # "bedecked-snipe-683",
    # "funny-fox-259",

    # # Small augmented:
    # "smiling-whale-784",
    # "persistent-deer-342",
    # "polite-snake-306",

    "kindly-asp-746", #swin
    # "unleashed-eel-979",

    "colorful-turtle-986", #vit
    # "bedecked-snipe-683",

    "abrasive-stoat-465",  #deit
    # "funny-fox-259",

    "inquisitive-fawn-338", #beit
    # "redolent-crow-15", #beit

     "bittersweet-swan-37",  # resnet50
    # "treasured-mouse-243", #resnet50

     "adorable-cow-615",     # convnext
    # "delicate-hawk-251",  #convnext
   
    # "redolent-mule-431", #cvt
    "stylish-foal-287",  # cvt
]

# 🔵 List of experiments to search
experiment_names = [
    "Vit-Swin-and-DeIT",
    "Vit-Swin-and-DeIT-othersmall",
    "Vit-Swin-and-DeIT-other-smallbalanced",
    "Vit-Swin-and-DeIT-large",
    "cnn-and-other-models-othersmall",
    "cnn-and-other-models-large",
    "convnext-large"
]

default_epochs = 30

# Get experiment IDs
experiment_ids = []
for name in experiment_names:
    exp = mlflow.get_experiment_by_name(name)
    if exp:
        experiment_ids.append(exp.experiment_id)
    else:
        print(f"Warning: Experiment '{name}' not found!")

if not experiment_ids:
    raise ValueError("No valid experiments found.")

# Get runs
runs = mlflow.search_runs(experiment_ids=experiment_ids)
runs = runs[runs["tags.mlflow.runName"].isin(wanted_run_names)]

# Collect history
all_data = []
combo_labels = set()
model_names = set()

for _, run in runs.iterrows():
    run_id = run["run_id"]
    model_name = run.get("params.model_name", "unknown_model")
    train_dir = run.get("params.train_dir", "unknown_dataset")

    dataset_label = custom_labels.get(train_dir, train_dir)
    dataset_group = dataset_group_labels.get(train_dir, "Unknown")

    combo_label = f"{model_name} | {dataset_label}"
    combo_labels.add(combo_label)
    model_names.add(model_name)

    try:
        history = client.get_metric_history(run_id, metric_to_plot)
        steps = [m.step for m in history]
        if not steps:
            print(f"No {metric_to_plot} metrics for run {run_id}")
            continue

        try:
            run_epochs = int(run.get("params.num_train_epochs", default_epochs))
        except:
            run_epochs = default_epochs

        steps_per_epoch = max(steps) / run_epochs if run_epochs > 0 else 1

        for m in history:
            all_data.append({
                "epoch": m.step / steps_per_epoch,
                metric_to_plot: m.value,
                "Model + Dataset": combo_label,
                "model_name": model_name,
                "dataset": dataset_group,
                "run_id": run_id
            })

    except Exception as e:
        print(f"Could not fetch {metric_to_plot} for run {run_id}: {e}")

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Build title
title_main = f"{metric_to_plot.replace('_', ' ').title()} over Epochs"
title_sub = "Color: model, Line style: dataset" # (Small vs Large)

# Assign consistent colors for each model
model_palette = dict(zip(
    sorted(model_names),
    sns.color_palette(n_colors=len(model_names))
))

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="epoch",
    y=metric_to_plot,
    hue="model_name",
    style="dataset",
    units="Model + Dataset",  # Ensures each line is distinct
    estimator=None,
    palette=model_palette
)

plt.title(f"{title_main}\n{title_sub}")
plt.xlabel("Epoch")
plt.ylabel(metric_to_plot.replace("_", " ").title())
plt.grid(True)
plt.tight_layout()
plt.ylim(0.85, 1.0)
plt.savefig('accuracy_large2_transformers+.pdf', bbox_inches='tight')
plt.show()



