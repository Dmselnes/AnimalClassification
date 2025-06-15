import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

# Connect to tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# 🔄 Toggle: Grouping mode and metric
group_by = "train_dir"  # "model_name" or "train_dir"
metric_to_plot = "eval_accuracy"  # "eval_accuracy" or "loss"

# ✅ Custom label mapping for datasets (only used if group_by == "train_dir")
custom_labels = {
    "smallsplit/train": "Unbalanced",
    "smallCombinedClassesCorrectSplit/train": "Small",
    "smallsplit/trainaug": "Augmented",
    "nina_images/structuredImagesCombinedClassesSplit/train": "Large",
    "nina_images/smallCombinedClassesCorrectSplit/train" : "Small",
    "unknown_dataset": "Unknown",
    # Add more mappings as needed
}

# 🔵 Filter runs based on run_name
wanted_run_names = [
    "kindly-asp-746", #swin
    # "unleashed-eel-979",

    "colorful-turtle-986", #vit
    # "bedecked-snipe-683",

    "abrasive-stoat-465",  #deit
    # "funny-fox-259",

    "inquisitive-fawn-338", #beit
    # "redolent-crow-15", #beit

    # "bittersweet-swan-37",  # resnet50
    # "treasured-mouse-243", #resnet50

    # "adorable-cow-615",     # convnext
    # "delicate-hawk-251",  #convnext
   
    # "redolent-mule-431", #cvt
    "stylish-foal-287",  # cvt

    # "kindly-asp-746",
    # "unleashed-eel-979",

    # "colorful-turtle-986",
    # "bedecked-snipe-683",

    # "abrasive-stoat-465",
    # "funny-fox-259",
    # # #"efficient-auk-336",
    # # #"amazing-flea-724",

    # "inquisitive-fawn-338",
    # "stylish-foal-287", #cvt

    # # "bittersweet-swan-37", #resnet50
    # # "adorable-cow-615" #convnext

    # "redolent-crow-15",
    # "redolent-mule-431",
]

"""
Runs:

Small not balanced:
"swin-base-patch4-window12-384_lr1em04_seed42_ep30_bs8", 
"vit-base-patch16-384_lr1em04_seed42_ep30_bs8",
"deit-base-patch16-384_lr1em04_seed42_ep30_bs8",

Small combined classes:
"unleashed-eel-979",
"bedecked-snipe-683",
"funny-fox-259",

Small augmented:
"smiling-whale-784",
"persistent-deer-342",
"polite-snake-306",

Large:
"kindly-asp-746",
"colorful-turtle-986",
"abrasive-stoat-465",

CNN and other small:
"sneaky-snipe-94",
"gaudy-croc-855",
"redolent-crow-15",
"redolent-mule-431",
"delicate-hawk-251",
"treasured-mouse-243",

CNN and other large:
"efficient-auk-336",
"amazing-flea-724",
"inquisitive-fawn-338",
"stylish-foal-287",
"bittersweet-swan-37",
"adorable-cow-615",

"""

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

# Number of epochs
total_epochs = 30

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
models_in_plot = set()
datasets_in_plot = set()

for _, run in runs.iterrows():
    run_id = run["run_id"]
    model_name = run.get("params.model_name", "unknown_model")
    train_dir = run.get("params.train_dir", "unknown_dataset")
    dataset_label = custom_labels.get(train_dir, train_dir)

    models_in_plot.add(model_name)
    datasets_in_plot.add(dataset_label)

    if group_by == "model_name":
        label_value = model_name
        label_column_name = "model_name"
    elif group_by == "train_dir":
        label_value = dataset_label
        label_column_name = "Dataset"
    else:
        raise ValueError("group_by must be 'model_name' or 'train_dir'")

    try:
        history = client.get_metric_history(run_id, metric_to_plot)
        steps = [m.step for m in history]
        if not steps:
            print(f"No {metric_to_plot} metrics for run {run_id}")
            continue

        # Try to get logged number of epochs
        try:
            run_epochs = int(run.get("params.num_train_epochs", 30))
        except:
            run_epochs = 30

        steps_per_epoch = max(steps) / run_epochs if run_epochs > 0 else 1


        for m in history:
            all_data.append({
                "epoch": m.step / steps_per_epoch,
                metric_to_plot: m.value,
                label_column_name: label_value,
                "run_id": run_id
            })

    except Exception as e:
        print(f"Could not fetch {metric_to_plot} for run {run_id}: {e}")

# Build title
if group_by == "train_dir":
    title_main = f"{metric_to_plot.replace('_', ' ').title()} over Epochs (by Dataset)"
    title_sub = "Model: " + ", ".join(sorted(models_in_plot))
else:
    title_main = f"{metric_to_plot.replace('_', ' ').title()} over Epochs (by Model)"
    title_sub = "Dataset: " + ", ".join(sorted(datasets_in_plot))

# Plot
df = pd.DataFrame(all_data)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="epoch", y=metric_to_plot, hue=label_column_name)
plt.title(f"{title_main}\n{title_sub}")
plt.xlabel("Epoch")
plt.ylabel(metric_to_plot.replace("_", " ").title())
#plt.ylim(0.8, 1.0)
plt.grid(True)
plt.tight_layout()
#plt.savefig('loss_large_transformers+_plot.pdf', bbox_inches='tight')
plt.show()



"""
plot plan:

✅ all small sets base models loss and validation accuracy over epochs

✅ all large models loss and validation accuracy over epochs (one including CNNs, one not)
(table for inference metrics)

✅ all small vs large models loss and validation accuracy over epochs
(table for inference metrics)

graph of class-wise accuracy of all models (on all large, and all small, and all aug datasets) (one including CNNs, one not?)

"""