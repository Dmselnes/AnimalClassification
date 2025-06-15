import mlflow.projects
from itertools import product
import argparse
import os

#  "facebook/deit-base-distilled-patch16-224"

def main(args):
    # Sweep values
    models = [
        "facebook/deit-base-patch16-384",
        "google/vit-base-patch16-224-in21k",
        "microsoft/swin-base-patch4-window7-224-in22k",
    ]
    learning_rates = [1e-04, 3e-04, 5e-04, 1e-03]
    seeds = [42]
    num_train_epochs = [30]
    batch_sizes = [32]  # Applies to both train and eval

    # Sweep over all combinations including models
    for model_name, lr, seed, epochs, batch_size in product(models, learning_rates, seeds, num_train_epochs, batch_sizes):
        model_tag = model_name.split("/")[-1]
        lr_str = f"{lr:.0e}".replace("-", "m")  # e.g., 3e-4 -> 3e4

        run_output_dir = os.path.join(
            args.output_dir,
            f"{model_tag}_lr{lr_str}_seed{seed}_ep{epochs}_bs{batch_size}"
        )

        print(f"Running: model={model_tag}, lr={lr}, seed={seed}, epochs={epochs}, batch_size={batch_size}")
        mlflow.projects.run(
            uri=".",
            entry_point="main",
            parameters={
                "model_name_or_path": model_name,
                "train_dir": args.train_dir,
                "validation_dir": args.validation_dir,
                "learning_rate": lr,
                "seed": seed,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": batch_size,
                "per_device_eval_batch_size": batch_size,
                "output_dir": run_output_dir,
            },
            env_manager="local",
            experiment_name=args.experiment_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow sweep for image classification.")
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default="Default-Experiment", 
        help="Name of the MLflow experiment to log runs under."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs", 
        help="Base output directory for model checkpoints and logs."
    )
    parser.add_argument(
        "--model-name-or-path", 
        type=str, 
        default="google/vit-base-patch16-224-in21k", 
        help="(Unused in sweep) Model name or path to pretrained weights."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="LessImagesTest/train",
        help="Path to training image folder."
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default="LessImagesTest/val",
        help="Path to validation image folder."
    )
    args = parser.parse_args()
    main(args)
