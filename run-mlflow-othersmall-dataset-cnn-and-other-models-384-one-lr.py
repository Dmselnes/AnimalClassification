import mlflow.projects
from itertools import product
import argparse
import os

def main(args):
    # Sweep values
    models = [
        "microsoft/resnet-50",
        "facebook/convnext-base-384",
        "microsoft/cvt-13-384",
        "microsoft/beit-base-patch16-384",
        "microsoft/resnet-18",
        "microsoft/resnet-152",
        #"zetatech/pvt-medium-224"
        #"google/efficientnet-b3"
    ]
    learning_rates = [1e-04]
    seeds = [42]
    num_train_epochs = [30]
    batch_sizes = [8]  # Applies to both train and eval
    
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
                "run_name": f"{model_tag}_lr{lr_str}_seed{seed}_ep{epochs}_bs{batch_size}",
                "train_dir": args.train_dir,
                "validation_dir": args.validation_dir,
                "learning_rate": lr,
                "seed": seed,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": batch_size,
                "per_device_eval_batch_size": batch_size,
                "output_dir": run_output_dir,
                "ignore_mismatched_sizes": True,
            },
            env_manager="local",
            experiment_name=args.experiment_name,
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
        "--run_name",
        type=str,
        default="default-run"
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
