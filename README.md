# Structure
- The run-mlflo-... scripts are what was used for fine-tuning the models. They use the run_image_classification.py script which was modified from HuggingFace example code (https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification).
- Inference scripts are found in the ViT folder.
- Code relating to the processing of the dataset is in the nina_images folder.
- The visualizations folder contains scripts used to create plots and tables from the data collected.

# Reproduce datasets
In order to reproduce the datasets used, the dataprocessing scripts may be applied in the given order:
- combine.py
- structure.py
- preprocessing.py
- smallify.py (if wanting limited dataset)
- split.py
- augment.py (if using limited dataset and wanting augmentation to balance classes more)

# Run mlflow experiments
To track experiments, we can use mlflow run command.

##Single run with parameters
You can omit any of the -P settings (defaults will be used instead)
```
mlflow run . --experiment-name "Test-ViT-Image-Classification"  --env-manager=local -P --m
odel_name_or_path="facebook/deit-base-distilled-patch16-224"   -P train_dir="LessImagesTest/train"   -P validation_dir="LessImagesTest/val"   -P output_dir="./outputs/test-DeiT"   -P learning_rate=5e-05   -P num_train_epochs=3   -P per_device_train_batch_size=8   -P per_device_eval_batch_size=8   -P seed=42
```
##Multiple runs with parameters

To run multiple experiments with different parameters, we can use a script. There is an example in run-mlflow-experiments.py script. Change the script according to the experiments you want to run.


```
python run-mlflow-othersmall-dataset-all-models-384-one-lr.py --experiment-name "Vit-Swin-and-DeIT-other-smallbalanced" --output-dir "outputs/vit-swin-and-deit-other-smallbalanced" --train-dir "smallsplit/trainaug" --validation-dir "smallsplit/val"
```

## Howto start the mlflow server
mlflow ui --host 0.0.0.0 --port 8080

## Run inference with class wise accuracy reporting
python ViT/inference-gpu-classwise-statistics.py --model-path "outputs/vit-swin-and-deit/vit-base-patch16-384_lr1em04_seed42_ep30_bs8" --model-name "google/vit-base-patch16-384" --image-folder "nina_images/testset/" --max-images 100
