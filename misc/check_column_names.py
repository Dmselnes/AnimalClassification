from datasets import load_dataset

dataset = load_dataset("LessImagesTest/train")
print(dataset['train'].column_names)
