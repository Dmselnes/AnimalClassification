import os

# Path to the main folder containing subfolders for each class
base_folder = 'nina_images/testset/'

# Get a list of subfolders in the base folder
class_labels = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]

# Optionally, sort the class labels (if needed)
class_labels.sort()

# Print the class labels
print(f"Number of classes: {len(class_labels)}")
print(class_labels)
