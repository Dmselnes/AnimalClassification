import os
import json
import shutil

# Define paths
json_file_path = 'json/JSON_files/all_species.json'  # Path to your JSON file
source_images_dir = 'allImages'  # Path to your source image directory
output_dir = 'structuredImagesCombinedClasses'  # Path to the target directory where you want the structure

# Load the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Create the directories if they don't exist
for annotation in data['annotations']:
    species_id = annotation['Species_ID']
    species_name = annotation['Nor_species_name']
    species_dir = os.path.join(output_dir, str(species_name))  # You can use species_name if you prefer
    if not os.path.exists(species_dir):
        os.makedirs(species_dir)

# Move images into their respective directories
for annotation in data['annotations']:
    filename = annotation['Filename']
    species_id = annotation['Species_ID']
    species_name = annotation['Nor_species_name']
    
    # Define source and destination paths
    source_image_path = os.path.join(source_images_dir, filename)
    species_dir = os.path.join(output_dir, str(species_name))
    
    # Check if the image exists in the source directory
    if os.path.exists(source_image_path):
        # Move the image to the corresponding species folder
        destination_image_path = os.path.join(species_dir, filename)
        shutil.move(source_image_path, destination_image_path)
        #print(f"Moved {filename} to {species_dir}")
    else:
        print(f"Warning: {filename} not found in source directory")

print("All images have been moved to the appropriate species folders.")
