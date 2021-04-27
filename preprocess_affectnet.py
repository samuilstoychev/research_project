#!/usr/bin/env python3
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Replace this variable with a list of top-level directories 
containing the AffectNet dataset. 

For example, if AffectNet is split across home/folder1, home/folder2, home/folder3, 
each in turn containing subfolders such as 1, 2, 3 ..., set the variable as: 
TOP_LEVEL_DIRECTORIES = ["/home/folder1/", "home/folder2/", "home/folder3/"]
 
Make sure to include the training slash character. 
"""

TOP_LEVEL_DIRECTORIES = ["/Users/samuilstoychev/Manually_Annotated_Images/"]

TRAINING_ANNOTATIONS_PATH = "/Users/samuilstoychev/affectnet_annotations/training.csv"
VALIDATION_ANNOTATIONS_PATH = "/Users/samuilstoychev/affectnet_annotations/validation.csv"

# Also with a trailing slash 
PROCESSED_DATASET_DESTINATION = "/Users/samuilstoychev/"

# ==========================================================================
# ========================= AUXILIARY FUNCTIONS ============================
# ==========================================================================

import os
def listdir(path):
    """List all items in a given directory."""
    res = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            res.append(f)
    return res

# ==========================================================================
# ========================= LOAD ANNOTATIONS ===============================
# ==========================================================================

training_df = pd.read_csv(TRAINING_ANNOTATIONS_PATH)
validation_df = pd.read_csv(VALIDATION_ANNOTATIONS_PATH, header=None)

# A dictionary mapping (subfolder name, image name) --> expression label
train_annotations = dict()
for row in training_df.values: 
    folder, image = row[0].split("/")
    expression = int(row[-3])
    train_annotations[(folder, image)] = expression 
    
test_annotations = dict()
for row in validation_df.values: 
    folder, image = row[0].split("/")
    expression = int(row[-3])
    test_annotations[(folder, image)] = expression 

print("Total number of annotations:") 
print("Training annotations =", len(train_annotations))
print("Test annotations =", len(test_annotations))
print()

# ==========================================================================
# ========================= GET ALL IMAGE LOCATIONS ========================
# ==========================================================================

# A list of triples (directory path, subfolder name, image name)
all_images = []

for directory in TOP_LEVEL_DIRECTORIES: 
    folders = listdir(directory)
    for folder in folders: 
        for image in listdir(directory + "/" + folder): 
            all_images.append((directory, folder , image))

# ==========================================================================
# ==================== DISTRIBUTE IMAGES TO FOLDERS ========================
# ==========================================================================

# A list of triples (path to subfolder, image_name, expression_label)
train_images = []
test_images = [] 
# A list of tuples (path to subfolder, image_name)
unannotated_images = []

for (directory, folder, image) in all_images: 
    if (folder, image) in train_annotations: 
        train_images.append((directory + folder + "/", image, train_annotations[(folder, image)]))
    elif (folder, image) in test_annotations: 
        test_images.append((directory + folder + "/", image, test_annotations[(folder, image)]))
    else: 
        unannotated_images.append((directory + folder + "/", image))

print("Total number of images:", len(all_images))
print("Number of train images:", len(train_images))
print("Number of test images:", len(test_images))
print("Number of unannotated images:", len(unannotated_images))
print()

# ==========================================================================
# ======================= CREATE DESTINATION FOLDER ========================
# ==========================================================================

os.mkdir(PROCESSED_DATASET_DESTINATION + "affectnet_preprocessed")
os.mkdir(PROCESSED_DATASET_DESTINATION + "affectnet_preprocessed/train")
os.mkdir(PROCESSED_DATASET_DESTINATION + "affectnet_preprocessed/test")

for i in range(8): 
    os.mkdir(PROCESSED_DATASET_DESTINATION + "affectnet_preprocessed/train/class_" + str(i))
    os.mkdir(PROCESSED_DATASET_DESTINATION + "affectnet_preprocessed/test/class_" + str(i))

# ==========================================================================
# ============== CONVERT AND COPY IMAGES TO DESTINATION FOLDER =============
# ==========================================================================

done = 0 

# Copying train images
for (directory, image_name, expression) in train_images: 
    try: 
        # Only store images with a valid expression label
        if expression < 8: 
            source_path = directory + image_name 
            dest_folder = PROCESSED_DATASET_DESTINATION + "affectnet_preprocessed/train/class_" + str(expression) 
            # Converting all images to the .jpg extension
            dest_path = dest_folder + "/" + image_name.split(".")[0] + ".jpg"

            im = Image.open(source_path)
            im.save(dest_path)
            done += 1
            if done % 10000 == 0: 
                print("Copied", done, "train images...")
    except Exception as e: 
        print("[ERROR] Failed to copy image", image_name, "located in", directory)

# Copying test images
for (directory, image_name, expression) in test_images: 
    try: 
        if expression < 8: 
            source_path = directory + image_name 
            dest_folder = PROCESSED_DATASET_DESTINATION + "affectnet_preprocessed/test/class_" + str(expression) 
            # Converting all images to the .jpg extension
            dest_path = dest_folder + "/" + image_name.split(".")[0] + ".jpg"

            im = Image.open(source_path)
            im.save(dest_path) 
    except Exception as e: 
        print("[ERROR] Failed to copy image", image_name, "located in", directory)
        print(e)
        
print("Preprocessing has completed. You can find the preprocessed dataset at:")
print(PROCESSED_DATASET_DESTINATION + "affectnet_preprocessed")
