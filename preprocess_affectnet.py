#!/usr/bin/env python3
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Set BALANCED to False to disable down-sampling
BALANCED = True

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

from numpy.random import default_rng
def random_sample(images, n): 
    """Randomly sample n items (without repetition)""" 
    rng = default_rng()
    return rng.choice(images, size=min(len(images), n), replace=False)

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

if BALANCED: 
    train_images_by_class = dict()
    for (folder, image_name, expression) in train_images: 
        if expression not in train_images_by_class: 
            train_images_by_class[expression] = [(folder, image_name)]
        else: 
            train_images_by_class[expression].append((folder, image_name))

    MIN_SIZE = min([len(train_images_by_class[i]) for i in range(8)])

    balanced_train_images = [random_sample(train_images_by_class[i], MIN_SIZE) for i in range(8)]

    print("Balancing dataset to " + str(MIN_SIZE) + " images per class")

# ==========================================================================
# ======================= CREATE DESTINATION FOLDER ========================
# ==========================================================================

FOLDER_NAME = "affectnet_preprocessed_balanced" if BALANCED else "affectnet_preprocessed"

os.mkdir(PROCESSED_DATASET_DESTINATION + FOLDER_NAME)
os.mkdir(PROCESSED_DATASET_DESTINATION + FOLDER_NAME + "/train")
os.mkdir(PROCESSED_DATASET_DESTINATION + FOLDER_NAME + "/test")

for i in range(8): 
    os.mkdir(PROCESSED_DATASET_DESTINATION + FOLDER_NAME + "/train/class_" + str(i))
    os.mkdir(PROCESSED_DATASET_DESTINATION + FOLDER_NAME + "/test/class_" + str(i))

# ==========================================================================
# ============== CONVERT TRAIN IMAGES TO DESTINATION FOLDER ================
# ==========================================================================

done = 0 

if not BALANCED: 
    for (directory, image_name, expression) in train_images: 
        try: 
            # Only store images with a valid expression label
            if expression < 8: 
                source_path = directory + image_name 
                dest_folder = PROCESSED_DATASET_DESTINATION + FOLDER_NAME + "/train/class_" + str(expression) 
                # Converting all images to the .jpg extension
                dest_path = dest_folder + "/" + image_name.split(".")[0] + ".jpg"

                im = Image.open(source_path)
                im.save(dest_path)
                done += 1
                if done % 1000 == 0: 
                    print("Copied", done, "train images...")
        except Exception as e: 
            print("[ERROR] Failed to copy image", image_name, "located in", directory)
            print(e)
else: 
    for expression in range(len(balanced_train_images)): 
        for (directory, image_name) in balanced_train_images[expression]: 
            try: 
                # Only store images with a valid expression label
                source_path = directory + image_name 
                dest_folder = PROCESSED_DATASET_DESTINATION + FOLDER_NAME + "/train/class_" + str(expression) 
                # Converting all images to the .jpg extension
                dest_path = dest_folder + "/" + image_name.split(".")[0] + ".jpg"

                im = Image.open(source_path)
                im.save(dest_path)
                done += 1
                if done % 10000 == 0: 
                    print("Copied", done, "train images...")
            except Exception as e: 
                print("[ERROR] Failed to copy image", image_name, "located in", directory)
                print(e)

# ==========================================================================
# =============== CONVERT TEST IMAGES TO DESTINATION FOLDER ================
# ==========================================================================

for (directory, image_name, expression) in test_images: 
    try: 
        if expression < 8: 
            source_path = directory + image_name 
            dest_folder = PROCESSED_DATASET_DESTINATION + FOLDER_NAME + "/test/class_" + str(expression) 
            # Converting all images to the .jpg extension
            dest_path = dest_folder + "/" + image_name.split(".")[0] + ".jpg"

            im = Image.open(source_path)
            im.save(dest_path) 
    except Exception as e: 
        print("[ERROR] Failed to copy image", image_name, "located in", directory)
        print(e)
        
print("Preprocessing has completed. You can find the preprocessed dataset at:")
print(PROCESSED_DATASET_DESTINATION + FOLDER_NAME)
