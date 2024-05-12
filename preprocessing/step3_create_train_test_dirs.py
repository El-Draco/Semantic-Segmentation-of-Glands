import os
import shutil
import random

# Define paths
dataset_path = "dataset/resized"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Create train and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Get the list of images
images = os.listdir(os.path.join(dataset_path, "images"))

# Randomly shuffle the list
random.shuffle(images)

# Define the ratio of train and test data
train_ratio = 0.9  # 90% train+val, 10% test

# Calculate the number of images for train and test
num_train = int(len(images) * train_ratio)
num_test = len(images) - num_train

# Divide the images into train and test sets
train_images = images[:num_train]
test_images = images[num_train:]

# Create directories for images and labels within train and test directories
train_images_path = os.path.join(train_path, "images")
train_labels_path = os.path.join(train_path, "labels")
test_images_path = os.path.join(test_path, "images")
test_labels_path = os.path.join(test_path, "labels")

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# Copy images to train and test directories
for image in train_images:
    shutil.copy(os.path.join(dataset_path, "images", image), train_images_path)
    shutil.copy(os.path.join(dataset_path, "labels", image), train_labels_path)

for image in test_images:
    shutil.copy(os.path.join(dataset_path, "images", image), test_images_path)
    shutil.copy(os.path.join(dataset_path, "labels", image), test_labels_path)
