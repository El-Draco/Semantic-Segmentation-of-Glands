import os
import shutil
from PIL import Image

# Define the paths to your images and labels directories
images_dir = "../dataset/images/"
labels_dir = "../dataset/labels/"
moved_images_dir = "../dataset/moved/images/"
moved_labels_dir = "../dataset/moved/labels/"

# Ensure the directories for moved images and labels exist
os.makedirs(moved_images_dir, exist_ok=True)
os.makedirs(moved_labels_dir, exist_ok=True)

# Iterate through images directory
for image_filename in os.listdir(images_dir):
    if image_filename.endswith(".png"):  # Assuming images and labels are both png format
        # Load mask image
        mask_filename = image_filename
        mask_path = os.path.join(labels_dir, mask_filename)
        mask_image = Image.open(mask_path)
        
        # Check if all pixels are equal to 0 (assuming masks are grayscale)
        if all(pixel == 0 for pixel in mask_image.getdata()):
            # Move both image and mask to moved directory
            shutil.move(os.path.join(images_dir, image_filename), os.path.join(moved_images_dir, image_filename))
            shutil.move(mask_path, os.path.join(moved_labels_dir, mask_filename))

print("Moved images with only class 0")


# Counter to keep track of moved images
moved_count = 0

# Iterate through images directory
for image_filename in os.listdir(images_dir):
    if image_filename.endswith(".png"):  # Assuming images and labels are both png format
        # Load mask image
        mask_filename = image_filename
        mask_path = os.path.join(labels_dir, mask_filename)
        mask_image = Image.open(mask_path)
        
        # Check if mask contains only classes 0 and 1 but not class 2
        if all(pixel in [0, 1] for pixel in mask_image.getdata()) and not any(pixel == 2 for pixel in mask_image.getdata()):
            # Move both image and mask to moved directory
            shutil.move(os.path.join(images_dir, image_filename), os.path.join(moved_images_dir, image_filename))
            shutil.move(mask_path, os.path.join(moved_labels_dir, mask_filename))
            moved_count += 1
            
            # Check if we have moved 900 images, exit the loop
            if moved_count == 900:
                break

print("Moved", moved_count, "images and their labels with only classes 0 and 1 (excluding class 2).")