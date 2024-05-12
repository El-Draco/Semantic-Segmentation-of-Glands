import os
from PIL import Image

# Define the paths to your images and labels directories
images_dir = "dataset/images/"
labels_dir = "dataset/labels/"
resized_images_dir = "dataset/resized/images/"
resized_labels_dir = "dataset/resized/labels/"

# Ensure the directories for resized images and labels exist
os.makedirs(resized_images_dir, exist_ok=True)
os.makedirs(resized_labels_dir, exist_ok=True)

# Function to resize image and save
def resize_and_save(image_path, label_path, output_image_path, output_label_path, size=(256, 256)):
    image = Image.open(image_path)
    label = Image.open(label_path)

    # Resize image
    resized_image = image.resize(size)
    resized_label = label.resize(size)

    # Save resized images
    resized_image.save(output_image_path)
    resized_label.save(output_label_path)

# Iterate through images directory
for image_filename in os.listdir(images_dir):
    if image_filename.endswith(".png"):  # Assuming images and labels are both png format
        # Construct paths
        image_path = os.path.join(images_dir, image_filename)
        label_filename = image_filename
        label_path = os.path.join(labels_dir, label_filename)
        output_image_path = os.path.join(resized_images_dir, image_filename)
        output_label_path = os.path.join(resized_labels_dir, label_filename)
        
        # Resize and save
        resize_and_save(image_path, label_path, output_image_path, output_label_path)

print("Images and labels resized and saved successfully.")
