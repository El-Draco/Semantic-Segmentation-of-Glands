# Imports
import pathlib

import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize

# from inference import predict
from transformations import normalize_01, re_normalize
from UNet import UNet

# root directory
root = pathlib.Path.cwd() / '../dataset/test/'
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
images_names = get_filenames_of_path(root / 'images')
targets_names = get_filenames_of_path(root / 'labels')

# read images and store them in memory
images = [imread(img_name) for img_name in images_names]
targets = [imread(tar_name) for tar_name in targets_names]

# Resize images and targets
#images_res = [resize(img, (128, 128, 3)) for img in images]
#resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
#targets_res = [resize(tar, (128, 128), **resize_kwargs) for tar in targets]


import torch


def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs

    return result


images_res = images
targets_res = targets
# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')


# model
model = UNet(in_channels=3,
             out_channels=3,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)


model_name = 'unet_100_epochs_adam_actual_ftl.pt'
model_weights = torch.load(pathlib.Path.cwd() / model_name)

model.load_state_dict(model_weights)
print('model loaded succesfully')
    

# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# postprocess function with thresholding
def postprocess(img: torch.tensor, threshold: float = 0.5):
    img = torch.argmax(img, dim=1)  # perform argmax to get the class index
    img = img.cpu().numpy()  # send to CPU and transform to numpy.ndarray
    return img






# predict the segmentation maps 
threshold = 0.80  # Adjust as needed
output = [predict(img, model, preprocess, lambda x: postprocess(x, threshold), device) for img in images_res]


import numpy as np
import matplotlib.pyplot as plt

def save_predictions(images, true_masks, predicted_masks, num_images=5, output_filename='predictions.png'):
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 3*num_images))
    for i in range(num_images):
        axes[i, 0].imshow(images[i])
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original Image')
        
        axes[i, 1].imshow(true_masks[i], cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Original Mask')
        
        axes[i, 2].imshow(predicted_masks[i][0], cmap='gray')  # Assuming predicted_masks is a list of single-channel masks
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Predicted Mask')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

# Assuming images_res is a list of RGB images, targets_res is a list of single-channel masks, and output is a list of single-channel predicted masks
save_predictions(images_res, targets_res, output)

import multiprocessing
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score


import numpy as np

# Define a function to calculate Dice coefficient for a single class
def calculate_dice(true_mask_class, pred_mask_class):
    intersection = (true_mask_class * pred_mask_class).sum()
    total_true = true_mask_class.sum()
    total_pred = pred_mask_class.sum()
    dice = 2 * intersection / (total_true + total_pred)
    return dice
    
def calculate_average_dice(true_masks, predicted_masks, num_classes, batch_size):
    # Split the data into batches
    num_batches = int(np.ceil(len(true_masks) / batch_size))
    batch_avg_dice = []
    classwise_dice = [[] for _ in range(num_classes)]  # List to store class-wise Dice coefficients

    # Compute average Dice coefficient for each batch
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(true_masks))
        batch_true_masks = true_masks[start_idx:end_idx]
        batch_predicted_masks = predicted_masks[start_idx:end_idx]

        dice_coefficients = []

        # Compute Dice coefficient for each class in the batch
        for class_id in range(num_classes):
            true_mask_class = (batch_true_masks == class_id).astype(int)
            pred_mask_class = (batch_predicted_masks == class_id).astype(int)
            dice = calculate_dice(true_mask_class, pred_mask_class)
            dice_coefficients.append(dice)
            classwise_dice[class_id].append(dice)  # Append instead of extend

        # Compute average Dice coefficient for the batch
        batch_avg_dice.append(np.mean(dice_coefficients))

    # Compute overall average Dice coefficient
    overall_avg_dice = np.mean(batch_avg_dice)

    # Compute final class-wise Dice coefficients
    final_classwise_dice = [np.mean(dice_list) for dice_list in classwise_dice]

    return overall_avg_dice, final_classwise_dice

# Define true_masks_np and predicted_masks_np within the scope of calculate_metrics_parallel
true_masks_np = np.array(targets_res)
output = np.squeeze(output, axis=1)
predicted_masks_np = np.array(output)


num_classes = 3
batch_size = 64

overall_avg_dice, final_classwise_dice = calculate_average_dice(true_masks_np, predicted_masks_np, num_classes, batch_size)

print("Overall Average Dice Coefficient:", overall_avg_dice)
print("Final Class-wise Dice Coefficients:")
for class_id, dice_coefficient in enumerate(final_classwise_dice):
    print(f"Class {class_id}: {dice_coefficient}")


print()
# Define calculate_metrics_for_class as a top-level function
def calculate_metrics_for_class(class_id, true_masks, predicted_masks):
    true_mask_class = (true_masks == class_id).astype(int)
    pred_mask_class = (predicted_masks == class_id).astype(int)

    # Calculate metrics
    f1 = f1_score(true_mask_class.flatten(), pred_mask_class.flatten())
    precision = precision_score(true_mask_class.flatten(), pred_mask_class.flatten())
    recall = recall_score(true_mask_class.flatten(), pred_mask_class.flatten())
    iou = jaccard_score(true_mask_class.flatten(), pred_mask_class.flatten())

    # Calculate sensitivity and specificity
    tp = np.sum(true_mask_class * pred_mask_class)
    tn = np.sum((1 - true_mask_class) * (1 - pred_mask_class))
    fp = np.sum((1 - true_mask_class) * pred_mask_class)
    fn = np.sum(true_mask_class * (1 - pred_mask_class))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        'F1 score': f1,
        'Precision': precision,
        'Recall': recall,
        'IoU': iou,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }


def calculate_metrics_parallel(true_masks, predicted_masks, num_classes, batch_size=64):
    num_batches = len(true_masks) // batch_size + (1 if len(true_masks) % batch_size != 0 else 0)
    
    pool = multiprocessing.Pool()
    metrics_per_class = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(true_masks))

        batch_true_masks = true_masks[start_idx:end_idx]
        batch_predicted_masks = predicted_masks[start_idx:end_idx]

        batch_metrics = pool.starmap(calculate_metrics_for_class, [(class_id, batch_true_masks, batch_predicted_masks) for class_id in range(num_classes)])
        metrics_per_class.append(batch_metrics)
    pool.close()
    pool.join()

    # Calculate average class-wise metrics
    classwise_avg_metrics = {}
    for class_id in range(num_classes):
        class_metrics = {metric_name: [] for metric_name in metrics_per_class[0][0].keys()}
        for batch_metrics in metrics_per_class:
            for metric_name, value in batch_metrics[class_id].items():
                class_metrics[metric_name].append(value)
        avg_class_metrics = {metric_name: np.mean(values) for metric_name, values in class_metrics.items()}
        classwise_avg_metrics[f'Class {class_id}'] = avg_class_metrics

    # Calculate final aggregated metrics
    final_metrics = {}
    for metric_name in classwise_avg_metrics['Class 0'].keys():
        final_metrics[metric_name] = np.mean([classwise_avg_metrics[f'Class {class_id}'][metric_name] for class_id in range(num_classes)])

    return final_metrics, classwise_avg_metrics




# Define true_masks_np and predicted_masks_np within the scope of calculate_metrics_parallel
true_masks_np = np.array(targets_res)
predicted_masks_np = np.array(output)



# Calculate metrics in parallel
final_metrics, classwise_avg_metrics = calculate_metrics_parallel(true_masks_np, predicted_masks_np, num_classes=3)

# Print the final aggregated metrics
print("Final Aggregated Metrics:")
for metric_name, value in final_metrics.items():
    print(f"{metric_name}: {value}")

print()

# Print the average class-wise metrics
print("\nAverage Class-wise Metrics:")
for class_name, metrics in classwise_avg_metrics.items():
    print(f"\n{class_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")
