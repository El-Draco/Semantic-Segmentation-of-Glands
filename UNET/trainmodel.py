from torch import nn
# from transformations import *
# from trainer import Trainer
from UNet import UNet
import torch
from Trainer import Trainer
from itertools import filterfalse
from transformations import *
# Imports
import pathlib
import torch
import albumentations
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from skimage.transform import resize
from GlandDataset import GlandDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# root directory
root = pathlib.Path.cwd()


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
inputs = get_filenames_of_path(root / '../dataset/train/images')
targets = get_filenames_of_path(root / '../dataset/train/labels')

#inputs = inputs[:1000]
#targets = targets[:1000]

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# training transformations and augmentations
transforms_training = ComposeDouble([
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# validation transformations
transforms_validation = ComposeDouble([
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# random seed
random_seed = 42

# split dataset into training set and validation set
train_size = 0.9  # 

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    targets,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)


import cv2

# Load target images and convert them into numerical labels
targets_train_images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in targets_train]

# Convert target images to numerical labels
targets_train_labels = [np.unique(img) for img in targets_train_images]

# Flatten and concatenate all labels
targets_train_flat = np.concatenate(targets_train_labels)

# Calculate class counts and class weights for training data
class_counts_train = np.bincount(targets_train_flat)
num_classes_train = len(class_counts_train)
total_samples_train = len(targets_train_flat)
class_weights_train = [1 / (count / total_samples_train) for count in class_counts_train]



# inputs_train, inputs_valid = inputs[:80], inputs[80:]
# targets_train, targets_valid = targets[:80], targets[:80]

# dataset training
dataset_train = GlandDataset(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training,
                                    use_cache=True)
print("finished constructing dataset_train")


# dataset validation
dataset_valid = GlandDataset(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation,
                                    use_cache=True)
print("finished constructing dataset_valid")



# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=16,
                                 shuffle=True)
print("finished constructing dataloader training")

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=16,
                                   shuffle=True)
print("finished constructing dataloader valid")



# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("device : ", device)


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

def postprocess(img: torch.tensor, threshold: float = 0.75):
    res = torch.argmax(img, dim=1)  # perform argmax to get the class index
    return res
    




# model
model = UNet(in_channels=3,
             out_channels=3,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)

# criterion
class_weights = torch.FloatTensor(class_weights_train).to(device)
criterion = FocalLoss(alpha=class_weights, gamma=2)
# optimizer
optimizer = torch.optim.Adam(model.parameters())

# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=100,
                  epoch=0,
                  notebook=False)


#from lr_finder import LearningRateFinder

#lrf = LearningRateFinder(model, criterion, optimizer, device)
#lrf.fit(dataloader_training, steps=1000)
#lrf.plot()


# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

# save the model
model_name =  'unet_100_epochs_adam_actual_ftl.pt'
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

def plot_training(training_losses,
                  validation_losses,
                  learning_rate,
                  gaussian=True,
                  sigma=2,
                  figsize=(8, 6)
                  ):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    subfig2 = fig.add_subplot(grid[0, 1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')

    # Subfig 2
    subfig2.plot(x_range, learning_rate, color='black')
    subfig2.title.set_text('Learning rate')
    subfig2.set_xlabel('Epoch')
    subfig2.set_ylabel('LR')

    return fig

fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10, 4))
fig.savefig('training_plot.png')
