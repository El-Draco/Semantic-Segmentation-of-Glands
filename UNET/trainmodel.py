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
from plot import plot_training

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


# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

# save the model
model_name =  'unet_100_epochs_adam_actual_ftl.pt'
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10, 4))
fig.savefig('training_plot.png')
