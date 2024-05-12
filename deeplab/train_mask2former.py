from torch import nn
# from transformations import *
# from trainer import Trainer
#from lib.models.axialnet import MedT
#from unet import UNet
#import torchvision.models.segmentation as segmentation
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

import torch
from trainer import Trainer
from transformations import *
from itertools import filterfalse


# Imports
import pathlib
import torch

import albumentations
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from skimage.transform import resize
#from unet import UNet
from customdataset import SegmentationDataSet3

# root directory
root = pathlib.Path.cwd()


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
inputs = get_filenames_of_path(root / '../dataset/resized/balanced_ds/train_test_split/train/images')
targets = get_filenames_of_path(root / '../dataset/resized/balanced_ds/train_test_split/train/labels')

#inputs = inputs[:1000]
#targets = targets[:1000]

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# training transformations and augmentations
transforms_training = ComposeDouble([
    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
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
dataset_train = SegmentationDataSet3(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training,
                                    use_cache=True)
print("finished constructing dataset_train")


# dataset validation
dataset_valid = SegmentationDataSet3(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation,
                                    use_cache=True)
print("finished constructing dataset_valid")



# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=8,
                                 shuffle=True)
print("finished constructing dataloader training")

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=8,
                                   shuffle=True)
print("finished constructing dataloader valid")

for batch_images, batch_labels in dataloader_training:
    print("Batch shape:", batch_images.shape)
    break

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
        inputs = inputs['class_queries_logits']
        #print(f"Input 1: {inputs.shape}")
        #print(f"Target 2: {targets.shape}")
        
        # Ensure input has the correct spatial dimensions for interpolation
        inputs = inputs.unsqueeze(2)  # Add a dummy dimension for interpolation
        #print(f"Input 2: {inputs.shape}")
        inputs = nn.functional.interpolate(
            inputs,
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        #print(f"Input 3: {inputs.shape}")
        inputs = inputs.squeeze(2)  # Remove the dummy dimension after interpolation
        #print(f"Input 4: {inputs.shape}")
        
        # Apply softmax to the inputs
        inputs = nn.functional.softmax(inputs, dim=1)
        #print(f"Input 5: {inputs.shape}")
        
        # Convert targets to long tensor
        targets = targets.long()
        #print(f"Target 3: {targets.shape}")
        
        # Calculate cross-entropy loss
        ce_loss = F.nll_loss(torch.log(inputs), targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        
        return loss




def postprocess(img: torch.tensor, threshold: float = 0.75):
    res = torch.argmax(img, dim=1)  # perform argmax to get the class index
    return res
    

'''
ALPHA = 0.7
BETA = 0.3
GAMMA = 0.75

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        # Comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)
        processed_inputs = postprocess(inputs, threshold=0.75)
        
        # True Positives, False Positives & False Negatives
        TP = (processed_inputs * targets).sum()    
        FP = ((1 - targets) * processed_inputs).sum()
        FN = (targets * (1 - processed_inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)  
        FocalTversky = (1 - Tversky) ** gamma
        FocalTversky.requires_grad=True
        return FocalTversky
'''


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(l, ignore_nan=True, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
    
    
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
    

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, (lovasz_grad(fg_sorted))))
    return mean(losses)
    
    
    
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss
    
#PyTorch
class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        # processed_inputs = postprocess(inputs, threshold=0.75)
        Lovasz = lovasz_softmax(inputs, targets, per_image=False)                  
        return Lovasz

import torch
import torch.nn as nn
import torch.nn.functional as F


# Example usage:
class_weights = torch.FloatTensor(class_weights_train).to(device)
# criterion = FocalLoss(alpha=class_weights, gamma=2)
# loss = loss_function(predictions, ground_truth_masks)


import torch.optim as optim

imgsize = 256
imgchans = 3 
num_classes = 3

config = Mask2FormerConfig(
    image_size=(imgsize, imgsize),
    num_channels=imgchans,
    num_classes=num_classes
)

model = Mask2FormerForUniversalSegmentation(config).to(device)


# criterion
criterion = FocalLoss(alpha=class_weights, gamma=2)
#criterion = FocalTverskyLoss()
#criterion = LovaszHingeLoss()
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
                  epochs=50,
                  epoch=0,
                  notebook=False)


#from lr_finder import LearningRateFinder

#lrf = LearningRateFinder(model, criterion, optimizer, device)
#lrf.fit(dataloader_training, steps=1000)
#lrf.plot()


# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

# save the model
model_name =  'mask2former_50_epochs_adam_actual_ftl.pt'
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
fig.savefig('training_plot_mask2former.png')
