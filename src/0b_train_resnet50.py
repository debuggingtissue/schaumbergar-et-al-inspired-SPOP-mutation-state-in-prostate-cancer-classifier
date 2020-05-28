# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from pathlib import Path
from utils import dataset_generator, data_loader_generator, model_trainer

# plt.ion()  # interactive mode


if __name__ == '__main__':
    CURRENT_PATH = Path.cwd()
    DATA_PATH = CURRENT_PATH.parent / "data" / "splits"
    TRAIN = "train"
    VALIDATION = "val"
    fold_names = [TRAIN, VALIDATION]

    image_datasets = dataset_generator.generate_transformed_train_and_validation_image_datasets(DATA_PATH, fold_names)
    data_loaders = data_loader_generator.generate_data_loaders_from_image_datasets(image_datasets, fold_names)
    dataset_sizes = {x: len(image_datasets[x]) for x in fold_names}
    class_names = image_datasets[TRAIN].classes
    number_of_output_classes = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, number_of_output_classes) #update last layer to fit our classification problem
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = model_trainer.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data_loaders,
                                         dataset_sizes,
                                         device,
                                         num_epochs=25)
