# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
from pathlib import Path
from utils import dataset_generator, data_loader_generator, ensemble_manager

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
    output_classes_count = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ensambles_count = 2
    ensembles = ensemble_manager.generate_ensembles(ensambles_count)
