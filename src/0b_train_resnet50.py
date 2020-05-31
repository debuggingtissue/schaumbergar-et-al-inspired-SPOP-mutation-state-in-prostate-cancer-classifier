# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from pathlib import Path
from utils import ensemble_manager, path_utils

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ensambles_count = 2
    model_count_in_each_ensemble = 11
    validation_accuracy_requirement_for_each_model = 0.6

    ensembles = ensemble_manager.generate_ensembles(ensambles_count, model_count_in_each_ensemble,
                                                    validation_accuracy_requirement_for_each_model,
                                                    device)
