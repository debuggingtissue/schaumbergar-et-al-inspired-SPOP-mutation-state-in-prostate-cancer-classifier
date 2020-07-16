from torchvision import models
import torch.nn as nn

def pretrained_model_optimized_for_data(output_classes_count, device):

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, output_classes_count)  # update last layer to fit our classification problem
    model_ft = model_ft.to(device)