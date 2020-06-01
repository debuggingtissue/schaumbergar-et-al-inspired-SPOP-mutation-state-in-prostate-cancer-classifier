import torch
from utils import constants

def generate_data_loaders_from_image_datasets(image_datasets):

    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True)
                   for x in [constants.TRAIN, constants.VALIDATION]}

    return data_loaders
