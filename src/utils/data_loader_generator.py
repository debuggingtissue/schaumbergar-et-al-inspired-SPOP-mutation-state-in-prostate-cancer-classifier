import torch


def generate_data_loaders_from_image_datasets(image_datasets, fold_names):

    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True)
                   for x in fold_names}

    return data_loaders
