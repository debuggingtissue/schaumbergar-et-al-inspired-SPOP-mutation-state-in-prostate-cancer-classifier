from utils import model_trainer, model_manager, dataset_splitters, data_loader_generator, dataset_generator, constants
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


def generate_ensembles(ensemble_count, model_count_in_each_ensemble, path_to_origin_image_folder,
                       validation_accuracy_threshold, device):
    # data_sets, output_classes_count, dataset_sizes, device

    for ensemble_index in range(ensemble_count):

        # TODO - draw test set here without replacement

        for model_index in range(model_count_in_each_ensemble):

            monte_carlo_drawn_images_root_path = dataset_splitters.get_monte_carlo_drawn_train_and_validation_sets_root_path(
                path_to_origin_image_folder, ensemble_index, model_index)
            image_datasets = dataset_generator.generate_transformed_train_and_validation_image_datasets(monte_carlo_drawn_images_root_path)
            data_loaders = data_loader_generator.generate_data_loaders_from_image_datasets(image_datasets)

            dataset_sizes = {x: len(image_datasets[x]) for x in [constants.TRAIN, constants.VALIDATION]}
            class_names = image_datasets[constants.TRAIN].classes
            output_classes_count = len(class_names)

            pretrained_model_optimized_for_data = model_manager.pretrained_model_optimized_for_data(output_classes_count)
            criterion = nn.CrossEntropyLoss()

            # Observe that all parameters are being optimized
            optimizer = optim.SGD(pretrained_model_optimized_for_data.parameters(), lr=0.001, momentum=0.9)

            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            model_with_validation_accuracy_threshold = model_trainer.train_model(pretrained_model_optimized_for_data,
                                                                                 criterion, optimizer,
                                                                                 exp_lr_scheduler, data_loaders,
                                                                                 dataset_sizes,
                                                                                 device,
                                                                                 validation_accuracy_threshold)
