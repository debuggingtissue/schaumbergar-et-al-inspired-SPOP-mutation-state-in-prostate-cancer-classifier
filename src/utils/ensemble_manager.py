from utils import model_trainer, model_manager
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


def generate_ensembles(ensemble_count, model_count_in_each_ensemble, validation_accuracy_threshold, data_loaders,
                       output_classes_count, dataset_sizes, device):
    for ensemble in range(ensemble_count):
        # TODO - draw test set here without replacement
        for model in range(model_count_in_each_ensemble):
            # TODO - draw train set here without replacement
            # TODO - draw validation set here without replacement

            pretrained_model_optimized_for_data = model_manager.pretrained_model_optimized_for_data(
                output_classes_count)
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
                                                                                 validation_accuracy_threshold,
                                                                                 num_epochs=25)
