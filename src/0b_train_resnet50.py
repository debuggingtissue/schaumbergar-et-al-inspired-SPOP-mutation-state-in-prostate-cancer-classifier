# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

from utils import ensemble_manager, data_manager, transform_definitions_generator, test_manager,data_paths

if __name__ == '__main__':

    data_augmentation_transforms = transform_definitions_generator.generate_simple_fastai_transformations_for_train_and_validation_image_datasets()
    data_manager = data_manager.DataManager(data_augmentation_transforms)

    ensambles_count = 2
    model_count_in_each_ensemble = 11
    ensembles = ensemble_manager.generate_ensembles(ensambles_count, model_count_in_each_ensemble, data_manager)

    test_manager = test_manager.TestManager(data_paths.TEST_DATA_PATH, ensembles)
    test_manager.validate()



    # validation_accuracy_requirement_for_each_model = 0.6
    #
    #
    #
    #     data_manager.DataManager(transforms)
    # test_data_bunch = data_manager.generate_data_bunch_from_path(monte_carlo_drawn_images_root_path, transforms, constants.TEST)
    #
    #
    #
    # generate_data_bunch_from_path
    #
    #
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# image_datasets = dataset_generator.generate_transformed_train_and_validation_image_datasets(monte_carlo_drawn_images_root_path)
    # data_loaders = data_loader_generator.generate_data_loaders_from_image_datasets(image_datasets)
    #
    # dataset_sizes = {x: len(image_datasets[x]) for x in [constants.TRAIN, constants.VALIDATION]}
    # class_names = image_datasets[constants.TRAIN].classes
    # output_classes_count = len(class_names)

    # pretrained_model_optimized_for_data = model_manager.pretrained_model_optimized_for_data(output_classes_count)
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that all parameters are being optimized
    # optimizer = optim.SGD(pretrained_model_optimized_for_data.parameters(), lr=0.001, momentum=0.9)
    #
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #
    # model_with_validation_accuracy_threshold = model_trainer.train_model(pretrained_model_optimized_for_data,
    #                                                                      criterion, optimizer,
    #                                                                      exp_lr_scheduler, data_loaders,
    #                                                                      dataset_sizes,
    #                                                                      device,
    #                                                                      validation_accuracy_threshold)
