from utils.constants import *
import torch
import cv2


def get_all_files_in_directory(path):
    return [x for x in path.iterdir() if x.is_file()]


def add_labeled_numpy_image_to_dataset(dataset, numpy_image, label):
    return dataset.append(tuple((numpy_image, label)))


def image_class_string_to_image_class_number(image_class_string):
    if image_class_string == SPOP_FALSE:
        return 0
    elif image_class_string == SPOP_TRUE:
        return 1


def create_image_dataset_from_image_data_paths(image_data_paths):
    dataset = []
    for data_path in image_data_paths:
        image_file_paths = get_all_files_in_directory(data_path)
        for image_file_path in image_file_paths:
            numpy_image = cv2.imread(str(image_file_path))
            numpy_image_flatten = numpy_image.flatten()

            image_class_string = str(image_file_path.parent.stem)
            image_class_number = image_class_string_to_image_class_number(image_class_string)

            add_labeled_numpy_image_to_dataset(dataset, numpy_image_flatten, image_class_number)
    return dataset


def split_into_train_validation_test_sets(full_dataset, train_proportion, validation_proportion, test_proportion):
    train_and_validation_set_count = int(len(full_dataset) * (train_proportion + validation_proportion))
    test_set_count = int(len(full_dataset) * test_proportion)

    train_and_validation_set, test_set = torch.utils.data.random_split(full_dataset, [train_and_validation_set_count
        , test_set_count])

    train_set_count = int(len(train_and_validation_set) * train_proportion / (train_proportion + validation_proportion))
    validation_set_count = int(
        len(train_and_validation_set) * validation_proportion / (train_proportion + validation_proportion))

    train_set, validation_set = torch.utils.data.random_split(train_and_validation_set,
                                                              [train_set_count, validation_set_count])
    return train_set, validation_set, test_set
