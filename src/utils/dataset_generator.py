from utils.constants import *
import cv2
import numpy as np
from pathlib import Path
from utils.dataset_generator import *
from utils.dataset_splitters import *
from utils.constants import *
from torchvision import datasets, transforms
import os


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
            numpy_image_flatten = numpy_image.flatten().astype(np.float32)
            numpy_image_flatten *= 255.0 / numpy_image_flatten.max()

            image_class_string = str(image_file_path.parent.stem)
            image_class_number = image_class_string_to_image_class_number(image_class_string)

            add_labeled_numpy_image_to_dataset(dataset, numpy_image_flatten, image_class_number)
    return dataset


# For data augmentation, the 800x800px images at
# high magnification were trimmed to the centermost 512x512px ===== DONE IN DATA AQUASITION PART
# in 6 degree rotations ==== DONE IN , scaled to 256x256px, flipped and unflipped ==== DONE but WTF, then randomly cropped to 224x224 within Caffe [22]
def generate_transformed_train_and_validation_image_datasets(image_data_path):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((0, 360)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # Always make sure the values are such that the range of each channel value is [0,1]??
        ]),
        'val': transforms.Compose([
            transforms.RandomRotation((0, 360)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # Always make sure the values are such that the range of each channel value is [0,1]??
        ]),
    }

    print(image_data_path)
    image_datasets = {x: datasets.ImageFolder(os.path.join(image_data_path, x),
                                              data_transforms[x])
                      for x in [constants.TRAIN, constants.VALIDATION]}

    return image_datasets
