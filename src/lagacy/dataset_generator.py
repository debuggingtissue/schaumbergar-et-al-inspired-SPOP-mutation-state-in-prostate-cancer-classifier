from lagacy.dataset_generator import *
from utils.constants import *


def get_all_files_in_directory(path):
    return [x for x in path.iterdir() if x.is_file()]


def add_labeled_numpy_image_to_dataset(dataset, numpy_image, label):
    return dataset.append(tuple((numpy_image, label)))


def image_class_string_to_image_class_number(image_class_string):
    if image_class_string == SPOP_NOT_MUTATED:
        return 0
    elif image_class_string == SPOP_MUTATED:
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



