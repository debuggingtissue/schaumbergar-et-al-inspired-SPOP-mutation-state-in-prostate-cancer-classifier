from utils import constants
from torchvision import datasets, transforms
from fastai.vision import *

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

def generate_simple_fastai_transformations_for_train_and_validation_image_datasets():
    transforms = get_transforms(do_flip=False, max_rotate=180.)
    return transforms
