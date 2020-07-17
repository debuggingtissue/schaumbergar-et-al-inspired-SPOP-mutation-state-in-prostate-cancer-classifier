from utils import constants
from fastai.vision import *


class DataManager:

    def __init__(self, transforms):
        self.transforms = transforms

    def generate_data_bunch_from_path(self, path):
        data = (ImageList.from_folder(path)  # Where to find the data? -> in path and its subfolders
                .split_by_folder(train=constants.TRAIN,
                                 # TRAIN CAN BE EMPTY WHEN USING FOR TEST SET, THUS ONLY PLACEHOLDER
                                 valid=constants.VALIDATION)  # How to split in train/valid? -> use the folders
                .label_from_folder()  # How to label? -> depending on the folder of the filenames
                .transform(self.transforms, size=256)  # Data augmentation? -> use tfms with a size of 64
                .databunch(bs=4))  # Finally? -> use the defaults for conversion to ImageDataBunc
        print(data)
        return data
