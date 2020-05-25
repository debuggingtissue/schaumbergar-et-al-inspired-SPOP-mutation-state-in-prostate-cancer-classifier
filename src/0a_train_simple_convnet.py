from pathlib import Path
from utils.dataset_generator import *
from utils.constants import *

if __name__ == '__main__':
    CURRENT_PATH = Path.cwd()
    SPOP_FALSE_DATA_PATH = CURRENT_PATH.parent / "data" / SPOP_FALSE
    SPOP_TRUE_DATA_PATH = CURRENT_PATH.parent / "data" / SPOP_TRUE

    image_data_paths = [SPOP_FALSE_DATA_PATH, SPOP_TRUE_DATA_PATH]
    dataset = create_image_dataset_from_image_data_paths(image_data_paths)


    print(dataset)
    # print(SPOP_FALSE_DATA_PATH)
    # print(SPOP_TRUE_DATA_PATH)

# PATH = DATA_PATH / "mnist"
#
# PATH.mkdir(parents=True, exist_ok=True)
#
# URL = "http://deeplearning.net/data/mnist/"
# FILENAME = "mnist.pkl.gz"
#
# if not (PATH / FILENAME).exists():
#         content = requests.get(URL + FILENAME).content
#         (PATH / FILENAME).open("wb").write(content)
#
# import pickle
# import gzip
#
# with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
#         ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
#
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset
#
# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size=bs)
