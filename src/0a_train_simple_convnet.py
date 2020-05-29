from pathlib import Path
from utils.dataset_generator import *
from utils.dataset_splitters import *
from utils.constants import *
from matplotlib import pyplot
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet50

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 3, 512, 512)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(786432, 2)

    def forward(self, xb):
        return self.lin(xb)


def get_model(lr):
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)



def preprocess(x, y):
    return x.view(-1, 3, 512, 512), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))



def get_data(train_ds, valid_ds, test_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
        DataLoader(test_ds, batch_size)
    )

if __name__ == '__main__':
    CURRENT_PATH = Path.cwd()
    SPOP_FALSE_DATA_PATH = CURRENT_PATH.parent / "data" / SPOP_FALSE
    SPOP_TRUE_DATA_PATH = CURRENT_PATH.parent / "data" / SPOP_TRUE

    TRAIN_PROPORTION = 0.7
    VALIDATION_PROPORTION = 0.1
    TEST_PROPORTION = 0.2

    image_data_paths = [SPOP_FALSE_DATA_PATH, SPOP_TRUE_DATA_PATH]
    full_dataset = create_image_dataset_from_image_data_paths(image_data_paths)
    train_set, validation_set, test_set = split_into_train_validation_test_sets(full_dataset, TRAIN_PROPORTION,
                                                                                VALIDATION_PROPORTION, TEST_PROPORTION)

    x_train, y_train = split_set_into_examples_and_labels(train_set)
    x_validation, y_validation = split_set_into_examples_and_labels(validation_set)
    x_test, y_test = split_set_into_examples_and_labels(test_set)

    x_train, y_train, x_validation, y_validation, x_test, y_test = map(
        torch.tensor, (x_train, y_train, x_validation, y_validation, x_test, y_test)
    )

    batch_size = 8

    train_dataset = TensorDataset(x_train, y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_data_loader = WrappedDataLoader(train_data_loader, preprocess)


    validation_dataset = TensorDataset(x_validation, y_validation)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size*2)
    validation_data_loader = WrappedDataLoader(validation_data_loader, preprocess)

    test_dataset = TensorDataset(x_test, y_test)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_data_loader = WrappedDataLoader(test_data_loader, preprocess)

    #############################


    xb = x_train[0:batch_size]  # a mini-batch from x
    yb = y_train[0:batch_size]

    loss_func = nll

    lr = 0.1
    epochs = 2
    train_data_loader, validation_data_loader, test_data_loader = get_data(train_dataset, validation_dataset, test_dataset, batch_size)
    # model, opt = get_model(lr=lr)
    # fit(epochs, model, loss_func, opt, train_data_loader, validation_data_loader)

    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    fit(epochs, model, loss_func, opt, train_data_loader, validation_data_loader)




    # print(len(full_dataset))
    # print(len(train_set))
    # print(len(validation_set))
    # print(len(test_set))

    # pyplot.imshow(test_set[5][0].reshape((512, 512, 3)), cmap="Blues")
    # pyplot.show()

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
#     print(accuracy(preds, yb))
#     n, c = x_train.shape
#
#     lr = 0.005  # learning rate
#     epochs = 10  # how many epochs to train for
#     print("####")
#     for epoch in range(epochs):
#         for i in range((n - 1) // bs + 1):
#             start_i = i * bs
#             end_i = start_i + bs
#             xb,yb = train_dataset[i*bs : i*bs+bs]
#             pred = model(xb)
#             loss = loss_func(pred, yb)
#
#             loss.backward()
#             with torch.no_grad():
#                 weights -= weights.grad * lr
#                 bias -= bias.grad * lr
#                 weights.grad.zero_()
#                 bias.grad.zero_()
#
#     print(loss_func(model(xb), yb), accuracy(model(xb), yb))
#
#     bs = 99
#     model = Mnist_Logistic()
#     print(loss_func(model(xb), yb))
#     fit()
#     print(loss_func(model(xb), yb))
#
#     model, opt = get_model()
#     print(loss_func(model(xb), yb))
