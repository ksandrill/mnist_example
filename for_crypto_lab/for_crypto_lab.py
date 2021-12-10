from fit import train, test_only_acc
from main import get_data_loaders
from models.perceptron import Perceptron

EPOCH_COUNT = 1
LEARNING_RATE = 0.01
LAYER_SIZE = 10


def make_train_test(layer_size: int, epoch_count: int, learning_rate: float):
    train_data_loader, test_data_loader = get_data_loaders('../data')
    model = Perceptron(layer_size=layer_size)
    _ = train(model, data_loader=train_data_loader, epoch_count=epoch_count, lr=learning_rate)
    test_only_acc(model, test_data_loader)


if __name__ == '__main__':
    make_train_test(layer_size=LAYER_SIZE, epoch_count=EPOCH_COUNT, learning_rate=LEARNING_RATE)
