import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from fit import train, test
from models.cnn import Cnn
from models.perceptron import Perceptron
from util.drawer import draw_conf_matrix, draw_roc_curve, draw_roc_curves, draw_metric_bar, draw_graph
from util.metrics import calc_metrics, calc_roc_curve

CLASS_NUMBER = 10
EPOCH_COUNT = 10
LEARNING_RATE = 0.007
from PIL import Image


def get_data_loaders(path_to_save: str, train_batch_size: int = 50) -> tuple[DataLoader, DataLoader]:
    train_set = MNIST(root=path_to_save, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_set = MNIST(root=path_to_save, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    return DataLoader(train_set, batch_size=train_batch_size), DataLoader(test_set)


def main():
    cls_names = [str(i) for i in range(CLASS_NUMBER)]
    train_data_loader, test_data_loader = get_data_loaders('data')
    model = Perceptron(layer_size=40)
    loss_list = train(model, data_loader=train_data_loader, epoch_count=EPOCH_COUNT, lr=LEARNING_RATE)
    conf_matrix, conf_list = test(model, test_data_loader, CLASS_NUMBER)
    draw_graph(loss_list, 'batch', 'cross_entropy', 'loss func')
    precision_list, recall_list, f1_list, cls_counter, not_cls_counter, union_recall = calc_metrics(conf_matrix)
    metrics = {'precision': precision_list, 'recall': recall_list, 'f1 measure': f1_list}
    draw_conf_matrix(conf_matrix, cls_names)
    rog_data = []
    for i in range(len(cls_names)):
        fpr, tpr, auc = calc_roc_curve(i, cls_counter[i], not_cls_counter[i], conf_list)
        rog_data.append((fpr, tpr, auc))
        draw_roc_curve(fpr, tpr, auc, cls_names[i])
    draw_roc_curves(rog_data, cls_names)
    for metrics_name, value in metrics.items():
        draw_metric_bar(cls_names, value, metrics_name)
    draw_metric_bar(['all'], [union_recall], 'union recall')


if __name__ == '__main__':
    main()
