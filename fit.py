import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, data_loader: DataLoader, epoch_count: int, lr: float):
    model.train()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    timer = tqdm(range(epoch_count), desc="it's training time!")
    loss_list = []
    for _ in timer:
        avg_loss = 0.0
        for x, y in data_loader:
            outputs = model(x)
            outputs_loss = loss(outputs, y)
            softed_outputs = F.softmax(outputs, 1).data
            avg_loss += outputs_loss.item()
            optimizer.zero_grad()
            outputs_loss.backward()
            optimizer.step()
            torch.argmax(softed_outputs.data, 1)
        avg_loss /= len(data_loader)
        timer.set_description("loss:  " + str(f'{avg_loss:.4f}'), refresh=True)
        loss_list.append(avg_loss)
    return loss_list


def test(model, test_loader: DataLoader, cls_number: int) -> (np.ndarray, list[tuple[int, np.ndarray]]):
    conf_matrix = np.zeros(shape=(cls_number, cls_number))
    confidence_list: list[tuple[int, np.ndarray]] = []
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            outputs = model(x)
            softed_outputs = F.softmax(outputs, 1).data
            predicted = torch.argmax(softed_outputs, 1)
            total += len(y)
            for i in range(len(y)):
                conf_matrix[predicted[i]][y[i]] += 1
                confidence_list.append((y[i], softed_outputs[i]))
            correct += (predicted == y).sum().item()
        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
        return conf_matrix, confidence_list


def test_only_acc(model, test_loader: DataLoader) -> float:
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            outputs = model(x)
            softed_outputs = F.softmax(outputs, 1).data
            predicted = torch.argmax(softed_outputs, 1)
            total += len(y)
            correct += (predicted == y).sum().item()
        accuracy = (correct / total) * 100
        print('Test Accuracy of the model: {} %'.format(accuracy))
    return accuracy
