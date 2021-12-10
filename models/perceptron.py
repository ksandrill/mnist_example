import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size: int = 784, layer_size: int = 20, output_size: int = 10):
        super(Perceptron, self).__init__()
        self.layer1_size = layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.layer1 = nn.Sequential(nn.Linear(input_size, layer_size), nn.LeakyReLU(0.04))
        self.layer2 = nn.Sequential(nn.Linear(layer_size, output_size), nn.LeakyReLU(0.04))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x.view(-1, self.input_size))
        out = self.layer2(out)
        return out
