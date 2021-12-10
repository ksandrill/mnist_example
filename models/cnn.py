import torch
from torch import nn


def calc_convolution_output(input_size: int, kernel_size: int, padding: int = 0, stride: int = 1, dilation: int = 1):
    return int(((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)


class Cnn(nn.Module):
    def __init__(self, kernel_number: int = 42, kernel_size: tuple[int, int] = (10, 10), linear_size: int = 40,
                 output_size: int = 10, image_size: tuple[int, int] = (28, 28)):
        super(Cnn, self).__init__()
        self.kernel_number = kernel_number
        self.kernel_size = kernel_size
        self.linear_size = linear_size
        self.output_size = output_size
        self.linear_input_size = calc_convolution_output(image_size[0], kernel_size[0]) * calc_convolution_output(
            image_size[1], kernel_size[1]) * self.kernel_number
        self.convolution_layer = nn.Sequential(nn.Conv2d(1, kernel_number, kernel_size), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Linear(self.linear_input_size, self.linear_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(self.linear_size, self.output_size), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convolution_layer(x)
        out = out.view(-1, self.linear_input_size)
        out = self.layer1(out)
        out = self.layer2(out)
        return out
