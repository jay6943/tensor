import torch


class ConvolutionNeuralNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.f1 = torch.nn.Linear(4, 4)

  def forward(self, data):
    return self.f1(data)


def training():
  print()


if __name__ == '__main__': training()
