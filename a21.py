import cfg
import a20
import torch
import numpy as np
import matplotlib.pyplot as plt


def ploting():
  data = np.loadtxt(cfg.workspace + 'a20.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()

def testing():
  filename = cfg.workspace + 'a20.pt'

  model = a20.NeuralNetwork()
  model.load_state_dict(torch.load(filename, weights_only=True))
  model.eval()

  x_test = torch.Tensor([
    [5, 5, 0],
    [2, 3, 1],
    [-1, 0, -1],
    [10, 5, 2],
    [4, -1, -2]])

  print(model(x_test))


if __name__ == '__main__': testing()
