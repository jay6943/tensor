import cfg
import torch
import numpy as np
import multi_linearR
import matplotlib.pyplot as plt


def testing():
  model_file = cfg.works + 'multi_linear_model.pt'

  model = multi_linearR.NeuralNetwork()
  model.load_state_dict(torch.load(model_file, weights_only=True))
  model.eval()

  x_test = torch.Tensor([
    [5, 5, 0],
    [2, 3, 1],
    [-1, 0, -1],
    [10, 5, 2],
    [4, -1, -2]])

  print(model(x_test))


def ploting():
  data = np.loadtxt(cfg.works + 'multi_linear.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


if __name__ == '__main__': testing()
