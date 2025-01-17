import cfg
import torch
import linearR
import numpy as np
import matplotlib.pyplot as plt


def testing():
  model_file = cfg.works + 'linear.pt'

  model = linearR.NeuralNetwork()
  model.load_state_dict(torch.load(model_file, weights_only=True))
  model.eval()

  x_test = torch.Tensor([-3.1, 3.0, 1.2, -2.5]).view(4, 1)
  print(model(x_test))


def ploting():
  data = np.loadtxt(cfg.works + 'linear.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


if __name__ == '__main__': ploting()
