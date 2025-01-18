import cfg
import torch
import numpy as np
import p10_deep_learning
import matplotlib.pyplot as plt


def testing():
  model_file = cfg.works + 'deep_learning_model.pt'

  model = p10_deep_learning.DeepLearning()
  model.load_state_dict(torch.load(model_file, weights_only=True))
  model.eval()

  x_test = torch.Tensor([0.5, 3.0, 3.5, 11.0, 13.0, 31.0]).view(6, 1)
  prediction = model(x_test)
  logical_value = (prediction > 0.5).float()

  print(prediction.T)
  print(logical_value.T)


def ploting():
  data = np.loadtxt(cfg.works + 'multi_linear.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


if __name__ == '__main__': testing()
