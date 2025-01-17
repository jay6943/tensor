import cfg
import numpy as np
import matplotlib.pyplot as plt


def loss():
  data = np.loadtxt(cfg.works + 'diabetes_loss.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


def accuracy():
  data = np.loadtxt(cfg.works + 'diabetes_accuracy.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


if __name__ == '__main__': accuracy()
