import cfg
import numpy as np
import matplotlib.pyplot as plt


def ploting():
  data = [np.loadtxt(cfg.works + 'diabetes_loss.txt'),
          np.loadtxt(cfg.works + 'diabetes_accuracy.txt')]
  text = ['Loss', 'Accuracy']

  plt.figure(figsize=(12, 6))
  for i, title in enumerate(text):
    plt.subplot(1, 2, i+1)
    plt.plot(data[i])
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.xlim(0, len(data[i]))
    plt.grid()
  plt.show()


def loss():
  data = np.loadtxt(cfg.works + 'diabetes_loss.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.xlim(0, len(data))
  plt.grid()
  plt.show()


if __name__ == '__main__': loss()
