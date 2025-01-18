import cfg
import numpy as np
import matplotlib.pyplot as plt


def ploting():
  data = {
    '1': np.loadtxt(cfg.works + 'diabetes_loss.txt'),
    '2': np.loadtxt(cfg.works + 'diabetes_accuracy.txt')
  }
  title = {'1': 'Loss', '2': 'Accuracy'}

  plt.figure(figsize=(12, 6))
  for i in [1, 2]:
    plt.subplot(1, 2, i)
    plt.plot(data[str(i)])
    plt.xlabel('Epoch')
    plt.ylabel(title[str(i)])
    plt.xlim(0, len(data[str(i)]))
    plt.grid()
  plt.show()


if __name__ == '__main__': ploting()
