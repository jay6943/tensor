import cfg
import icr
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls


def testing():
  model_file = cfg.works + 'icr/model.pt'

  model = icr.NeuralNetwork()
  model.load_state_dict(torch.load(model_file, weights_only=True))
  model.eval()

  x_data = np.load(cfg.works + 'icr/1108-164402.npy')
  x_data = x_data[:, :40002].T
  print(x_data.shape)

  x_test = torch.Tensor(x_data)
  y_test = model(x_test).detach().numpy()

  return y_test


def ploting():
  data = np.loadtxt(cfg.works + 'icr/loss.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


def constellation(fp, data):
  lch = np.max(data)
  plt.figure(figsize=(6 * cfg.NPol, 6))
  for i in range(cfg.NPol):
    x = data[2 * i]
    y = data[2 * i + 1]
    plt.subplot(1, cfg.NPol, i+1)
    plt.hist2d(x, y, bins=500, norm=cls.LogNorm(), cmap='jet')
    plt.gca().set_aspect('equal')
    plt.xlim(-lch, lch)
    plt.ylim(-lch, lch)
    plt.tick_params(bottom=False, labelbottom=False)
    plt.tick_params(left=False, labelleft=False)
    plt.grid(linestyle=':')
  plt.savefig(fp + '.png')
  plt.close()


if __name__ == '__main__':
  constellation(cfg.works + 'icr/model.png', testing().T)
