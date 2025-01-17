import cfg
import numpy as np


def get_data():
  data = np.load(cfg.works + 'icr/1108-164402.npy')
  print(data.shape)


if __name__ == '__main__': get_data()
