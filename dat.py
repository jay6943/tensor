import matplotlib.pyplot as plt


def plot(data):
  plt.figure(figsize=(10, 6))
  plt.plot(data)
  plt.grid()
  plt.show()
