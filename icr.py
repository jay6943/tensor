import torch
import numpy as np
import torch.utils.data as tud
import matplotlib.pyplot as plt
import matplotlib.colors as cls

path = '../data/torch/icr/'


class Neural_Network(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.f1 = torch.nn.Linear(4, 32)
    self.relu = torch.nn.ReLU()
    self.f2 = torch.nn.Linear(32, 4)

  def forward(self, data):
    data = self.f1(data)
    data = self.relu(data)
    data = self.f2(data)
    return data


class DataSetting(tud.Dataset):
  def __init__(self, x_train, y_train):
    self.x_train = x_train
    self.y_train = y_train

  def __getitem__(self, index):
    return self.x_train[index], self.y_train[index]

  def __len__(self):
    return self.x_train.shape[0]


def training():
  x_data = np.load(path + '1108-164402.npy')
  y_data = np.loadtxt(path + '1108-164402_bits.txt')
  x_data = x_data.transpose()

  x_train = torch.Tensor(x_data[40001:, :])
  y_train = torch.Tensor(y_data)

  dataset = DataSetting(x_train, y_train)
  train_data = tud.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

  model = Neural_Network()
  loss_function = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

  epochs = 101
  loss, loss_data = None, np.zeros(epochs)
  for epoch in range(epochs):
    for i, batch_data in enumerate(train_data):
      x_batch, y_batch = batch_data
      prediction = model(x_batch)
      loss = loss_function(prediction, y_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_data[epoch] = loss.item()
    print(f'{epoch:>5d}, {loss.item():10.2e}')

  for name, child in model.named_children():
    for parameter in child.parameters():
      print(name, parameter)

  np.savetxt(path + 'loss.txt', loss_data)
  torch.save(model.state_dict(), path + 'model.pt')


def testing():
  model_file = path + 'model.pt'

  model = Neural_Network()
  model.load_state_dict(torch.load(model_file, weights_only=True))
  model.eval()

  x_data = np.load(path + '1108-164402.npy')
  x_data = x_data.transpose()

  x_test = torch.Tensor(x_data[:40002, :])
  y_test = model(x_test).detach().numpy()

  return y_test


def ploting():
  data = np.loadtxt(path + 'loss.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


def constellation(fp, data):
  lch = np.max(data)
  plt.figure(figsize=(6 * 2, 6))
  for i in range(2):
    x = data[2 * i]
    y = data[2 * i + 1]
    plt.subplot(1, 2, i+1)
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
  training()
  constellation(path + 'model.png', testing().T)
