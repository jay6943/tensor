import cfg
import torch
import numpy as np
import torch.utils.data as tud


class NeuralNetwork(torch.nn.Module):
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
  x_data = np.load(cfg.works + 'icr/1108-164402.npy')
  y_data = np.loadtxt(cfg.works + 'icr/1108-164402_bits.txt')
  x_data = x_data.transpose()

  x_train = torch.Tensor(x_data[40001:, :])
  y_train = torch.Tensor(y_data)

  dataset = DataSetting(x_train, y_train)
  train_data = tud.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

  model = NeuralNetwork()
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

  np.savetxt(cfg.works + 'icr/loss.txt', loss_data)
  torch.save(model.state_dict(), cfg.works + 'icr/model.pt')


if __name__ == '__main__': training()
