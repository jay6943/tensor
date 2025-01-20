import torch
import torch.utils.data as tud


class NeuralNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = torch.nn.Sequential(torch.nn.Linear(1, 1))

  def forward(self, x):
    return self.linear_relu_stack(x)


class DataSetting(tud.Dataset):
  def __init__(self, x_train, y_train):
    self.x_train = x_train
    self.y_train = y_train

  def __getitem__(self, index):
    return self.x_train[index], self.y_train[index]

  def __len__(self):
    return self.x_train.shape[0]


def training():
  x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
  y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6, 1)

  dataset = DataSetting(x_train, y_train)
  train_data = tud.DataLoader(dataset=dataset, batch_size=3, shuffle=True)

  model = NeuralNetwork()
  loss_function = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

  for epoch in range(2):
    for i, batch_data, in enumerate(train_data):
      x_batch, y_batch = batch_data
      prediction = model(x_batch)
      loss = loss_function(prediction, y_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f'epoch = {epoch:>5d}, i = {i:>5d}')


if __name__ == '__main__': training()
