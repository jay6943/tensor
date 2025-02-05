import cfg
import torch
import numpy as np
import matplotlib.pyplot as plt


class Deep_Learning(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.deeplearning_stack = torch.nn.Sequential(
      torch.nn.Linear(1, 8),
      torch.nn.Linear(8, 1),
      torch.nn.Sigmoid()
    )

  def forward(self, x):
    return self.deeplearning_stack(x)


def training():
  x_train = torch.Tensor([2, 4, 5, 8, 10, 12, 14, 16, 18, 20]).view(10, 1)
  y_train = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).view(10, 1)

  model = Deep_Learning()
  loss_function = torch.nn.BCELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

  epochs = 5001
  loss_data = np.zeros(epochs)
  for epoch in range(epochs):
    prediction = model(x_train)
    loss = loss_function(prediction, y_train)

    loss_data[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print(f'i = {epoch:>4d}, loss = {loss.item():.4e}')

  np.savetxt(cfg.path + 'deep_learning.txt', loss_data)
  torch.save(model.state_dict(), cfg.path + 'deep_learning_model.pt')


def testing():
  model_file = cfg.path + 'deep_learning_model.pt'

  model = Deep_Learning()
  model.load_state_dict(torch.load(model_file, weights_only=True))
  model.eval()

  x_test = torch.Tensor([0.5, 3.0, 3.5, 11.0, 13.0, 31.0]).view(6, 1)
  prediction = model(x_test)
  logical_value = (prediction > 0.5).float()

  print(prediction.T)
  print(logical_value.T)


def ploting():
  data = np.loadtxt(cfg.path + 'multi_linear.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


if __name__ == '__main__':
  training()
  ploting()
