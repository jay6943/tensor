import cfg
import torch
import numpy as np
import matplotlib.pyplot as plt


class Neural_Network(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = torch.nn.Sequential(torch.nn.Linear(1, 1))

  def forward(self, x):
    return self.linear_relu_stack(x)


def training():
  x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
  y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6, 1)

  model = Neural_Network()
  loss_function = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

  epochs = 2001
  loss_data = np.zeros(epochs)
  for epoch in range(epochs):
    prediction = model(x_train)
    loss = loss_function(prediction, y_train)

    loss_data[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print(f'i = {epoch:>4d}, loss = {loss.item():10.2e}')

  np.savetxt(cfg.works + 'linear.txt', loss_data)
  torch.save(model.state_dict(), cfg.works + 'linear_model.pt')


def testing():
  model_file = cfg.works + 'linear_model.pt'

  model = Neural_Network()
  model.load_state_dict(torch.load(model_file, weights_only=True))
  model.eval()

  x_test = torch.Tensor([-3.1, 3.0, 1.2, -2.5]).view(4, 1)
  print(model(x_test))


def ploting():
  data = np.loadtxt(cfg.works + 'linear.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.grid()
  plt.show()


if __name__ == '__main__':
  training()
  ploting()
