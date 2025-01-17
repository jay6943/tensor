import cfg
import torch
import numpy as np


class NeuralNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = torch.nn.Sequential(
      # torch.nn.Linear(4, 4))
      torch.nn.Linear(4, 64),
      torch.nn.ReLU(),
      torch.nn.Linear(64, 64),
      torch.nn.ReLU(),
      torch.nn.Linear(64, 4))

  def forward(self, x):
    return self.linear_relu_stack(x)


def training():
  x_data = np.loadtxt(cfg.works + 'icr/1108-164402.txt')
  y_data = np.loadtxt(cfg.works + 'icr/1108-164402_bits.txt')

  x_train = torch.Tensor(x_data)
  y_train = torch.Tensor(y_data)

  model = NeuralNetwork()
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
      print(f'{epoch:>5d}, {loss.item():10.2e}')

  for name, child in model.named_children():
    for parameter in child.parameters():
      print(name, parameter)

  np.savetxt(cfg.works + 'icr/loss.txt', loss_data)
  torch.save(model.state_dict(), cfg.works + 'icr/model.pt')


if __name__ == '__main__': training()
