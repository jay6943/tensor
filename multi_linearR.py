import cfg
import torch
import numpy as np


class NeuralNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = torch.nn.Sequential(
      torch.nn.Linear(3, 1)
    )

  def forward(self, x):
    return self.linear_relu_stack(x)


def training():
  data = torch.Tensor([
    [1, 2, 0, -4],
    [5, 4, 3, 4],
    [1, 2, -1, -6],
    [3, 1, 0, 3],
    [2, 4, 2, -4],
    [4, 1, 2, 9],
    [-1, 3, 2, -7],
    [4, 3, 3, 5],
    [0, 2, 6, 6],
    [2, 2, 1, 0],
    [1, -2, -2, 4],
    [0, 1, 3, 3],
    [1, 1, 3, 5],
    [0, 1, 4, 5],
    [2, 3, 3, 1],
  ])

  x_train = torch.Tensor(data[:, :-1])
  y_train = torch.Tensor(data[:, [-1]])

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

  np.savetxt(cfg.works + 'multi_linear.txt', loss_data)
  torch.save(model.state_dict(), cfg.works + 'multi_linear_model.pt')


if __name__ == '__main__': training()
