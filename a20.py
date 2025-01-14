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
    logits = self.linear_relu_stack(x)
    return logits


def training():
  data = np.loadtxt(cfg.workspace + 'a20.csv', delimiter=',')

  x_train_data = data[:, :-1]
  y_train_data = data[:, [-1]]

  x_train = torch.Tensor(x_train_data)
  y_train = torch.Tensor(y_train_data)

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

  np.savetxt(cfg.workspace + 'a20.txt', loss_data)
  torch.save(model.state_dict(), cfg.workspace + 'a20.pt')

if __name__ == '__main__': training()
