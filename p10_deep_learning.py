import cfg
import torch
import numpy as np


class DeepLearning(torch.nn.Module):
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

  model = DeepLearning()
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

  np.savetxt(cfg.works + 'deep_learning.txt', loss_data)
  torch.save(model.state_dict(), cfg.works + 'deep_learning_model.pt')


if __name__ == '__main__': training()
