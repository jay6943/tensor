import cfg
import torch

class NeuralNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = torch.nn.Sequential(
      torch.nn.Linear(1, 1)
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits


def training():
  x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
  y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6, 1)

  model = NeuralNetwork()
  loss_function = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

  for epoch in range(2001):
    prediction = model(x_train)
    loss = loss_function(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print(f'{epoch:>5d}, {loss.item():10.2e}')

  torch.save(model.state_dict(), cfg.workspace + 'a10.pt')

if __name__ == '__main__': training()
