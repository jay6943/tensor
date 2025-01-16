import cfg
import torch
import numpy as np


class LogisticRegression(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.logistic_stack = torch.nn.Sequential(
      torch.nn.Linear(8, 1),
      torch.nn.Sigmoid()
    )

  def forward(self, data):
    return self.logistic_stack(data)


def training():
  data = np.loadtxt(cfg.workspace + 'diabetes.csv', delimiter=',', skiprows=1)

  x_train = torch.Tensor(data[:, :-1])
  y_train = torch.Tensor(data[:, [-1]])

  model = LogisticRegression()
  loss_function = torch.nn.BCELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

  epochs = 5001
  loss_data = np.zeros(epochs)
  accuracy_data = np.zeros(epochs)
  for epoch in range(epochs):
    outputs = model(x_train)
    loss = loss_function(outputs, y_train)

    loss_data[epoch] = loss.item()

    prediction = outputs > 0.5
    # correct = (prediction.float() == y_train)
    # accuracy = correct.sum().item() / len(correct)
    correct = np.where(prediction.float() == y_train)
    accuracy = len(correct[0]) / len(outputs)

    accuracy_data[epoch] = accuracy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
      print(f'{epoch:>5d}, {loss.item():10.3f}, {accuracy:10.3f}')

  np.savetxt(cfg.workspace + 'diabetes_loss.txt', loss_data)
  np.savetxt(cfg.workspace + 'diabetes_accuracy.txt', accuracy_data)

  torch.save(model.state_dict(), cfg.workspace + 'diabetes_model.pt')


if __name__ == '__main__': training()
