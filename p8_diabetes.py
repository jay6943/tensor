import cfg
import torch
import numpy as np
import matplotlib.pyplot as plt


class Logistic_Regression(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.logistic_stack = torch.nn.Sequential(
      torch.nn.Linear(8, 1),
      torch.nn.Sigmoid()
    )

  def forward(self, data):
    return self.logistic_stack(data)


def training():
  data = np.loadtxt(cfg.path + 'diabetes.csv', delimiter=',', skiprows=1)

  x_train = torch.Tensor(data[:, :-1])
  y_train = torch.Tensor(data[:, [-1]])

  model = Logistic_Regression()
  loss_function = torch.nn.BCELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

  epochs = 5001
  loss_data = np.zeros(epochs)
  accuracy_data = np.zeros(epochs)
  for epoch in range(epochs):
    outputs = model(x_train)
    loss = loss_function(outputs, y_train)

    loss_data[epoch] = loss.item()

    prediction = (outputs > 0.5).float()
    correct = (prediction == y_train)[:].float()
    accuracy = correct.sum() / len(correct)

    accuracy_data[epoch] = accuracy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
      print(f'i = {epoch:>4d}', end=', ')
      print(f'loss = {loss.item():>7.3f}', end=', ')
      print(f'accuracy = {accuracy:>7.3f}')

  np.savetxt(cfg.path + 'diabetes_loss.txt', loss_data)
  np.savetxt(cfg.path + 'diabetes_accuracy.txt', accuracy_data)

  torch.save(model.state_dict(), cfg.path + 'diabetes_model.pt')


def ploting():
  data = [np.loadtxt(cfg.path + 'diabetes_loss.txt'),
          np.loadtxt(cfg.path + 'diabetes_accuracy.txt')]
  text = ['Loss', 'Accuracy']

  plt.figure(figsize=(12, 6))
  for i, title in enumerate(text):
    plt.subplot(1, 2, i+1)
    plt.plot(data[i])
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.xlim(0, len(data[i]))
    plt.grid()
  plt.show()


def loss_plot():
  data = np.loadtxt(cfg.path + 'diabetes_loss.txt')

  plt.figure(figsize=(12, 6))
  plt.plot(data)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.xlim(0, len(data))
  plt.grid()
  plt.show()


if __name__ == '__main__':
  training()
  loss_plot()
