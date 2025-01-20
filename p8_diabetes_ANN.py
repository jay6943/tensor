import cfg
import torch
import numpy as np
import pandas as pd


class ANeuralNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.f1 = torch.nn.Linear(8, 20)
    self.relu1 = torch.nn.ReLU()
    self.f2 = torch.nn.Linear(20, 10)
    self.relu2 = torch.nn.ReLU()
    self.f3 = torch.nn.Linear(10, 1)
    self.out = torch.nn.Sigmoid()

  def forward(self, data):
    data = self.f1(data)
    data = self.relu1(data)
    data = self.f2(data)
    data = self.relu2(data)
    data = self.f3(data)
    data = self.out(data)
    return data


def training():
  df = pd.read_csv(cfg.works + 'diabetes.csv')

  x = df.drop('Outcome', axis=1)
  y = df[['Outcome']]

  x_train = torch.Tensor(x.values)
  y_train = torch.Tensor(y.values)

  model = ANeuralNetwork()
  loss_function = torch.nn.BCELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

  epochs = 501
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

    if epoch % 50 == 0:
      print(f'i = {epoch:>4d}', end=', ')
      print(f'loss = {loss.item():>7.3f}', end=', ')
      print(f'accuracy = {accuracy:>7.3f}')

  np.savetxt(cfg.works + 'diabetes_loss.txt', loss_data)
  np.savetxt(cfg.works + 'diabetes_accuracy.txt', accuracy_data)

  torch.save(model.state_dict(), cfg.works + 'diabetes_model.pt')


if __name__ == '__main__': training()
