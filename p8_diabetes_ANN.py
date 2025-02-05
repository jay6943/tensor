import cfg
import torch
import numpy as np
import pandas as pd
import sklearn.model_selection as sms
import sklearn.metrics as skm


class Artificial_Neural_Network(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.f1 = torch.nn.Linear(8, 20)
    self.relu1 = torch.nn.ReLU()
    self.f2 = torch.nn.Linear(20, 20)
    self.relu2 = torch.nn.ReLU()
    self.f3 = torch.nn.Linear(20, 2)

  def forward(self, data):
    data = self.f1(data)
    data = self.relu1(data)
    data = self.f2(data)
    data = self.relu2(data)
    data = self.f3(data)
    return data


def training():
  df = pd.read_csv(cfg.path + 'diabetes.csv')

  x = df.drop('Outcome', axis=1).values
  y = df['Outcome'].values

  x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size=0.2, random_state=0)

  x_train = torch.FloatTensor(x_train)
  y_train = torch.LongTensor(y_train)
  x_test = torch.FloatTensor(x_test)
  y_test = torch.LongTensor(y_test)

  torch.manual_seed(20)
  model = Artificial_Neural_Network()
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  epochs = 501
  loss_data = np.zeros(epochs)
  for epoch in range(epochs):
    outputs = model.forward(x_train)
    loss = loss_function(outputs, y_train)

    loss_data[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
      print(f'i = {epoch:>4d}, loss = {loss.item():>7.3f}')

  np.savetxt(cfg.path + 'diabetes_loss.txt', loss_data)
  torch.save(model.state_dict(), cfg.path + 'diabetes_model.pt')

  prediction = np.zeros(len(x_test))
  with torch.no_grad():
    for i, data in enumerate(x_test):
      outputs = model(data)
      prediction[i] = outputs.argmax().item()

  print(skm.accuracy_score(y_test, prediction) * 100)


if __name__ == '__main__': training()
