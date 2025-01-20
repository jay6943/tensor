import torch
import torchvision as tvs
import torch.utils.data as tud


class DeepLearning(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = torch.nn.Flatten()
    self.fc1 = torch.nn.Linear(784, 256)
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(0.3)
    self.fc2 = torch.nn.Linear(256, 10)

  def forward(self, data):
    data = self.flatten(data)
    data = self.fc1(data)
    data = self.relu(data)
    data = self.dropout(data)
    logits = self.fc2(data)
    return logits


def model_train(dataloader, model, loss_function, optimizer):
  model.train()

  train_loss_sum = train_correct = train_total = 0
  total_train_batch = len(dataloader)

  for images, labels in dataloader:
    x_train = images.view(-1, 28 * 28)
    y_train = labels

    outputs = model(x_train)
    loss = loss_function(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_sum += loss.item()
    train_total += y_train.size(0)
    train_correct += (torch.argmax(outputs, 1) == y_train).sum().item()

  train_avg_loss = train_loss_sum / total_train_batch
  train_avg_accuracy = 100 * train_correct / train_total

  return train_avg_loss, train_avg_accuracy


def mode_evaluation(dataloader, model, loss_function):
  model.eval()

  with torch.no_grad():
    val_loss_sum = val_correct = val_total = 0
    total_val_batch = len(dataloader)

    for images, labels in dataloader:
      x_val = images.view(-1, 28 * 28)
      y_val = labels

      outputs = model(x_val)
      loss = loss_function(outputs, y_val)

      val_loss_sum += loss.item()
      val_total += y_val.size(0)
      val_correct += (torch.argmax(outputs, 1) == y_val).sum().item()

    val_avg_loss = val_loss_sum / total_val_batch
    val_avg_accuracy = 100 * val_correct / val_total

  return val_avg_loss, val_avg_accuracy


def model_test(dataloader, model, loss_function):
  model.eval()

  with torch.no_grad():
    test_loss_sum = test_correct = test_total = 0
    total_test_batch = len(dataloader)

    for images, labels in dataloader:
      x_test = images.view(-1, 28 * 28)
      y_test = labels

      outputs = model(x_test)
      loss = loss_function(outputs, y_test)

      test_loss_sum += loss.item()
      test_total += y_test.size(0)
      test_correct += (torch.argmax(outputs, 1) == y_test).sum().item()

    test_avg_loss = test_loss_sum / total_test_batch
    test_avg_accuracy = 100 * test_correct / test_total

    return test_avg_loss, test_avg_accuracy


def training():
  train_data = tvs.datasets.MNIST(
    root='../data/torch',
    transform=tvs.transforms.ToTensor()
  )

  test_data = tvs.datasets.MNIST(
    root='../data/torch',
    train=False,
    transform=tvs.transforms.ToTensor()
  )

  train_data, valid_data = tud.random_split(train_data, [50000, 10000])
  train_load = tud.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
  valid_load = tud.DataLoader(dataset=valid_data, batch_size=32, shuffle=True)
  test_load = tud.DataLoader(dataset=test_data, batch_size=32, shuffle=True)

  model = DeepLearning()
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

  train_loss = []
  train_accuracy = []
  val_loss = []
  val_accuracy = []

  for epoch in range(20):
    loss, accuracy = model_train(train_load, model, loss_function, optimizer)
    train_loss.append(loss)
    train_accuracy.append(accuracy)
    print(f'{epoch}, {loss}, {accuracy}', end=', ')

    loss, accuracy = mode_evaluation(valid_load, model, loss_function)
    val_loss.append(loss)
    val_accuracy.append(accuracy)
    print(f'{loss}, {accuracy}')

  loss, accuracy = model_test(test_load, model, loss_function)
  print(f'{loss}, {accuracy}')


if __name__ == '__main__': training()
