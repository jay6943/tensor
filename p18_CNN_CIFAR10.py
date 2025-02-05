import torch
import torchvision as tvs
import torch.utils.data as tud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


class Convolution_Neural_Network(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.c1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    self.c2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
    self.c3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.c4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
    self.c5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.c6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
    self.c7 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.f1 = torch.nn.Linear(1 * 1 * 256, 128)
    self.f2 = torch.nn.Linear(128, 10)
    self.d4 = torch.nn.Dropout(p=0.25)
    self.d2 = torch.nn.Dropout()

  def forward(self, data):
    data = self.c1(data)
    data = torch.relu(data)
    data = self.c2(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.d4(data)

    data = self.c3(data)
    data = torch.relu(data)
    data = self.c4(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.d4(data)

    data = self.c5(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.d4(data)

    data = self.c6(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.d4(data)

    data = self.c7(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.d4(data)

    data = data.view(-1, 1 * 1 * 256)

    data = self.f1(data)
    data = torch.relu(data)
    data = self.d2(data)
    data = self.f2(data)

    return data


def training():
  train_data = tvs.datasets.CIFAR10(
    root='../data/torch/cifar10',
    transform=tvs.transforms.ToTensor()
  )
  test_data = tvs.datasets.CIFAR10(
    root='../data/torch/cifar10',
    train=False,
    transform=tvs.transforms.ToTensor()
  )

  train_data, valid_data = tud.random_split(train_data, [40000, 10000])
  train_data = tud.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
  valid_data = tud.DataLoader(dataset=valid_data, batch_size=32, shuffle=True)
  test_data = tud.DataLoader(dataset=test_data, batch_size=32, shuffle=True)

  model = Convolution_Neural_Network()
  if torch.cuda.device_count() > 1:
    print('Use', torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
  model.to(device)

  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(20):
    loss, accuracy = model_train(train_data, model, loss_function, optimizer)
    print(f'[{epoch+1:2d}] {loss:.3f}, {accuracy:.2f}', end=', ')

    loss, accuracy = mode_evaluation(valid_data, model, loss_function)
    print(f'{loss:.3f}, {accuracy:.2f}')

  loss, accuracy = model_test(test_data, model, loss_function)
  print(f'loss: {loss:.3f}, accuracy: {accuracy:.2f}')


def model_train(dataloader, model, loss_function, optimizer):
  model.train()

  train_loss_sum = train_correct = train_total = 0
  total_train_batch = len(dataloader)

  for images, labels in dataloader:
    x_train = images.to(device)
    y_train = labels.to(device)

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
      x_val = images.to(device)
      y_val = labels.to(device)

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
      x_test = images.to(device)
      y_test = labels.to(device)

      outputs = model(x_test)
      loss = loss_function(outputs, y_test)

      test_loss_sum += loss.item()
      test_total += y_test.size(0)  # batch size = 32
      test_correct += (torch.argmax(outputs, 1) == y_test).sum().item()

    test_avg_loss = test_loss_sum / total_test_batch
    test_avg_accuracy = 100 * test_correct / test_total

    return test_avg_loss, test_avg_accuracy


if __name__ == '__main__': training()
