import torch
import torchvision as tvs
import torch.utils.data as tud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


class Convolution_Neural_Network(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.c1 = torch.nn.Conv2d(
      in_channels=1,
      out_channels=32,
      kernel_size=3,
      padding=1
    )
    self.c2 = torch.nn.Conv2d(
      in_channels=32,
      out_channels=64,
      kernel_size=3,
      padding=1
    )
    self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.f1 = torch.nn.Linear(7 * 7 * 64, 256)
    self.f2 = torch.nn.Linear(256, 10)
    self.d4 = torch.nn.Dropout(p=0.25)
    self.d2 = torch.nn.Dropout()

  def forward(self, data):
    data = self.c1(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.d4(data)

    data = self.c2(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.d4(data)

    data = data.view(-1, 7 * 7 * 64)

    data = self.f1(data)
    data = torch.relu(data)
    data = self.d2(data)
    data = self.f2(data)

    return data


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
  train_data = tud.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
  valid_data = tud.DataLoader(dataset=valid_data, batch_size=32, shuffle=True)
  test_data = tud.DataLoader(dataset=test_data, batch_size=32, shuffle=True)

  model = Convolution_Neural_Network()
  if torch.cuda.device_count() > 1:
    print('Use', torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
  model.to(device)

  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


if __name__ == '__main__': training()
