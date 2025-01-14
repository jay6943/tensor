import torch
import torch.utils.data as tud
import torchvision.datasets as tds
import torchvision.transforms as ttr
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_data = tds.FashionMNIST(
  root='../data/torch',
  train=True,
  download=False,
  transform=ttr.ToTensor())

test_data = tds.FashionMNIST(
  root='../data/torch',
  train=False,
  download=False,
  transform=ttr.ToTensor())

training_dataloader = tud.DataLoader(training_data, batch_size=64)
test_dataloader = tud.DataLoader(test_data, batch_size=64)


def mapping():
  labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
  }
  figure = plt.figure(figsize=(8, 8))
  cols, rows = 3, 3
  for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
  plt.show()


def shapes():
  for X, y in test_dataloader:
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')
    break


class NeuralNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = torch.nn.Flatten()
    self.liear_relu_stack = torch.nn.Sequential(
      torch.nn.Linear(28 * 28, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, 10))

  def forward(self, x):
    x = self.flatten(x)
    logits = self.liear_relu_stack(x)
    return logits


def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def run():
  model = NeuralNetwork().to(device)
  print(model)

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

  epochs = 5
  for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(training_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
  print('Done!')


if __name__ == '__main__': run()
