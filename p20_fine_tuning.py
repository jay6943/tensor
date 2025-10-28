import cfg
import torch
import torchvision as tvs
import torch.utils.data as tud
import matplotlib.pyplot as plt

path = f'{cfg.path}/cats_and_dogs_filtered'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


class Transfer_Learning(torch.nn.Module):
  def __init__(self, trained_model, feature_extractor):
    super().__init__()

    if feature_extractor:
      for param in trained_model.parameters():
        param.require_grad = False

    trained_model.heads = torch.nn.Sequential(
      torch.nn.Linear(trained_model.heads[0].in_features, 128),
      torch.nn.ReLU(),
      torch.nn.Dropout(),
      torch.nn.Linear(128, 2)
    )

    self.model = trained_model

  def forward(self, data):
    return self.model(data)


def training():
  train_config = tvs.transforms.Compose(
    [tvs.transforms.Resize((224, 224)),
     tvs.transforms.RandomHorizontalFlip(),
     tvs.transforms.ToTensor()]
  )
  valid_config = tvs.transforms.Compose(
    [tvs.transforms.Resize((224, 224)),
     tvs.transforms.RandomHorizontalFlip(),
     tvs.transforms.ToTensor()]
  )
  train_data = tvs.datasets.ImageFolder(path, train_config)
  valid_data = tvs.datasets.ImageFolder(path, valid_config)

  train_data = tud.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
  valid_data = tud.DataLoader(dataset=valid_data, batch_size=32, shuffle=True)

  trained_model = tvs.models.vit_b_16(weights=tvs.models.ViT_B_16_Weights.DEFAULT)
  model = Transfer_Learning(trained_model, False).to(device)

  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

  for epoch in range(10):
    loss, accuracy = model_train(train_data, model, loss_function, optimizer)
    print(f'[{epoch+1:2d}] {loss:.3f}, {accuracy:.2f}', end=', ')

    loss, accuracy = mode_evaluation(valid_data, model, loss_function)
    print(f'{loss:.3f}, {accuracy:.2f}')


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


def showing():
  train_config = tvs.transforms.Compose(
    [tvs.transforms.Resize((224, 224)),
     tvs.transforms.RandomHorizontalFlip(),
     tvs.transforms.ToTensor()]
  )
  train_dataset = tvs.datasets.ImageFolder(path, train_config)
  train_data = tud.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

  images, labels = next(iter(train_data))
  labels_map = {v: k for k, v in train_dataset.class_to_idx.items()}
  figure = plt.figure(figsize=(6, 7))
  cols, rows = 1, 4
  for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(torch.permute(img, (1, 2, 0)))
  plt.show()


if __name__ == '__main__': showing()
