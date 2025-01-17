import torchvision.datasets as tds
import torchvision.transforms as ttr

tds.FashionMNIST(
  root='../data/torch',
  train=True,
  download=True,
  transform=ttr.ToTensor())

tds.FashionMNIST(
  root='../data/torch',
  train=False,
  download=True,
  transform=ttr.ToTensor())
