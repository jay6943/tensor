import torchvision.datasets as tds
import torchvision.transforms as ttr


def number_MNIST():
  tds.MNIST(
    root='../data/torch',
    download=True,
    transform=ttr.ToTensor()
  )
  tds.MNIST(
    root='../data/torch',
    train=False,
    download=True,
    transform=ttr.ToTensor()
  )


def fashion_MNIST():
  tds.FashionMNIST(
    root='../data/torch',
    download=True,
    transform=ttr.ToTensor()
  )
  tds.FashionMNIST(
    root='../data/torch',
    train=False,
    download=True,
    transform=ttr.ToTensor()
  )


def cifar10():
  tds.CIFAR10(
    root='../data/torch/cifar10',
    download=True,
    transform=ttr.ToTensor()
  )
  tds.CIFAR10(
    root='../data/torch/cifar10',
    train=False,
    download=True,
    transform=ttr.ToTensor()
  )


if __name__ == '__main__': cifar10()
