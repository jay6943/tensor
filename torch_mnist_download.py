import cfg
import torchvision.datasets as tds
import torchvision.transforms as ttr


def number_MNIST():
  tds.MNIST(
    root=cfg.path,
    download=True,
    transform=ttr.ToTensor()
  )
  tds.MNIST(
    root=cfg.path,
    train=False,
    download=True,
    transform=ttr.ToTensor()
  )


def fashion_MNIST():
  tds.FashionMNIST(
    root=cfg.path,
    download=True,
    transform=ttr.ToTensor()
  )
  tds.FashionMNIST(
    root=cfg.path,
    train=False,
    download=True,
    transform=ttr.ToTensor()
  )


def cifar10():
  tds.CIFAR10(
    root=f'{cfg.path}/cifar10',
    download=True,
    transform=ttr.ToTensor()
  )
  tds.CIFAR10(
    root=f'{cfg.path}/cifar10',
    train=False,
    download=True,
    transform=ttr.ToTensor()
  )


if __name__ == '__main__': cifar10()
