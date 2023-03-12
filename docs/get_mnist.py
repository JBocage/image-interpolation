import torchvision.datasets as datasets
from utils import PackagePaths

ds = datasets.MNIST(
    root = PackagePaths.DATA,
    train = True,
    download=True,
    transform=None
)