import pathlib


class PackagePaths:
    """Contains useful path for the package"""

    ROOT = pathlib.Path(__file__).resolve().absolute().parents[1].absolute()

    DATA = ROOT / "data"
    MODELS = ROOT / "model-checkpoints"
