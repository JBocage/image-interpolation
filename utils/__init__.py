import pathlib


class PackagePaths:
    """Contains useful path for the package"""

    ROOT = pathlib.Path(__file__).resolve().parents[3].absolute()
    TEST = ROOT / "tests"
    DATA = ROOT / "data"
    MODELS = ROOT / "models"
