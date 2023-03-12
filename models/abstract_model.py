"""Implements an abstract class for the models"""
from abc import abstractmethod

import torch
import torch.nn as nn

from ..utils import PackagePaths


class AbstractModel(nn.Module):
    """Abstract class for models"""

    # Global Variable: No need to overwrite
    CHECKPOINT_DIR = PackagePaths.MODELS / "layout_model_chkpt"

    def __init__(self):
        super(AbstractModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """Processes x"""

    def load_model(self, folder_name, file_to_load: str = "model"):
        """Loads model from the checkpoint folder"""
        savepath = self.CHECKPOINT_DIR / folder_name
        self.load_state_dict(
            torch.load(
                savepath / (file_to_load + ".chkpt"), map_location=torch.device("cpu")
            )
        )

    def save_model(self, folder_name, file_name: str = "model"):
        """Saves model in the checkpoint folder"""
        savepath = self.CHECKPOINT_DIR / folder_name
        savepath.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), savepath / (file_name + ".chkpt"))

    def eval_mode(self):
        """Sets eval mode for the model"""
        self.eval()

    def train_mode(self):
        """Sets train model for the model"""
        self.train()
