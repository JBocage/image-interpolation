"""Implements a AbstractTrainer class"""
import datetime
import json
from abc import ABC, abstractmethod

import joblib
import matplotlib.pyplot as plt

from ..models import AbstractModel


class AbstractTrainer(ABC):
    """Abstract class for trainer objects"""

    MODEL_TYPE = AbstractModel

    SAVED_IN_STATE = [
        "epochs_trained",
        "loss_record",
        "val_loss_record",
        "val_acc_record",
        "trainer_init_date",
        "long_description",
    ]

    def __init__(
        self,
        trainer_name: str = "default_trainer",
        overwrite_checkpoint: bool = False,
        long_description: str = "",
    ):

        self.name = trainer_name
        self.model: AbstractModel = self.MODEL_TYPE()

        # RESULT
        self.epochs_trained = 0
        self.loss_record = []  # Loss of each batch
        self.val_loss_record = []  # Average val loss per epoch
        self.val_acc_record = []

        self.long_description: str = long_description
        self.trainer_init_date: datetime.datetime = datetime.datetime.now()

        # LOADING TRAINING INFORMATION (if exist)
        self.save_dest = self.model.CHECKPOINT_DIR / self.name
        if self.save_dest.exists() and not overwrite_checkpoint:
            # if "Model Checkpoint Folder" exist & overwrite_checkpoint=False
            self.load_state()

    def load_state(self):
        """Loads the state of the trainer using its name"""
        # Load the model chkpt 's state_dict based on Folder Name (self.name)
        self.model.load_model(self.name)
        # Load the trainer state dict
        trainer_state_dict = joblib.load(self.save_dest / "trainer.state")
        # Update the class's parameters based on "trainer.state"
        for k in trainer_state_dict.keys():
            if k in self.__dict__.keys():
                self.__dict__[k] = trainer_state_dict[k]

    def save_state(self):
        """Saves the trainer state"""
        self.model.save_model(self.name)
        self.save_loss_fig()
        trainer_state_dict = {k: self.__dict__[k] for k in self.SAVED_IN_STATE}
        joblib.dump(trainer_state_dict, self.save_dest / "trainer.state")
        training_nfo = {
            "creation_date": self.trainer_init_date.strftime("%m/%d/%Y %H:%M:%S"),
            "last_save_date": datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            "model_class": str(self.model.__class__),
            "model_type": str(self.model.__class__.__name__),
            "number_of_backwards": len(self.loss_record),
            "long_description": self.long_description,
        }
        with open(self.save_dest / "training_nfo.json", "w+") as f:
            json.dump(training_nfo, f, indent=4, separators=(",", ": "), sort_keys=True)

    def save_loss_fig(self):
        """Saves the loss figure of the trainer"""
        plt.ioff()  # to make the matplotlib not in interactive mode

        fig, ax = plt.subplots()
        ax.plot(self.loss_record, color="blue", label="Loss per Batch")
        ax.plot(self.val_loss_record, color="green", label="Val Loss per Epoch")
        ax.set(
            title=f"Training Loss for {self.name} Model",
            xlabel="Number of Batches trained",
            ylabel="Log Loss",
            yscale="log",
        )

        ax.legend()
        plt.tight_layout()

        fig.savefig(self.save_dest / ".trainingloss.png")
        plt.close(fig)
        plt.ion()

    @abstractmethod
    def train(self):
        """Trains the model"""
