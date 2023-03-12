import torch
from .abstract_model import AbstractModel
from .encoder import Encoder
from .decoder import Decoder

class AutoEncoder(AbstractModel):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(img))
    
    def save_model(self, folder_name, file_name: str = "model"):
        self.encoder.save_model(folder_name=folder_name, file_name=file_name + "encoder")
        self.decoder.save_model(folder_name=folder_name, file_name=file_name + "decoder")

    def load_model(self, folder_name, file_to_load: str = "model"):
        self.encoder.load_model(folder_name=folder_name, file_to_load=file_to_load + "encoder")
        self.decoder.load_model(folder_name=folder_name, file_to_load=file_to_load + "decoder")