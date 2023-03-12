from models.auto_encoder import AutoEncoder
from utils import PackagePaths
from .abstract_trainer import AbstractTrainer
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class AutoencoderTrainer(AbstractTrainer):

    MODEL_TYPE = AutoEncoder

    def __init__(self, trainer_name: str = "default_trainer", overwrite_checkpoint: bool = False, long_description: str = ""):
        super().__init__(trainer_name, overwrite_checkpoint, long_description)
        self.loss_fn = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr = .001)

        self.model:AutoEncoder

    def train(self, batch_size=20, epochs = 10):
        
        dataset = datasets.MNIST(
            root = PackagePaths.DATA,
            train = True,
            download=True,
            transform=lambda x: torch.tensor(np.array(x)/255).float()[None, :]
        )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

        for e in range(epochs):
            for imgs, _ in tqdm(dataloader, desc=f"Training epoch [{e}]"):

                pred_imgs = self.model(imgs)

                loss = self.loss_fn(pred_imgs, imgs)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.loss_record.append(loss.item())
            
            print(f"Finished epoch {e} with loss={loss.item(): .5f}")
            self.save_state()
        
    def save_transfo_fig(self):

        transfo_to_observe = [
            [0, 8],
            [5, 7],
            [5, 2],
            [6, 9]
        ]

        ds = datasets.MNIST(
            root = PackagePaths.DATA,
            train = True,
            download=True,
            transform=lambda x: torch.tensor(np.array(x)/255).float()[None, :]
        )

        nums = {
            i: None for i in range(10)
        }
        idx = 0
        while None in nums.values:
            img, lbl = ds[idx]
            nums[lbl] = img
            idx += 1
        
        fig, axes = plt.subplots(len(transfo_to_observe), 5)
        for (ax_0, ax_25, ax_50, ax_75, ax_100), (lbl1, lbl2) in zip(axes, transfo_to_observe):

            input_1 = nums[lbl1]
            input_2 = nums[lbl2]

            with torch.no_grad():
                enc_1 = self.model.encoder(input_1[None, :])
                enc_2 = self.model.encoder(input_2[None, :])

                out_0 = self.model.decoder(enc_1)[0,0]
                out_25 = self.model.decoder(.75*enc_1 + .25*enc_2)[0,0]
                out_50 = self.model.decoder(.5*enc_1 + .5*enc_2)[0,0]
                out_75 = self.model.decoder(.25*enc_1 + .75*enc_2)[0,0]
                out_100 = self.model.decoder(enc_2)[0,0]
            
            ax_0.imshow(out_0.numpy())
            ax_0.set_title(str(lbl1))
            ax_25.imshow(out_25.numpy())
            ax_50.imshow(out_50.numpy())
            ax_75.imshow(out_75.numpy())
            ax_100.imshow(out_100.numpy())
            ax_100.set_title(str(lbl2))

            fig.savefig(self.save_dest / ".perf_expl.png")
            
    def save_state(self):
        super().save_state()
        self.save_transfo_fig()




