from models.abstract_models import AbstractModel
import torch
import torch.nn as nn

class Decoder(AbstractModel):

    def __init__(self):

        super(Decoder, self).__init__()

        self.deconv1=nn.ConvTranspose2d(16, 10, kernel_size=4)
        self.deconv2=nn.ConvTranspose2d(10, 4, kernel_size=7)
        self.deconv3=nn.ConvTranspose2d(4, 1, kernel_size=7)

        self.unpool1=nn.MaxUnpool2d(kernel_size=3)
        self.unpool2=nn.MaxUnpool2d(kernel_size=7)

        self.relu=nn.ReLU()

    def forward(self, Z:torch.Tensor):

        deconv1=self.unpool1(self.relu(self.deconv1(Z)))
        deconv2=self.unpool2(self.relu(self.deconv2(deconv1)))
        return self.deconv3(deconv2)