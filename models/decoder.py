from models.abstract_model import AbstractModel
import torch
import torch.nn as nn

class Decoder(AbstractModel):

    def __init__(self):

        super(Decoder, self).__init__()

        self.deconv1=nn.ConvTranspose2d(16, 28, kernel_size=2)
        self.deconv2=nn.ConvTranspose2d(28, 8, kernel_size=4)
        self.deconv3=nn.ConvTranspose2d(8, 4, kernel_size=3)
        self.conv1=nn.Conv2d(4, 1, kernel_size=5)

        self.unpool=nn.UpsamplingNearest2d(scale_factor=(2,2))

        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, Z:torch.Tensor):

        deconv1=self.unpool(self.relu(self.deconv1(Z)))
        deconv2=self.unpool(self.relu(self.deconv2(deconv1)))
        deconv3=self.unpool(self.relu(self.deconv3(deconv2)))

        return self.sigmoid(self.conv1(deconv3))
