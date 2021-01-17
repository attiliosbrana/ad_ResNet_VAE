import torch
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, resnet18


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * (t.tanh(F.softplus(x)))
        return x

class MLP_Block(t.nn.Module):
    def __init__(self,
                 input_size: int,
                 z_dim: int,
                 layers: int,
                 layer_size: int):
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features = input_size, out_features = layer_size)] +
                                      [t.nn.Linear(in_features = layer_size, out_features = layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features = layer_size, out_features = 2 * z_dim)
        self.Mish = Mish()

    def forward(self, x: t.Tensor):
        block_input = x
        for layer in self.layers:
            block_input = self.Mish(layer(block_input))
        z = self.basis_parameters(block_input)
        return z

class ResNet_Encoder(nn.Module):
    def __init__(self, z_dim, mlp_layers, mlp_layer_size):
        super(ResNet_Encoder, self).__init__()
        self.z_dim = z_dim
        self.Mish = Mish()
        
        resnet = resnet18(pretrained=True, progress=True)
        modules = list(resnet.children())[:-1]
        for module in modules:
          for param in module.parameters():
            param.requires_grad = False
        self.en_conv = t.nn.Sequential(*modules)
        self.Flatten = t.nn.Flatten()
        self.en_mlp = MLP_Block(512, z_dim, mlp_layers, mlp_layer_size)

    def split_z(self, z):
        z_mu = z[:, :self.z_dim]
        z_sigma = z[:, self.z_dim:]
        return z_mu, z_sigma

    def sample_z(self, mu, sigma):
        epsilon = torch.randn_like(mu)
        sample = mu + (sigma * epsilon)
        return sample

    def forward(self, input):
        convout = self.Flatten(self.en_conv(input))
        z = self.en_mlp(convout)
        z_mu, z_sigma = z[:, :self.z_dim], z[:, self.z_dim:]
        z_enc = self.sample_z(mu = z_mu, sigma = z_sigma)
        return z_enc, convout, z

class Decoder(nn.Module):

    def __init__(self, height, width, channel, kernel_size, z_dim):
        super(Decoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ksize, self.z_dim = kernel_size, z_dim
        self.Mish = Mish()

        self.de_dense = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            Mish(),
            nn.Linear(512, (self.height//(2**2))*(self.width//(2**2))*64),
            Mish(),
        )

        self.de_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            Mish(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            Mish(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.ksize+1, stride=2, padding=1),
            Mish(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            Mish(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.ksize+1, stride=2, padding=1),
            Mish(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            Mish(),
            nn.Conv2d(in_channels=16, out_channels=self.channel, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.Sigmoid(),
        )

    def forward(self, input):
        denseout = self.de_dense(input)
        denseout_res = denseout.view(denseout.size(0), 64, (self.height//(2**2)), (self.width//(2**2)))
        x_hat = self.de_conv(denseout_res)
        return x_hat.view(-1, self.channel, self.height, self.width)