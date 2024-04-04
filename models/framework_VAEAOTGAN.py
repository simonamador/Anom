from models.anomaly import Anomaly
import models.aotgan.aotgan_new as inpainting
import copy
import torch.nn as nn
import torch

# Code inspired from https://github.com/ci-ber/PHANES/

class Framework(nn.Module):
    def __init__(self, n, z_dim, method, device, model, ga, ga_n, th=99, cGAN = False, BOE_form = 'BOE'):
        super(Framework, self).__init__()
        self.z = z_dim
        self.ga = ga
        self.method = method
        self.th = th
        # print(f'{z_dim=}')
        # print(f'{ga_n=}')

        self.anomap = Anomaly(device)

        if ga:
            # from models.ga_vae import Encoder, Decoder
            from models.SI_VAE import Encoder, Decoder, Discriminator
            self.encoder = Encoder(n, n, z_dim, method, model=model, ga_n=ga_n, BOE_form = BOE_form)
            self.decoder = Decoder(BOE_size=ga_n, BOE_form = BOE_form)
            self.refineD = Discriminator(BOE_size=ga_n, BOE_form = BOE_form)
        else:
            raise NameError("Missing GA")

    def decode(self, z):
        y = self.decoder(z)
        return y
    
    def encode(self, x, ga = None):
        z_sample, mu, logvar, embed_dict = self.encoder(x, ga)
        return z_sample, mu, logvar, embed_dict

    def sample(self, z):
        y = self.decode(z)
        return y
    
    def ae(self, x, ga = None, deterministic=False):
        z, mu, logvar, embed_dict = self.encode(x, ga)
        y = self.decode(z).detach()
        return y, {'z_mu': mu, 'z_logvar': logvar,'z': z, 'embeddings': embed_dict['embeddings']}

    def forward(self, x_im, x_ga = None):
        z, _, _, _ = self.encoder(x_im, x_ga)
        x_recon = self.decoder(z)
        return x_recon