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
        # self.refineG = inpainting.InpaintGenerator().to(device)#BOE_size=0).to(device)
        # self.refineD = inpainting.Discriminator().to(device)#BOE_size=0).to(device)
        self.refineG = inpainting.InpaintGenerator(BOE_size=200, BOE_form = BOE_form)#BOE_size=0)
        self.refineD = inpainting.Discriminator(BOE_size=200, BOE_form = BOE_form)#BOE_size=0)

        if ga:
            # from models.ga_vae import Encoder, Decoder
            from models.SI_VAE import Encoder, Decoder
            self.encoder = Encoder(n, n, z_dim, method, model=model, ga_n=ga_n, BOE_form = BOE_form)
            self.decoder = Decoder(n, n, int(z_dim/2) + (ga_n if method in ['ordinal_encoding','one_hot_encoding', 'bpoe'] else 0))
        else:
            from models.vae import Encoder, Decoder
            self.encoder = Encoder(n, n, z_dim, model = model)
            self.decoder = Decoder(n, n, int(z_dim/2), model = model)

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
        y_ref = self.refineG(x_recon, x_ga)
        y_ref = torch.clamp(y_ref, 0, 1)
        y_ref = self.anomap.zero_pad(y_ref, x_im.shape[2])
        y_fin = y_ref

        return y_fin, {"x_recon": x_recon, "y_ref": y_ref}