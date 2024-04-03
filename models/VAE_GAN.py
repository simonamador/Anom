from models.aotgan.common import BaseNetwork
from torch.nn.utils import spectral_norm
import torch.distributions as dist
import torch.nn.functional as F
from models.vae import Basic
import torch.nn as nn
import torch


# Author: @GuillermoTafoya & @simonamador
# The following code builds an autoencoder model for unsupervised learning applications in MRI anomaly detection.


def calculate_ga_index(ga, size, min_GA = 20, max_GA = 40):
        # Map GA to the nearest increment starting from 20 (assuming a range of 20-40 GA)
        increment = (max_GA - min_GA)/size
        ga_mapped = torch.round((ga - min_GA) / increment)
        return ga_mapped

def calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, min_GA = 20, max_GA = 40):
        ga_mapped = torch.round( (torch.tensor(size) / (torch.exp(torch.tensor(a + b)))) * 
                                (torch.exp(torch.tensor(a) + torch.tensor(b) * ((ga - min_GA) /  (max_GA - min_GA)) )) )
        return ga_mapped 

def inv_calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, min_GA = 20, max_GA = 40):
        ga_mapped = torch.round( (torch.tensor(size) / (torch.exp(torch.tensor(a + b)))) * 
                                (torch.exp(torch.tensor(a) + torch.tensor(b) * ((-ga + max_GA) /  (max_GA - min_GA)) )) )
        return ga_mapped  

def inv_inv_calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, min_GA = 20, max_GA = 40):
        ψ = calculate_ga_index_exp(40, size, a, b, min_GA, max_GA )
        ga_mapped = - inv_calculate_ga_index_exp(ga, size, a, b, min_GA, max_GA) + ψ
        return ga_mapped 

BOE_forms = {
            'BOE': calculate_ga_index,
            'EBOE': calculate_ga_index_exp,
            'inv_BOE': inv_calculate_ga_index_exp,
            'inv_inv_BOE': inv_inv_calculate_ga_index_exp
        }

def create_bi_partitioned_ordinal_vector(gas, size, BOE_form='BOE'):
        threshold_index = size//2
        device = gas.device
        batch_size = gas.size(0)
        ga_indices= BOE_forms[BOE_form](gas, size)
        vectors = torch.full((batch_size, size), -1, device=device)
        for i in range(batch_size):
            idx = ga_indices[i].long()
            if idx > size:
                idx = size
            elif idx < 0:
                idx = 1
            if idx >= threshold_index:
                new_idx = (idx-threshold_index)*2
                vectors[i, :new_idx] = 1
                vectors[i, new_idx:] = 0
            else:
                new_idx = idx*2
                vectors[i, :new_idx] = 0
        return vectors


# Encoder class builds encoder model depending on the model type.
# Inputs: H, y (x and y size of the MRI slice),z_dim (length of the output z-parameters), model (the model type)
class Encoder(nn.Module):
    def __init__(
            self, 
            h,
            w,
            z_dim,
            method,
            model: str = 'default',
            ga_n = 100,
            BOE_form = 'BOE'
        ):

        method_type = ['bpoe']

        if method not in method_type:
            raise ValueError('Invalid method to include. Expected: %s' % method_type)

        ch = 16
        k_size = 4
        stride = 2
        self.method = method
        self.model = model
        self.size = ga_n

        super(Encoder,self).__init__()

        self.step0 = Basic(1,ch,k_size=k_size, stride=stride)

        self.step1 = Basic(ch,ch * 2, k_size=k_size, stride=stride)
        self.step2 = Basic(ch * 2,ch * 4, k_size=k_size, stride=stride)
        self.step3 = Basic(ch * 4,ch * 8, k_size=k_size, stride=stride)

        n_h = int(((h-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        n_w = int(((w-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        self.flat_n = n_h * n_w * ch * 8
        self.linear = nn.Linear(self.flat_n,z_dim)

        self.BOE_form = BOE_forms[BOE_form]

    def forward(self,x,ga):
        
        if self.size and self.method == 'bpoe':
            ga = create_bi_partitioned_ordinal_vector(ga)
        
        embeddings = []

        x = self.step0(x)
        embeddings.append(x)
        x = self.step1(x)
        embeddings.append(x)
        x = self.step2(x)
        embeddings.append(x)
        x = self.step3(x)
        embeddings.append(x)

        x = x.view(-1, self.flat_n)

        z_params = self.linear(x)
        
        mu, log_std = torch.chunk(z_params, 2, dim=1)

        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)

        z_sample = z_dist.rsample()

        if self.size and self.method in ['bpoe']:
            z_sample = torch.cat((z_sample,ga), 1)

        if self.model == 'bVAE':
            return z_sample, mu, log_std
        
        return z_sample, mu, log_std, {'embeddings': embeddings}

# Code from https://github.com/researchmm/AOT-GAN-for-Inpainting.git

class Decoder(BaseNetwork):
    def __init__(self, rates='1+2+4+8', block_num=8, BOE_size=0, BOE_form = 'BOE'):
        nr_channels = 1
        rates=[1, 2, 4, 8]
        super(Decoder, self).__init__()

        self.BOE_size = BOE_size

        self.aot = nn.Sequential(*[AOTBlock(256+self.BOE_size, rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            # nn.Linear(z_dim, self.z_develop)
            UpConv(256+self.BOE_size, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, nr_channels, 3, stride=1, padding=1)
        )

        self.init_weights()

        self.BOE_form = BOE_form

    def forward(self, x):
        x = self.aot(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            # print(rate)
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        # print(out.shape)
        out = self.fuse(out)
        # print(out.shape)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


""" TODO
# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(self,  BOE_size=0, BOE_form = 'BOE'):
        super(Discriminator, self).__init__()
        self.BOE_size = BOE_size
        self.inc = 1  # Assuming grayscale images, change if different
        # Additional layers for processing GA
        self.ga_embedding = nn.Linear(self.BOE_size, 158 * 158)  # Project GA into a space that can be reshaped into a spatial form
        self.ga_conv = nn.Conv2d(1, 64, 3, stride=1, padding=1)  # Convolve GA to integrate into image features
        
        # Original conv layers
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(self.inc + 64, 64, 4, stride=2, padding=1, bias=False)),  # Adjust input channels
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

        self.init_weights()

        self.BOE_form = BOE_form

    def forward(self, x, ga=None):
        # If GA is provided, process and integrate it
        if self.BOE_size and ga is not None:
            # Encode GA and reshape into spatial dimensions
            encoded_ga = create_bi_partitioned_ordinal_vector(ga, self.BOE_size, self.BOE_form)
            encoded_ga = encoded_ga.float() # Convert encoded GA to float dtype to match layer weights
            # Project encoded GA to match discriminator feature map size and reshape
            encoded_ga = self.ga_embedding(encoded_ga)  # Embed GA into a larger space
            encoded_ga = encoded_ga.view(-1, 1, 158, 158)  # Reshape to form a single-channel spatial map
            encoded_ga = F.relu(self.ga_conv(encoded_ga))  # Convolve the GA map to integrate into image feature dimensions
            
            # Concatenate the GA map with the input image
            x = torch.cat([x, encoded_ga], dim=1)  # Combine along channel dimension

        # Process through convolutional layers
        img_features = self.conv(x)
        return img_features
    


# Decoder class builds decoder model depending on the model type.
# Inputs: H, y (x and y size of the MRI slice),z_dim (length of the input z-vector), model (the model type) 
# Note: z_dim in Encoder is not the same as z_dim in Decoder, as the z_vector has half the size of the z_parameters.
class Decoder(nn.Module):
    def __init__(
            self, 
            h, 
            w, 
            z_dim, 
            ):
        super(Decoder, self).__init__()

        self.ch = 16
        self.k_size = 4
        self.stride = 2
        self.hshape = int(((h-self.k_size)/(self.stride**4)) - (self.k_size-1)/(self.stride**3) - (self.k_size-1)/(self.stride**2) - (self.k_size-1)/self.stride + 1)
        self.wshape = int(((w-self.k_size)/(self.stride**4)) - (self.k_size-1)/(self.stride**3) - (self.k_size-1)/(self.stride**2) - (self.k_size-1)/self.stride + 1)

        self.z_develop = self.hshape * self.wshape * 8 * self.ch
        self.linear = nn.Linear(z_dim, self.z_develop)
        self.step1 = Basic(self.ch* 8, self.ch * 4, k_size=self.k_size, stride=self.stride, transpose=True)
        self.step2 = Basic(self.ch * 4, self.ch * 2, k_size=self.k_size, stride=self.stride, transpose=True)
        self.step3 = Basic(self.ch * 2, self.ch, k_size=self.k_size, stride=self.stride, transpose=True)        
        self.step4 = Basic(self.ch, 1, k_size=self.k_size, stride=self.stride, transpose=True)
        self.activation = nn.ReLU()

    def forward(self,z):
        x = self.linear(z)
        x = x.view(-1, self.ch * 8, self.hshape, self.wshape)
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        recon = self.activation(x)
        return recon
    
"""