# Code from https://github.com/researchmm/AOT-GAN-for-Inpainting.git
# cGAN adapted-ish

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from models.aotgan.common import BaseNetwork

#size = 200

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
        # Adjusting the threshold for the nearest 0.1 increment
        threshold_index = size//2
        device = gas.device
        batch_size = gas.size(0)
        # ga_indices = calculate_ga_index_exp(gas, size)
        ga_indices= BOE_forms[BOE_form](gas, size)
        vectors = torch.full((batch_size, size), -1, device=device)  # Default fill with -1

        for i in range(batch_size):
            idx = ga_indices[i].long()
            if idx > size:
                idx = size
            elif idx < 0:
                idx = 1
            

            if idx >= threshold_index:  # GA >= 30
                new_idx = (idx-threshold_index)*2
                vectors[i, :new_idx] = 1  # First 100 elements to 1 (up to GA == 30)
                vectors[i, new_idx:] = 0  # The rest to 0
            else:  # GA < 30
                new_idx = idx*2
                vectors[i, :new_idx] = 0  # First 100 elements to 0
                # The rest are already set to -1

        return vectors


class InpaintGenerator(BaseNetwork):
    def __init__(self, rates='1+2+4+8', block_num=8, BOE_size=0, BOE_form = 'BOE'):  # 1046
        nr_channels = 1
        rates=[1, 2, 4, 8]
        super(InpaintGenerator, self).__init__()

        self.BOE_size = BOE_size

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, 7), #############
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[AOTBlock(256+self.BOE_size, rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            UpConv(256+self.BOE_size, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, nr_channels, 3, stride=1, padding=1)
        )

        self.init_weights()

        self.BOE_form = BOE_form

    def forward(self, x, ga=None):
        if ga is None:
            batch_size = x.size(0)
            ga = torch.full((batch_size, 1), 30.0, dtype=torch.float32, device=x.device)

        x = self.encoder(x)

        

        if ga is not None:
            # Encode GA using your method
            encoded_ga = create_bi_partitioned_ordinal_vector(ga, self.BOE_size, self.BOE_form)
            # You may need to expand the dimensions to match x
            encoded_ga_expanded = encoded_ga.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
            # Concatenate encoded GA with the feature map
            x = torch.cat([x, encoded_ga_expanded], dim=1)

        x = self.middle(x)
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
        if ga is not None:
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
    
       
if __name__ == '__main__':
    from torchsummary import summary
    emodel = Encoder(158, 158, 512, 'bpoe', model='default', ga_n=100, BOE_form = 'BOE')
    dmodel = Decoder(BOE_size=100, BOE_form = 'BOE')
    summary(emodel, (1, 158, 158), device='cpu')
    summary(dmodel, (1, 356), device='cpu')