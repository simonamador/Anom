# Code based on https://github.com/ci-ber/PHANES and https://github.com/researchmm/AOT-GAN-for-Inpainting

import monai.losses as losses 
import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter

from utils.vgg import VGG19

import torch.nn as nn
import torchvision
from torch.nn.modules.loss import _Loss


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error

def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


class VGGEncoder(nn.Module):
    """
    VGG Encoder used to extract feature representations for e.g., perceptual losses
    """
    def __init__(self, layers=[1, 6, 11, 20]):
        super(VGGEncoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.encoder = nn.ModuleList()
        temp_seq = nn.Sequential()
        for i in range(max(layers) + 1):
            temp_seq.add_module(str(i), vgg[i])
            if i in layers:
                self.encoder.append(temp_seq)
                temp_seq = nn.Sequential()

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features




class EmbeddingLoss(torch.nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, teacher_embeddings, student_embeddings):
        # print(f'LEN {len(output_real)}')
        layer_id = 0
        for teacher_feature, student_feature in zip(teacher_embeddings, student_embeddings):
            if layer_id == 0:
                total_loss = 0.5 * self.criterion(teacher_feature, student_feature)
            else:
                total_loss += 0.5 * self.criterion(teacher_feature, student_feature)
            total_loss += torch.mean(1 - self.similarity_loss(teacher_feature.view(teacher_feature.shape[0], -1),
                                                         student_feature.view(student_feature.shape[0], -1)))
            layer_id += 1
        return total_loss


class PerceptualLoss(_Loss):
    """
    """

    def __init__(
        self,
        reduction: str = 'mean',
        device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction
        self.loss_network = VGGEncoder().eval().to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        input_features = self.loss_network(input.repeat(1, 3, 1, 1)) if input.shape[1] == 1 else input
        output_features = self.loss_network(target.repeat(1, 3, 1, 1)) if target.shape[1] == 1 else target

        loss_pl = 0
        for output_feature, input_feature in zip(output_features, input_features):
            loss_pl += F.mse_loss(output_feature, input_feature)
        return loss_pl

def kld_loss(mu, log_var):
    kld = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    return kld
    
l1_loss = nn.L1Loss()

l1_error = nn.L1Loss(reduction="none")

l2_loss = nn.MSELoss()

ssim_loss = losses.SSIMLoss(spatial_dims=2)

ms_ssim_loss = losses.MultiScaleLoss(loss = losses.SSIMLoss(spatial_dims = 2, win_size = 5),
                                     scales = [0.5, 1.0, 2.0, 4.0, 8.0])

class Perceptual(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss

class Style(nn.Module):
    def __init__(self):
        super(Style, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f'relu{pre}_{pos}']), self.compute_gram(y_vgg[f'relu{pre}_{pos}']))
        return style_loss
    


# class GaussianBlur(nn.Module):
#     r"""Creates an operator that blurs a tensor using a Gaussian filter.
#     The operator smooths the given tensor with a gaussian kernel by convolving
#     it to each channel. It suports batched operation.
#     Arguments:
#       kernel_size (Tuple[int, int]): the size of the kernel.
#       sigma (Tuple[float, float]): the standard deviation of the kernel.
#     Returns:
#       Tensor: the blurred tensor.
#     Shape:
#       - Input: :math:`(B, C, H, W)`
#       - Output: :math:`(B, C, H, W)`

#     Examples::
#       >>> input = torch.rand(2, 4, 5, 5)
#       >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
#       >>> output = gauss(input)  # 2x4x5x5
#     """

#     def __init__(self, kernel_size, sigma):
#         super(GaussianBlur, self).__init__()
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#         self._padding = self.compute_zero_padding(kernel_size)
#         self.kernel = get_gaussian_kernel2d(kernel_size, sigma)

#     @staticmethod
#     def compute_zero_padding(kernel_size):
#         """Computes zero padding tuple."""
#         computed = [(k - 1) // 2 for k in kernel_size]
#         return computed[0], computed[1]

#     def forward(self, x):  # type: ignore
#         if not torch.is_tensor(x):
#             raise TypeError(
#                 "Input x type is not a torch.Tensor. Got {}".format(type(x)))
#         if not len(x.shape) == 4:
#             raise ValueError(
#                 "Invalid input shape, we expect BxCxHxW. Got: {}".format(x.shape))
#         # prepare kernel
#         b, c, h, w = x.shape
#         tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
#         kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

#         # TODO: explore solution when using jit.trace since it raises a warning
#         # because the shape is converted to a tensor instead to a int.
#         # convolve tensor with gaussian kernel
#         return conv2d(x, kernel, padding=self._padding, stride=1, groups=c)


# ######################
# # functional interface
# ######################

# def gaussian_blur(input, kernel_size, sigma):
#     r"""Function that blurs a tensor using a Gaussian filter.
#     See :class:`~kornia.filters.GaussianBlur` for details.
#     """
#     return GaussianBlur(kernel_size, sigma)(input)


# class smgan():
#     def __init__(self, ksize=71): 
#         self.ksize = ksize
#         self.loss_fn = nn.MSELoss()
    
#     def __call__(self, netD, fake, real, masks, ga = None): 
#         fake_detach = fake.detach()

#         g_fake = netD(fake, ga)
#         d_fake  = netD(fake_detach, ga)
#         d_real = netD(real, ga)

#         _, _, h, w = g_fake.size()
#         b, c, ht, wt = masks.size()
        
#         # Handle inconsistent size between outputs and masks
#         if h != ht or w != wt:
#             g_fake = F.interpolate(g_fake, size=(ht, wt), mode='bilinear', align_corners=True)
#             d_fake = F.interpolate(d_fake, size=(ht, wt), mode='bilinear', align_corners=True)
#             d_real = F.interpolate(d_real, size=(ht, wt), mode='bilinear', align_corners=True)
#         # d_fake_label = torch.Tensor(gaussian_filter(masks.cpu().detach().numpy(), sigma=1.2)).cuda()
#         d_fake_label = gaussian_blur(masks, (self.ksize, self.ksize), (10, 10)).detach().cuda()
#         d_real_label = torch.zeros_like(d_real).cuda()
#         g_fake_label = torch.ones_like(g_fake).cuda()

#         dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
#         gen_loss = self.loss_fn(g_fake, g_fake_label) * masks / torch.mean(masks)

#         return dis_loss.mean(), gen_loss.mean()
    
class smgan():
    def __init__(self): 
        self.loss_fn = nn.MSELoss()
    
    def __call__(self, netD, fake, real, masks, ga = None): 
        fake_detach = fake.detach()

        g_fake = netD(fake, ga)
        d_fake  = netD(fake_detach, ga)
        d_real = netD(real, ga)

        _, _, h, w = g_fake.size()
        b, c, ht, wt = masks.size()
        
        # Handle inconsistent size between outputs and masks
        if h != ht or w != wt:
            g_fake = F.interpolate(g_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_fake = F.interpolate(d_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_real = F.interpolate(d_real, size=(ht, wt), mode='bilinear', align_corners=True)
        d_fake_label = torch.Tensor(gaussian_filter(masks.cpu().detach().numpy(), sigma=1.2)).cuda()
        d_real_label = torch.zeros_like(d_real).cuda()
        g_fake_label = torch.ones_like(g_fake).cuda()

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake, g_fake_label) * masks / torch.mean(masks)

        return dis_loss.mean(), gen_loss.mean()

class smgan_nomask():
    def __init__(self): 
        self.loss_fn = nn.MSELoss()
    
    def __call__(self, netD, fake, real, ga=None): 
        fake_detach = fake.detach()

        g_fake = netD(fake, ga)
        d_fake  = netD(fake_detach, ga)
        d_real = netD(real, ga)

        # No need to resize or apply masks to align sizes now
        d_fake_label = torch.ones_like(d_fake).cuda()
        d_real_label = torch.zeros_like(d_real).cuda()
        g_fake_label = torch.ones_like(g_fake).cuda()

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake, g_fake_label)

        return dis_loss.mean(), gen_loss.mean()
