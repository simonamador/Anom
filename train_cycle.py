# Code adapted based on https://github.com/ci-ber/PHANES and https://github.com/researchmm/AOT-GAN-for-Inpainting
# Code written by  @GuillermoTafoya & @simonamador

import torch
from torch.nn import DataParallel
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from models.framework import Framework
from utils.config import loader, load_model
from utils import loss as loss_lib
from utils.debugging_printers import *
from utils.BOE import *


from models.csgan.cycle_GAN import CycleGANModel

from time import time
import copy

import os
from bunch_py3 import *

class Trainer:
    def __init__(self, parameters):
        
        # Determine if model inputs GA
        if parameters['VAE_model_type'] == 'ga_VAE':
            self.ga = True
            print('-'*50)
            print('')
            print('Training GA Model.')
            print('')
        else:
            self.ga = False
            print('-'*50)
            print('')
            print('Training default Model.')
            print('')

        self.device = parameters['device']
        #self.model_type = parameters['model']
        self.model_path = parameters['model_path']  
        self.tensor_path = parameters['tensor_path'] 
        self.image_path = parameters['image_path']  
        self.th = parameters['th'] if parameters['th'] else 99
        print(f'{self.ga=}')
        print(f'{parameters["ga_n"]=}')
        self.ga_n = parameters['ga_n'] if parameters['ga_n'] else None

        # Generate model
        self.model = Framework(parameters['slice_size'], parameters['z_dim'], 
                               parameters['ga_method'], parameters['device'], 
                               parameters['type'], self.ga, 
                               parameters['ga_n'], th=self.th, BOE_form = parameters['BOE_type'])

        # Load pre-trained parameters
        if parameters['pretrained'] == 'base':
            encoder, decoder = load_model(parameters['pretrained_path'], parameters['VAE_model_type'], 
                                          parameters['ga_method'], parameters['slice_size'], 
                                          parameters['slice_size'], parameters['z_dim'], 
                                          model=parameters['type'], pre = parameters['pretrained'], 
                                          ga_n = parameters['ga_n'],  BOE_form = parameters['BOE_type'])
            self.model.encoder = encoder
            self.model.decoder = decoder

        prGreen('Model successfully instanciated...')
        self.pre = parameters['pretrained']

        
        self.z_dim = parameters['z_dim']
        self.batch = parameters['batch']

        ### VAE ADVERSARIAL LOSS ### TODO
        
        prGreen('Losses successfully loaded...')

        # Establish data loaders
        train_dl, val_dl = loader(parameters['source_path'], parameters['view'], 
                                  parameters['batch'], parameters['slice_size'], 
                                  raw = parameters['raw'])
        self.loader = {"tr": train_dl, "ts": val_dl}
        prGreen('Data loaders successfully loaded...')
        
        # Optimizers
        self.optimizer_e = optim.Adam(self.model.encoder.parameters(), lr=1e-4, weight_decay=1e-5) # lr=1e-5, weight_decay=1e-6)
        self.optimizer_d = optim.Adam(self.model.decoder.parameters(), lr=1e-4, weight_decay=1e-5)  # lr=1e-5, weight_decay=1e-6) 
        # self.optimizer_netG = optim.Adam(self.model.refineG.parameters(), lr=5.0e-5)
        # self.optimizer_netD = optim.Adam(self.model.refineD.parameters(), lr=5.0e-5)

        opt = Bunch({
            'lambda_identity': 0.1,     # First, try using identity loss `--lambda_identity 1.0` or `--lambda_identity 0.1`. 
                                            # We observe that the identity loss makes the generator to be more conservative and make fewer unnecessary changes. 
                                            # However, because of this, the change may not be as dramatic.
                                            # use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. 
                                            # For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, 
                                            # please set lambda_identity = 0.1
            'input_nc': 1,              # # of input image channels: 3 for RGB and 1 for grayscale
            'output_nc': 1,             # # of output image channels: 3 for RGB and 1 for grayscale
            'ngf': 64,                  # # of gen filters in the last conv layer
            'netG': 'resnet_9blocks',   # specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128] 
            'norm': 'instance',         # instance normalization or batch normalization [instance | batch | none]
            'no_dropout': True,         # no dropout for the generator
            'init_type': 'normal',      # network initialization [normal | xavier | kaiming | orthogonal]
            'init_gain': 0.02,          # scaling factor for normal, xavier and orthogonal.
            'gpu_ids': [0,1,2],         # Please set`--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g., `--batch_size 32`) to benefit from multiple GPUs.
            'ndf': 64,                  # # of discrim filters in the first conv layer
            'netD': 'basic',            # specify discriminator architecture [basic | n_layers | pixel]. 
                                            # The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
            'n_layers_D': 3,            # only used if netD==n_layers
            'pool_size': 50,            # the size of image buffer, if pool_size=0, no buffer will be created
            'gan_mode': 'lsgan',        # the type of GAN objective. [vanilla| lsgan | wgangp]. 
                                            # vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
            'direction': 'AtoB',        # AtoB or BtoA
            'lr': 0.0002,               # initial learning rate for adam
            'beta1': 0.5,               # momentum term of adam 
                                            #     parser.add_argument('--lambda_identity', type=float, default=0.5, help='')
            'lambda_A': 10.0,           # weight for cycle loss (A -> B -> A)
            'lambda_B': 10.0,           # weight for cycle loss (B -> A -> B)
            'isTrain': True,
            'checkpoints_dir': 'Test',
            'name': 'Test',
            'continue_train': False,
            'load_iter': 0,             # which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]
            'lr_policy': 'linear',      # learning rate policy. [linear | step | plateau | cosine]
            'epoch_count': 1,           # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
            'n_epochs': 600,            # number of epochs with the initial learning rate
            'n_epochs_decay': 400,      # number of epochs to linearly decay learning rate to zero
            'verbose': True,
            'preprocess': 'crop', # scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
            'load_size': 160,
            'crop_size': 160
            }
        )

        ### Cycle GAN ###
        self.cycle_GAN = CycleGANModel(opt)
        self.cycle_GAN.setup(opt) 

        self.scale = 1 / (parameters['slice_size'] ** 2)  # normalize by images size (channels * height * width)
        self.gamma_r = 1e-8
        self.beta_kl = parameters['beta_kl'] if 'beta_kl' in parameters.keys() else 1.0
        self.beta_rec = parameters['beta_rec'] if 'beta_rec' in parameters.keys() else 0.5
        self.beta_neg = parameters['beta_neg'] if 'beta_neg' in parameters.keys() else self.z_dim // 2 + self.ga_n
        self.masking_threshold_train = parameters['masking_threshold_train'] if 'masking_threshold_train' in \
                                                                          parameters.keys() else None
        self.masking_threshold_inference = parameters['masking_threshold_infer'] if 'masking_threshold_infer' in \
                                                                          parameters.keys() else None

        self.base_loss = {'L2': loss_lib.l2_loss, 'L1': loss_lib.l1_loss, 'SSIM': loss_lib.ssim_loss, 
                     'MS_SSIM': loss_lib.ms_ssim_loss}
        self.loss_keys = {'L1': 1, 'Style': 250, 'Perceptual': 0.1}
        self.losses = {'L1':loss_lib.l1_loss,
                'Style':loss_lib.Style(),
                'Perceptual':loss_lib.Perceptual()}
        self.adv_loss = loss_lib.smgan()
        self.adv_weight = 0.01

        self.embedding_loss = loss_lib.EmbeddingLoss()
        # super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        print(f'{parameters["slice_size"]=}')
        prGreen('Optimizers successfully loaded...')

        self.BOE_type = parameters['BOE_type']

    def train(self, epochs, b_loss):
        
        # Training Loader
        current_loader = self.loader["tr"]
        
        # Create logger
        self.writer = open(self.tensor_path, 'w')
        self.writer.close()
        self.writer = open(self.tensor_path, 'a')
        self.writer.write('Epoch, tr_ed, tr_g, tr_d, v_ed, v_g, v_d, SSIM, MSE, MAE, Anomaly'+'\n')

        self.best_loss = 10000 # Initialize best loss (to identify the best-performing model)


        # Trains for all epochs
        for epoch in range(epochs):

              
            
            # Initialize models in device
            encoder = DataParallel(self.model.encoder).to(self.device).train()
            decoder = DataParallel(self.model.decoder).to(self.device).train()
            # cycle_GAN = DataParallel(self.cycle_GAN).to(self.device).train()

            # print('-'*15)
            # print(f'epoch {epoch+1}/{epochs}')

            epoch_ed_loss, epoch_refineG_loss, epoch_refineD_loss = 0.0, 0.0, 0.0

            start_time = time()

            diff_kls, batch_kls_real, batch_kls_fake, batch_kls_rec, batch_rec_errs, batch_exp_elbo_f,\
            batch_exp_elbo_r, batch_emb, count_images = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            batch_netGD_rec, batch_netG_loss, batch_netD_loss = 0.0, 0.0, 0.0

            # Runs through loader
            for data in current_loader:

                images = data['image'].to(self.device)
                # print(f'{images.shape=}')
                
                # transformed_images = copy.deepcopy(images)

                ga = data['ga'].to(self.device) if self.ga else None
                
                count_images += self.batch 
                encoded_ga = create_bi_partitioned_ordinal_vector(ga, self.ga_n) if self.ga_n else None
                noise_batch = torch.randn(size=(self.batch, self.z_dim//2)).to(self.device) 
                noise_batch = torch.cat((noise_batch,encoded_ga), 1)
                real_batch = images.to(self.device)
                # print(f'{real_batch.shape=}')
                # print(f'{noise_batch.shape=}')

                

                # =========== Update E ================
                if self.pre is None or self.pre == 'refine':
                    for param in self.model.encoder.parameters():
                        param.requires_grad = True
                    for param in self.model.decoder.parameters():
                        param.requires_grad = False
                    for param in self.model.refineG.parameters():
                        param.requires_grad = False
                    for param in self.model.refineD.parameters():
                        param.requires_grad = False

                    fake = self.model.sample(noise_batch)
                    
                    z, real_mu, real_logvar, anomaly_embeddings = self.model.encode(real_batch, ga)

                    # Reconstruct image
                    rec = self.model.decoder(z)
                    
                    #z, real_mu, real_logvar, anomaly_embeddings = self.model.encode(real_batch, ga)
                    _, _, _, healthy_embeddings = self.model.encode(rec.detach(), ga)
                
                    loss_emb = self.embedding_loss(anomaly_embeddings['embeddings'], healthy_embeddings['embeddings'])

                    loss_rec = loss_lib.calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")
                    lossE_real_kl = loss_lib.calc_kl(real_logvar, real_mu, reduce="mean")
                    rec_rec, z_dict = self.model.ae(rec.detach(), deterministic=False, ga=ga)
                    rec_mu, rec_logvar, z_rec = z_dict['z_mu'], z_dict['z_logvar'], z_dict['z']
                    rec_fake, z_dict_fake = self.model.ae(fake.detach(), deterministic=False, ga=ga)
                    fake_mu, fake_logvar, z_fake = z_dict_fake['z_mu'], z_dict_fake['z_logvar'], z_dict_fake['z']

                    kl_rec = loss_lib.calc_kl(rec_logvar, rec_mu, reduce="none")
                    kl_fake = loss_lib.calc_kl(fake_logvar, fake_mu, reduce="none")

                    loss_rec_rec_e = loss_lib.calc_reconstruction_loss(rec, rec_rec, loss_type="mse", reduction='none')
                    while len(loss_rec_rec_e.shape) > 1:
                        loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                    loss_rec_fake_e = loss_lib.calc_reconstruction_loss(fake, rec_fake, loss_type="mse", reduction='none')
                    while len(loss_rec_fake_e.shape) > 1:
                        loss_rec_fake_e = loss_rec_fake_e.sum(-1)

                    expelbo_rec = (-2 * self.scale * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)).exp().mean()
                    expelbo_fake = (-2 * self.scale * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)).exp().mean()

                    lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                    lossE_real = self.scale * (self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl) # ELBO

                    # lossE = lossE_real + lossE_fake + 0.005 * loss_emb     lambda = 0.005
                    lossE = lossE_real + lossE_fake + 0.01 * loss_emb
                    self.optimizer_e.zero_grad()
                    lossE.backward()
                    self.optimizer_e.step()

                    # ========= Update D ==================
                    for param in self.model.encoder.parameters():
                        param.requires_grad = False
                    for param in self.model.decoder.parameters():
                        param.requires_grad = True
                    for param in self.model.refineG.parameters():
                        param.requires_grad = False
                    for param in self.model.refineD.parameters():
                        param.requires_grad = False

                    fake = self.model.sample(noise_batch)
                    rec = self.model.decoder(z.detach())
                    loss_rec = loss_lib.calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")

                    z_rec, rec_mu, rec_logvar,_ = self.model.encode(rec, ga)

                    z_fake, fake_mu, fake_logvar,_ = self.model.encode(fake, ga)

                    rec_rec = self.model.decode(z_rec.detach())
                    rec_fake = self.model.decode(z_fake.detach())

                    loss_rec_rec = loss_lib.calc_reconstruction_loss(rec.detach(), rec_rec, loss_type="mse", reduction="mean")
                    loss_fake_rec = loss_lib.calc_reconstruction_loss(fake.detach(), rec_fake, loss_type="mse", reduction="mean")

                    lossD_rec_kl = loss_lib.calc_kl(rec_logvar, rec_mu, reduce="mean")
                    lossD_fake_kl = loss_lib.calc_kl(fake_logvar, fake_mu, reduce="mean")

                    lossD = self.scale * (loss_rec * self.beta_rec + (
                            lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl + self.gamma_r * 0.5 * self.beta_rec * (
                                                loss_rec_rec + loss_fake_rec))

                    self.optimizer_d.zero_grad()
                    lossD.backward()
                    self.optimizer_d.step()
                    if torch.isnan(lossD) or torch.isnan(lossE):
                        print('is non for D')
                        raise SystemError
                    if torch.isnan(lossE):
                        print('is non for E')
                        raise SystemError
                    
                    # ====================================
                    diff_kls += -lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item() * images.shape[0]
                    batch_kls_real += lossE_real_kl.data.cpu().item() * images.shape[0]
                    batch_kls_fake += lossD_fake_kl.cpu().item() * images.shape[0]
                    batch_kls_rec += lossD_rec_kl.data.cpu().item() * images.shape[0]
                    batch_rec_errs += loss_rec.data.cpu().item() * images.shape[0]

                    batch_exp_elbo_f += expelbo_fake.data.cpu() * images.shape[0]
                    batch_exp_elbo_r += expelbo_rec.data.cpu() * images.shape[0]

                    batch_emb += loss_emb.cpu().item() * images.shape[0]
                    
                else:
                    z, real_mu, real_logvar, anomaly_embeddings = self.model.encode(real_batch, ga)
                    rec = self.model.decoder(z)
                    diff_kls = -1
                    batch_kls_real = -1
                    batch_kls_fake = -1
                    batch_kls_rec = -1
                    batch_rec_errs = -1
                    batch_exp_elbo_f= -1
                    batch_exp_elbo_r= -1
                    batch_emb= -1

                # ------ Update Refine Model ------

                if self.pre is None or self.pre == 'base': 

                    padding = 1
                    real_batch_padded = F.pad(real_batch, (padding, padding, padding, padding), "constant", 0)
                    rec_padded = F.pad(rec.detach(), (padding, padding, padding, padding), "constant", 0)

                    self.cycle_GAN.set_input(rec_padded, real_batch_padded, 'AtoB')         # unpack data from dataset and apply preprocessing
                    self.cycle_GAN.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if self.pre is None or self.pre == 'base': self.cycle_GAN.update_learning_rate()

                    
                    

            # Testing
            test_dic = self.test(b_loss)
            val_loss = test_dic["losses"] 
            images = test_dic["images"] 

 


            end_time = time()
            print('Epoch: {} \t , computed in {} seconds for {} samples'.format(
                epoch, end_time - start_time, count_images))

            cycle_losses = self.cycle_GAN.get_current_losses()
            message = ''
            for k, v in cycle_losses.items():
                message += '%s: %.3f ' % (k, v)

            # Assuming you have variables `current_epoch`, `total_epochs`, `current_val_loss`, and `images` defined:
            self.log(epoch=epoch, epochs=epochs, tr_losses=message, images=images, val_losses=val_loss)

            #self.log(epoch, epochs, [epoch_ed_loss, epoch_refineG_loss, epoch_refineD_loss] , val_loss, metrics, images, pretrained = self.pre)

        self.writer.close()

    def test(self, b_loss):
        # Setting model for evaluation
        self.model.eval()
        self.cycle_GAN.eval()

        base_loss, refineG_loss, refineD_loss = 0.0, 0.0, 0.0
        mse_loss, mae_loss, ssim, anom = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for data in self.loader["ts"]:
                real_batch = data['image'].to(self.device)
                ga = data['ga'].to(self.device) if self.ga else None

                # Run the whole framework forward, no need to do each component separate
                
                # _, res_dic = self.model(real_batch, ga)
                z, _, _, _ = self.model.encode(real_batch, ga)
                rec = self.model.decoder(z)

                # Calc the losses

                #   encoder-decoder loss
                # ed_loss = self.base_loss[b_loss](res_dic["x_recon"],real_batch)

                #   refinement loss
                losses = {}
                for name, weight in self.loss_keys.items():
                    losses[name] = weight * self.losses[name](rec, real_batch)

                padding = 1
                real_batch_padded = F.pad(real_batch, (padding, padding, padding, padding), "constant", 0)
                rec_padded = F.pad(rec, (padding, padding, padding, padding), "constant", 0)

                self.cycle_GAN.set_input(rec_padded, real_batch_padded, 'AtoB') 
                self.cycle_GAN.test() 
                visuals = self.cycle_GAN.get_current_visuals() 
                cycle_losses = self.cycle_GAN.get_current_losses()
                message = ''
                for k, v in cycle_losses.items():
                    message += '%s: %.3f ' % (k, v)

            base_loss /= len(self.loader["ts"])            

            # Images dic for visualization
            images = {"real_A": visuals['real_A'][0], "fake_B": visuals['fake_B'][0], "rec_A": visuals['rec_A'][0],
                      "real_B": visuals['real_B'][0], "fake_A": visuals['fake_A'][0], "rec_B": visuals['rec_B'][0]
                    }    
        
        return {'losses': [message], 'images': images}

    def log(self, epoch, epochs, tr_losses, images, val_losses):
        # Format the new losses for logging
        # formatted_losses = ', '.join([f'{key}: {(value.item() if isinstance(value, torch.Tensor) else value):.4f}' for key, value in losses.items()])
        # header = 'Epoch, ED Loss, RefineG Loss, RefineD Loss, MSE Loss, MAE Loss, SSIM, Anom\n'
        log_file_path = f'{self.model_path}/training_log.csv'

        if not os.path.exists(log_file_path) or os.stat(log_file_path).st_size == 0:
            with open(log_file_path, 'w') as file:
                file.write('')
                
        # primary_val_loss = val_loss[0] 
        print(f'Epoch {epoch+1}')
        print(f'Training losses: {tr_losses=}')


        components_ED = ['encoder', 'decoder']
        components_Cycle = ['netG_A', 'netG_B', 'netD_A', 'netD_B']

        for component in components_ED:
            torch.save({
                'epoch': epoch + 1,
                component: getattr(self.model, component).state_dict(),
            }, f'{self.model_path}/{component}_latest.pth')

        
        for component in components_Cycle:
            torch.save({
                'epoch': epoch + 1,
                component: getattr(self.cycle_GAN, component).state_dict(),
            }, f'{self.model_path}/{component}_latest.pth')

        # Save and plot model components every n epochs or in the first or last epoch
        if (epoch == 0) or ((epoch + 1) % 10 == 0) or ((epoch + 1) == epochs):
            for component in components_ED:
                torch.save({'epoch': epoch + 1, component: getattr(self.model, component).state_dict()},
                        f'{self.model_path}/{component}_{epoch + 1}.pth')
            for component in components_Cycle:
                torch.save({
                    'epoch': epoch + 1,
                    component: getattr(self.cycle_GAN, component).state_dict(),
                }, f'{self.model_path}/{component}_{epoch + 1}.pth')
            
            # Plot and save the progress image
            progress_im = self.plot(images)
            progress_im.savefig(f'{self.image_path}epoch_{epoch+1}.png')

        

        log_entry = f'{epoch+1}, Tr Losses: {tr_losses}, Val Losses: {val_losses}\n'
        with open(log_file_path, 'a') as file:
            file.write(log_entry)

        


    def plot(self, images):
  
        fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        names = [["real_A", "fake_B", "rec_A"], 
                 ["real_B", "fake_A", "rec_B"]]

        for x in range(2):
            for y in range(3):
                axs[x, y].imshow(images[names[x][y]].detach().cpu().numpy().squeeze(), cmap='gray')
                axs[x, y].set_title(names[x][y])
                axs[x, y].axis("off")
        plt.tight_layout()
        return fig