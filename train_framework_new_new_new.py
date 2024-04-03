# Code adapted based on https://github.com/ci-ber/PHANES and https://github.com/researchmm/AOT-GAN-for-Inpainting
# Code written by  @GuillermoTafoya & @simonamador

import torch
from torch.nn import DataParallel
import torch.optim as optim

import matplotlib.pyplot as plt

from models.framework_new import Framework
from utils.config import loader, load_model_new_new
from utils import loss as loss_lib
from utils.debugging_printers import *
from utils.BOE import *

from time import time
import copy

import os

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
            encoder, decoder = load_model_new(parameters['pretrained_path'], parameters['VAE_model_type'], 
                                          parameters['ga_method'], parameters['slice_size'], 
                                          parameters['slice_size'], parameters['z_dim'], 
                                          model=parameters['type'], pre = parameters['pretrained'], 
                                          ga_n = parameters['ga_n'],  BOE_form = parameters['BOE_type'])
            self.model.encoder = encoder
            self.model.decoder = decoder
        if parameters['pretrained'] == 'refine':
            refineG, refineD = load_model_new(parameters['pretrained_path'], parameters['VAE_model_type'], 
                                          parameters['ga_method'], parameters['slice_size'],
                                          parameters['slice_size'], parameters['z_dim'], 
                                          model=parameters['type'], pre = parameters['pretrained'],
                                          ga_n = parameters['ga_n'],  BOE_form = parameters['BOE_type'])
            self.model.refineG = refineG
            self.model.refineD = refineD
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
        self.optimizer_netG = optim.Adam(self.model.refineG.parameters(), lr=5.0e-5)
        self.optimizer_netD = optim.Adam(self.model.refineD.parameters(), lr=5.0e-5)

        # TODO
        # self.e_scheduler = MultiStepLR(self.optimizer_e, milestones=(100,), gamma=0.1)
        # self.d_scheduler = MultiStepLR(self.optimizer_d, milestones=(100,), gamma=0.1)
        # self.netG_scheduler = MultiStepLR(self.optimizer_netG, milestones=(100,), gamma=0.1)
        # self.netD_scheduler = MultiStepLR(self.optimizer_netD, milestones=(100,), gamma=0.1)

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
        self.adv_loss = loss_lib.smgan_nomask()
        self.adv_weight = 0.01

        self.embedding_loss = loss_lib.EmbeddingLoss()
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

        epoch_losses = []

        # Trains for all epochs
        for epoch in range(epochs):
            
            # Initialize models in device
            encoder = DataParallel(self.model.encoder).to(self.device).train()
            decoder = DataParallel(self.model.decoder).to(self.device).train()
            refineG = DataParallel(self.model.refineG).to(self.device).train()
            refineD = DataParallel(self.model.refineD).to(self.device).train()

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
                    lossE = lossE_real + lossE_fake + 0.01 * loss_emb # lambda = 0.01
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

                    for param in self.model.encoder.parameters():
                        param.requires_grad = False
                    for param in self.model.decoder.parameters():
                        param.requires_grad = False
                    for param in self.model.refineG.parameters():
                        param.requires_grad = True
                    for param in self.model.refineD.parameters():
                        param.requires_grad = True

                    y_ref = self.model.refineG(rec, ga)
                    y_ref = torch.clamp(y_ref, 0, 1)
                    zero_pad = torch.nn.ZeroPad2d(1)
                    y_ref = zero_pad(y_ref)
                    ref_recon = y_ref

                    # Losses for GAN
                    losses = {}
                    for name, weight in self.loss_keys.items():
                        losses[name] = weight * self.losses[name](y_ref, real_batch)

                    dis_loss, gen_loss = self.adv_loss(self.model.refineD, ref_recon, real_batch, ga)

                    losses['advg'] = gen_loss * self.adv_weight
                    
                    self.optimizer_netG.zero_grad()
                    self.optimizer_netD.zero_grad()
                    sum(losses.values()).backward()
                    dis_loss.backward()
                    self.optimizer_netG.step()
                    self.optimizer_netD.step()

                    batch_netGD_rec += sum(losses.values()).cpu().item() * images.shape[0]
                    batch_netG_loss += losses['advg'].cpu().item() * images.shape[0]
                    batch_netD_loss += dis_loss.cpu().item() * images.shape[0]

            # Testing
            test_dic = self.test(b_loss)
            val_loss = test_dic["losses"] 
            metrics = test_dic["metrics"] 
            images = test_dic["images"] 

            epoch_loss_d_kls = diff_kls / count_images if count_images > 0 else diff_kls
            epoch_loss_kls_real = batch_kls_real / count_images if count_images > 0 else batch_kls_real
            epoch_loss_kls_fake = batch_kls_fake / count_images if count_images > 0 else batch_kls_fake
            epoch_loss_kls_rec = batch_kls_rec / count_images if count_images > 0 else batch_kls_rec
            epoch_loss_rec_errs = batch_rec_errs / count_images if count_images > 0 else batch_rec_errs
            epoch_loss_exp_f = batch_exp_elbo_f / count_images if count_images > 0 else batch_exp_elbo_f
            epoch_loss_exp_r = batch_exp_elbo_r / count_images if count_images > 0 else batch_exp_elbo_r
            epoch_loss_emb = batch_emb / count_images if count_images > 0 else batch_emb
            epoch_loss_netGD_rec = batch_netGD_rec / count_images if count_images > 0 else batch_netGD_rec
            epoch_loss_netG_loss = batch_netG_loss / count_images if count_images > 0 else batch_netG_loss
            epoch_loss_netD_loss = batch_netD_loss / count_images if count_images > 0 else batch_netD_loss

            epoch_losses.append(epoch_loss_rec_errs)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss_rec_errs, end_time - start_time, count_images))
            
            print({"Train/Loss_DKLS": epoch_loss_d_kls, '_step_': epoch})
            print({"Train/Loss_REAL": epoch_loss_kls_real, '_step_': epoch})
            print({"Train/Loss_FAKE": epoch_loss_kls_fake, '_step_': epoch})
            print({"Train/Loss_REC": epoch_loss_kls_rec, '_step_': epoch})
            print({"Train/Loss_REC_ERRS": epoch_loss_rec_errs, '_step_': epoch})
            print({"Train/Loss_EXP_F": epoch_loss_exp_f, '_step_': epoch})
            print({"Train/Loss_EXP_R": epoch_loss_exp_r, '_step_': epoch})
            print({"Train/Loss_EMB": epoch_loss_emb, '_step_': epoch})
            print({"Train/Loss_netGD_REC": epoch_loss_netGD_rec, '_step_': epoch})
            print({"Train/Loss_netG": epoch_loss_netG_loss, '_step_': epoch})
            print({"Train/Loss_netD": epoch_loss_netD_loss, '_step_': epoch})

            losses = {
                "Loss_DKLS": epoch_loss_d_kls,
                "Loss_REAL": epoch_loss_kls_real,
                "Loss_FAKE": epoch_loss_kls_fake,
                "Loss_REC": epoch_loss_kls_rec,
                "Loss_REC_ERRS": epoch_loss_rec_errs,
                "Loss_EXP_F": epoch_loss_exp_f,
                "Loss_EXP_R": epoch_loss_exp_r,
                "Loss_EMB": epoch_loss_emb,
                "Loss_netGD_REC": epoch_loss_netGD_rec,
                "Loss_netG": epoch_loss_netG_loss,
                "Loss_netD": epoch_loss_netD_loss
            }

            # Assuming you have variables `current_epoch`, `total_epochs`, `current_val_loss`, and `images` defined:
            self.log(epoch=epoch, epochs=epochs, losses=losses, images=images, val_loss=val_loss, metrics=metrics)

            #self.log(epoch, epochs, [epoch_ed_loss, epoch_refineG_loss, epoch_refineD_loss] , val_loss, metrics, images, pretrained = self.pre)

        self.writer.close()

    def test(self, b_loss):
        # Setting model for evaluation
        self.model.eval()

        base_loss, refineG_loss, refineD_loss = 0.0, 0.0, 0.0
        mse_loss, mae_loss, ssim, anom = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for data in self.loader["ts"]:
                real_batch = data['image'].to(self.device)
                ga = data['ga'].to(self.device) if self.ga else None

                # Run the whole framework forward, no need to do each component separate
                
                ref_recon, res_dic = self.model(real_batch, ga)

                # Obtain the anomaly metric from the model
                # anomap = abs(ref_recon-real_batch)*self.model.anomap.saliency_map(ref_recon,real_batch)

                # Calc the losses

                #   encoder-decoder loss
                ed_loss = self.base_loss[b_loss](res_dic["x_recon"],real_batch)

                #   refinement loss
                losses = {}
                for name, weight in self.loss_keys.items():
                    losses[name] = weight * self.losses[name](res_dic["y_ref"], real_batch)

                dis_loss, gen_loss = self.adv_loss(self.model.refineD, ref_recon, real_batch, ga)

                losses['advg'] = gen_loss * self.adv_weight

                base_loss += ed_loss
                refineG_loss += sum(losses.values()).cpu().item()
                refineD_loss += dis_loss.cpu().item()

                # Calc the metrics
                mse_loss += loss_lib.l2_loss(res_dic["y_ref"], real_batch).item()
                mae_loss += loss_lib.l1_loss(res_dic["y_ref"], real_batch).item()
                ssim     += 1 - loss_lib.ssim_loss(res_dic["y_ref"], real_batch).item()
                # anom     += torch.mean(anomap.flatten()).item()

            base_loss /= len(self.loader["ts"])
            refineG_loss /= len(self.loader["ts"])
            refineD_loss /= len(self.loader["ts"])

            mse_loss /= len(self.loader["ts"])
            mae_loss /= len(self.loader["ts"])
            ssim /= len(self.loader["ts"])
            # anom /= len(self.loader["ts"])    

            # Images dic for visualization
            images = {"input": real_batch[0][0], "recon": res_dic["x_recon"][0], "ref_recon": ref_recon[0] }    
        
        return {'losses': [ed_loss, refineG_loss, refineD_loss],'metrics': [mse_loss, mae_loss, ssim, anom], 'images': images}

    def log(self, epoch, epochs, losses, images, val_loss, metrics):
        # Format the new losses for logging
        formatted_losses = ', '.join([f'{key}: {(value.item() if isinstance(value, torch.Tensor) else value):.4f}' for key, value in losses.items()])
        header = 'Epoch, ED Loss, RefineG Loss, RefineD Loss, MSE Loss, MAE Loss, SSIM, Anom\n'
        log_file_path = f'{self.model_path}/training_log.csv'

        if not os.path.exists(log_file_path) or os.stat(log_file_path).st_size == 0:
            with open(log_file_path, 'w') as file:
                file.write(header)
                
        primary_val_loss = val_loss[0] 

        print(f'{losses=}')

        losses_str = ', '.join([f'{float(loss):.4f}' if loss.replace('.', '', 1).isdigit() else loss for loss in losses])
        val_losses_str = ', '.join([f'{val:.4f}' for val in val_loss])
        metrics_str = ', '.join([f'{metric:.4f}' for metric in metrics])


        components = ['encoder', 'decoder', 'refineG', 'refineD']
        for component in components:
            torch.save({
                'epoch': epoch + 1,
                component: getattr(self.model, component).state_dict(),
            }, f'{self.model_path}/{component}_latest.pth')

        # Save and plot model components every 50 epochs or in the first or last epoch
        if (epoch == 0) or ((epoch + 1) % 10 == 0) or ((epoch + 1) == epochs):
            for component in components:
                torch.save({'epoch': epoch + 1, component: getattr(self.model, component).state_dict()},
                        f'{self.model_path}/{component}_{epoch + 1}.pth')
            
            # Plot and save the progress image
            progress_im = self.plot(images)
            progress_im.savefig(f'{self.image_path}epoch_{epoch+1}.png')

        # Save the best model components if current validation loss is lower than the best known loss
        if isinstance(primary_val_loss, torch.Tensor):
            primary_val_loss_value = primary_val_loss.item()  # Convert tensor to a Python number
        else:
            primary_val_loss_value = primary_val_loss  # It's already a number, not a tensor
        if primary_val_loss_value < self.best_loss:
            self.best_loss = primary_val_loss
            for component in components:
                torch.save({
                    'epoch': epoch + 1,
                    component: getattr(self.model, component).state_dict(),
                }, f'{self.model_path}/{component}_best.pth')
            print(f'Saved best model components at epoch {epoch+1} with primary validation loss {primary_val_loss_value:.4f}')

        log_entry = f'{epoch+1}, {losses_str}, {val_losses_str}, {metrics_str}\n'
        with open(log_file_path, 'a') as file:
            file.write(log_entry)

        print(f'Epoch {epoch+1}, Losses: {losses_str}, Val Losses: {val_losses_str}, Metrics: {metrics_str}')
        print(f'{self.best_loss=}')


    def plot(self, images):
        fig, axs = plt.subplots(1, 3, figsize=(10, 6))
        names = ["input", "recon", "ref_recon"]
        for y in range(3):
            axs[y].imshow(images[names[y]].detach().cpu().numpy().squeeze(), cmap='gray')
            axs[y].set_title(names[y])
            axs[y].axis("off")
        plt.tight_layout()
        return fig