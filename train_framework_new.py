# Code adapted based on https://github.com/ci-ber/PHANES and https://github.com/researchmm/AOT-GAN-for-Inpainting
# Code written by  @GuillermoTafoya & @simonamador

import torch
from torch.nn import DataParallel
import torch.optim as optim

import matplotlib.pyplot as plt

from models.framework import Framework
from utils.config import loader, load_model
from utils import loss as loss_lib
from utils.debugging_printers import *

class Trainer:
    def __init__(self, source_path, model_path, tensor_path,
                 image_path, device, batch, z_dim, method, model, 
                 base, view, n, pretrained, pretrained_path, ga_n, raw, th = 99):
        
        # Determine if model inputs GA
        if base == 'ga_VAE':
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

        self.device = device
        self.model_type = model
        self.model_path = model_path  
        self.tensor_path = tensor_path 
        self.image_path = image_path  
        self.th = th

        # Generate model
        self.model = Framework(n, z_dim, method, device, model, self.ga, ga_n, th=self.th)

        # Load pre-trained parameters
        if pretrained == 'base':
            encoder, decoder = load_model(pretrained_path, base, method, n, n, z_dim, model=model, pre = pretrained, ga_n = ga_n)
            self.model.encoder = encoder
            self.model.decoder = decoder
        if pretrained == 'refine':
            refineG, refineD = load_model(pretrained_path, base, method, n, n, z_dim, model=model, pre = pretrained, ga_n = ga_n)
            self.model.refineG = refineG
            self.model.refineD = refineD
        prGreen('Model successfully instanciated...')
        self.pre = pretrained

        # Load losses
        self.base_loss = {'L2': loss_lib.l2_loss, 'L1': loss_lib.l1_loss, 'SSIM': loss_lib.ssim_loss, 
                     'MS_SSIM': loss_lib.ms_ssim_loss}
        self.loss_keys = {'L1': 1, 'Style': 250, 'Perceptual': 0.1}
        self.losses = {'L1':loss_lib.l1_loss,
                'Style':loss_lib.Style(),
                'Perceptual':loss_lib.Perceptual()}
        self.adv_loss = loss_lib.smgan()
        self.adv_weight = 0.01

        ### VAE ADVERSARIAL LOSS ### TODO
        
        prGreen('Losses successfully loaded...')

        # Establish data loaders
        train_dl, val_dl = loader(source_path, view, batch, n, raw = raw)
        self.loader = {"tr": train_dl, "ts": val_dl}
        prGreen('Data loaders successfully loaded...')
        
        # Optimizers
        
        # self.optimizer_base = optim.Adam([{'params': self.model.encoder.parameters()},
        #                       {'params': self.model.decoder.parameters()}], lr=1e-4, weight_decay=1e-5)
        self.optimizer_e = optim.Adam(self.model.encoder.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_d = optim.Adam(self.model.decoder.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_netG = optim.Adam(self.model.refineG.parameters(), lr=5.0e-5)
        self.optimizer_netD = optim.Adam(self.model.refineD.parameters(), lr=5.0e-5)

        # TODO
        # self.e_scheduler = MultiStepLR(self.optimizer_e, milestones=(100,), gamma=0.1)
        # self.d_scheduler = MultiStepLR(self.optimizer_d, milestones=(100,), gamma=0.1)
        # self.netG_scheduler = MultiStepLR(self.optimizer_netG, milestones=(100,), gamma=0.1)
        # self.netD_scheduler = MultiStepLR(self.optimizer_netD, milestones=(100,), gamma=0.1)

        # TODO
        # self.scale = 1 / (training_params['input_size'][1] ** 2)  # normalize by images size (channels * height * width)
        # self.gamma_r = 1e-8
        # self.beta_kl = training_params['beta_kl'] if 'beta_kl' in training_params.keys() else 1.0
        # self.beta_rec = training_params['beta_rec'] if 'beta_rec' in training_params.keys() else 0.5
        # self.beta_neg = training_params['beta_neg'] if 'beta_neg' in training_params.keys() else 128.0
        # self.z_dim = training_params['z_dim'] if 'z_dim' in training_params.keys() else 128
        # self.masking_threshold_train = training_params['masking_threshold_train'] if 'masking_threshold_train' in \
        #                                                                   training_params.keys() else None
        # self.masking_threshold_inference = training_params['masking_threshold_infer'] if 'masking_threshold_infer' in \
        #                                                                   training_params.keys() else None
        # rec_loss = '1*L1+250*Style+0.1*Perceptual'
        # self.adv_weight = training_params['adv_weight'] if 'adv_weight' in training_params.keys() else 0.01
        # gan_type = 'smgan'
        # losses = list(rec_loss.split('+'))
        # self.rec_loss = {}
        # for l in losses:
        #     weight, name = l.split('*')
        #     self.rec_loss[name] = float(weight)
        # # set up losses and metrics
        # self.rec_loss_func = {
        #     key: getattr(loss_module, key)() for key, val in self.rec_loss.items()}
        # self.adv_loss = getattr(loss_module, gan_type)()

        # self.embedding_loss = EmbeddingLoss()
        # super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

        prGreen('Optimizers successfully loaded...')

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
            refineG = DataParallel(self.model.refineG).to(self.device).train()
            refineD = DataParallel(self.model.refineD).to(self.device).train()

            print('-'*15)
            print(f'epoch {epoch+1}/{epochs}')

            epoch_ed_loss, epoch_refineG_loss, epoch_refineD_loss = 0.0, 0.0, 0.0

            # TODO
            # start_time = time()

            # diff_kls, batch_kls_real, batch_kls_fake, batch_kls_rec, batch_rec_errs, batch_exp_elbo_f,\
            # batch_exp_elbo_r, batch_emb, count_images = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            # batch_netGD_rec, batch_netG_loss, batch_netD_loss = 0.0, 0.0, 0.0

            # Runs through loader
            for data in current_loader:

                # ------ Grading Rules for Base Model   ------

                if self.pre is None or self.pre == 'refine':
                    for param in self.model.encoder.parameters():
                        param.requires_grad = True
                    for param in self.model.decoder.parameters():
                        param.requires_grad = True
                    for param in self.model.refineG.parameters():
                        param.requires_grad = False
                    for param in self.model.refineD.parameters():
                        param.requires_grad = False

                # TODO update ED independently as a AVAE
                # =========== Update E ================
                # if self.pre is None or self.pre == 'refine':
                    # for param in self.model.encoder.parameters():
                    #     param.requires_grad = True
                    # for param in self.model.decoder.parameters():
                    #     param.requires_grad = False
                    # for param in self.model.netG.parameters():
                    #     param.requires_grad = False
                    # for param in self.model.netD.parameters():
                    #     param.requires_grad = False

                img = data['image'].to(self.device)     # Extract image

                # TODO
                # noise_batch = torch.randn(size=(b, self.z_dim)).to(self.device)
                # real_batch = images.to(self.device)

                # Extract GA if required, encode z vector
                if self.ga:
                    ga = data['ga'].to(self.device)
                    z = encoder(img, ga)
                    #prCyan(f'{ga=}')
                else:
                    z = encoder(img)

                # TODO
                #  fake = self.model.sample(noise_batch)
                #  fake = sample(noise_batch)

                # Reconstruct image
                rec = decoder(z)


                # if self.pre is None or self.pre == 'refine': 

                # _, _, healthy_embeddings = self.model.encode(rec.detach())

                # loss_emb = self.embedding_loss(anomaly_embeddings['embeddings'], healthy_embeddings['embeddings'])

                # loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")
                # lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
                # rec_rec, z_dict = self.model.ae(rec.detach(), deterministic=False)
                # rec_mu, rec_logvar, z_rec = z_dict['z_mu'], z_dict['z_logvar'], z_dict['z']
                # rec_fake, z_dict_fake = self.model.ae(fake.detach(), deterministic=False)
                # fake_mu, fake_logvar, z_fake = z_dict_fake['z_mu'], z_dict_fake['z_logvar'], z_dict_fake['z']

                # kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
                # kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

                # loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type="mse", reduction='none')

                # while len(loss_rec_rec_e.shape) > 1:
                #     loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                # loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type="mse", reduction='none')
                # while len(loss_rec_fake_e.shape) > 1:
                #     loss_rec_fake_e = loss_rec_fake_e.sum(-1)

                # expelbo_rec = (-2 * self.scale * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)).exp().mean()
                # expelbo_fake = (-2 * self.scale * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)).exp().mean()

                # lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                # lossE_real = self.scale * (self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl)

                # if self.pre is None or self.pre == 'refine': 
                    # lossE = lossE_real + lossE_fake + 0.005 * loss_emb
                    # self.optimizer_e.zero_grad()
                    # lossE.backward()
                    # self.optimizer_e.step()

                # ------ Update Base Model   ------
                
                if self.pre is None or self.pre == 'refine': 
                    
                    ed_loss = self.base_loss[b_loss](rec,img)
                    self.optimizer_base.zero_grad()
                    ed_loss.backward()
                    self.optimizer_base.step()
                    epoch_ed_loss += ed_loss

                # TODO ========= Update D ==================
                # for param in self.model.encoder.parameters():
                #     param.requires_grad = False
                # for param in self.model.decoder.parameters():
                #     param.requires_grad = True
                # for param in self.model.netG.parameters():
                #     param.requires_grad = False
                # for param in self.model.netD.parameters():
                #     param.requires_grad = False

                # fake = self.model.sample(noise_batch)
                # rec = self.model.decoder(z.detach())
                # loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")

                # rec_mu, rec_logvar,_ = self.model.encode(rec)
                # z_rec = reparameterize(rec_mu, rec_logvar)

                # fake_mu, fake_logvar,_ = self.model.encode(fake)
                # z_fake = reparameterize(fake_mu, fake_logvar)

                # rec_rec = self.model.decode(z_rec.detach())
                # rec_fake = self.model.decode(z_fake.detach())

                # loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type="mse", reduction="mean")
                # loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type="mse", reduction="mean")

                # lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                # lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")


                # lossD = self.scale * (loss_rec * self.beta_rec + (
                #         lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl + self.gamma_r * 0.5 * self.beta_rec * (
                #                             loss_rec_rec + loss_fake_rec))

                # if self.pre is None or self.pre == 'refine': 
                    # self.optimizer_d.zero_grad()
                    # lossD.backward()
                    # self.optimizer_d.step()
                    # if torch.isnan(lossD) or torch.isnan(lossE):
                    #     print('is non for D')
                    #     raise SystemError
                    # if torch.isnan(lossE):
                    #     print('is non for E')
                    #     raise SystemError

                # diff_kls += -lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item() * images.shape[0]
                # batch_kls_real += lossE_real_kl.data.cpu().item() * images.shape[0]
                # batch_kls_fake += lossD_fake_kl.cpu().item() * images.shape[0]
                # batch_kls_rec += lossD_rec_kl.data.cpu().item() * images.shape[0]
                # batch_rec_errs += loss_rec.data.cpu().item() * images.shape[0]

                # batch_exp_elbo_f += expelbo_fake.data.cpu() * images.shape[0]
                # batch_exp_elbo_r += expelbo_rec.data.cpu() * images.shape[0]

                # batch_emb += loss_emb.cpu().item() * images.shape[0]

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

                    # Obtain anomaly metric, use it to generate the masks
                    saliency, anomalies = self.model.anomap.anomaly(rec.detach(), img)
                    anomalies = anomalies * saliency
                    masks = self.model.anomap.mask_generation(anomalies)

                    x_ref = (img * (1 - masks).float()) + masks

                    # Refined reconstruction through AOT-GAN
                    if self.ga:
                        y_ref = refineG(x_ref, masks, ga) 
                    else:
                        y_ref = refineG(x_ref, masks)
                    
                    y_ref = torch.clamp(y_ref, 0, 1)

                    zero_pad = torch.nn.ZeroPad2d(1)
                    y_ref = zero_pad(y_ref)

                    # Only include the parts from the refined reconstruction which the mask
                    # identified as anomalous
                    ref_recon = (1-masks)*img + masks*y_ref

                    # Losses for AOT-GAN
                    losses = {}
                    for name, weight in self.loss_keys.items():
                        losses[name] = weight * self.losses[name](y_ref, img)

                    if self.ga:
                        dis_loss, gen_loss = self.adv_loss(self.model.refineD, ref_recon, img, masks, ga)
                    else:
                        dis_loss, gen_loss = self.adv_loss(self.model.refineD, ref_recon, img, masks)

                    # No se incluye en el entrenamiento de SAPI.
                    losses['advg'] = gen_loss * self.adv_weight
                    
                    self.optimizer_netG.zero_grad()
                    self.optimizer_netD.zero_grad()
                    sum(losses.values()).backward()
                    dis_loss.backward()
                    self.optimizer_netG.step()
                    self.optimizer_netD.step()

                    epoch_refineG_loss += sum(losses.values()).cpu().item()
                    epoch_refineD_loss += dis_loss.cpu().item()
            
            # Epoch-loss
            epoch_ed_loss /= len(self.loader["tr"])
            epoch_refineG_loss /= len(self.loader["tr"])
            epoch_refineD_loss /= len(self.loader["tr"])

            # Testing
            test_dic = self.test(b_loss)
            val_loss = test_dic["losses"] 
            metrics = test_dic["metrics"] 
            images = test_dic["images"] 

            # Logging
            self.log(epoch, epochs, [epoch_ed_loss, epoch_refineG_loss, epoch_refineD_loss] , val_loss, metrics, images, pretrained = self.pre)

            # Printing current epoch losses acording to the component being trained.

            print(f'{epoch_ed_loss=:.6f}')
            print(f'{epoch_refineG_loss=:.6f}')
            print(f'{epoch_refineD_loss=:.6f}')

            print(f'ed_val_los={val_loss[0]:.6f}')
            print(f'refineG_val_loss={val_loss[1]:.6f}')
            print(f'refineD_val_loss={val_loss[2]:.6f}')



        self.writer.close()

    def test(self, b_loss):
        # Setting model for evaluation
        self.model.eval()

        base_loss, refineG_loss, refineD_loss = 0.0, 0.0, 0.0
        mse_loss, mae_loss, ssim, anom = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for data in self.loader["ts"]:
                img = data['image'].to(self.device)

                # Run the whole framework forward, no need to do each component separate
                if self.ga:
                    ga = data['ga'].to(self.device)
                    ref_recon, res_dic = self.model(img, ga)
                else:
                    ref_recon, res_dic = self.model(img)

                # Obtain the anomaly metric from the model
                anomap = abs(ref_recon-img)*self.model.anomap.saliency_map(ref_recon,img)

                # Calc the losses

                #   encoder-decoder loss
                ed_loss = self.base_loss[b_loss](res_dic["x_recon"],img)

                #   refinement loss
                losses = {}
                for name, weight in self.loss_keys.items():
                    losses[name] = weight * self.losses[name](res_dic["y_ref"], img)

                if self.ga:
                    dis_loss, gen_loss = self.adv_loss(self.model.refineD, ref_recon, img, res_dic["mask"], ga)
                else:
                    dis_loss, gen_loss = self.adv_loss(self.model.refineD, ref_recon, img, res_dic["mask"])

                losses['advg'] = gen_loss * self.adv_weight

                base_loss += ed_loss
                refineG_loss += sum(losses.values()).cpu().item()
                refineD_loss += dis_loss.cpu().item()

                # Calc the metrics
                mse_loss += loss_lib.l2_loss(res_dic["y_ref"], img).item()
                mae_loss += loss_lib.l1_loss(res_dic["y_ref"], img).item()
                ssim     += 1 - loss_lib.ssim_loss(res_dic["y_ref"], img).item()
                anom     += torch.mean(anomap.flatten()).item()

            base_loss /= len(self.loader["ts"])
            refineG_loss /= len(self.loader["ts"])
            refineD_loss /= len(self.loader["ts"])

            mse_loss /= len(self.loader["ts"])
            mae_loss /= len(self.loader["ts"])
            ssim /= len(self.loader["ts"])
            anom /= len(self.loader["ts"])    

            # Images dic for visualization
            images = {"input": img[0][0], "recon": res_dic["x_recon"][0], "saliency": res_dic["saliency"][0],
                      "mask": -res_dic["mask"][0], "ref_recon": ref_recon[0], "anomaly": anomap[0][0]}    
        
        return {'losses': [ed_loss, refineG_loss, refineD_loss],'metrics': [mse_loss, mae_loss, ssim, anom], 'images': images}

    def log(self, epoch, epochs, tr_loss, val_loss, metrics, images, pretrained):
        model_path = self.model_path
        
        # Every epoch log the training and validation losses for base, refinement_generator and refinement_discriminator,
        # as well as the metrics.
        self.writer.write(str(epoch+1) + ', ' +
                          str(tr_loss[0].item() if pretrained != 'base' else '0'
                              ) + ', ' +
                          str(tr_loss[1]) + ', ' +
                          str(tr_loss[2]) + ', ' +
                          str(val_loss[0].item() if pretrained != 'base' else '0'
                              ) + ', ' +
                          str(val_loss[1]) + ', ' +
                          str(val_loss[2]) + ', ' +
                          str(metrics[0]) + ', ' +
                          str(metrics[1]) + ', ' +
                          str(metrics[2]) + ', ' +
                          str(metrics[3]) + '\n')
        
        # Plot first iteration
        if epoch == 0:
            progress_im = self.plot(images)
            progress_im.savefig(self.image_path+'epoch_'+str(epoch+1)+'.png')

        # Save and plot model every 50 epochs
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'encoder': self.model.encoder.state_dict(),
            }, model_path + f'/encoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': self.model.decoder.state_dict(),
            }, model_path + f'/decoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'refineG': self.model.refineG.state_dict(),
            }, model_path + f'/refineG_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'refineD': self.model.refineD.state_dict(),
            }, model_path + f'/refineD_{epoch + 1}.pth')

            progress_im = self.plot(images)
            progress_im.savefig(self.image_path+'epoch_'+str(epoch+1)+'.png')

        # Save the best model
        if val_loss[0] < self.best_loss:
            self.best_loss = val_loss[0]
            torch.save({
                'epoch': epoch + 1,
                'encoder': self.model.encoder.state_dict(),
            }, model_path + f'/encoder_best.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': self.model.decoder.state_dict(),
            }, model_path + f'/decoder_best.pth')

            torch.save({
                'epoch': epoch + 1,
                'refineG': self.model.refineG.state_dict(),
            }, model_path + f'/refineG_best.pth')

            torch.save({
                'epoch': epoch + 1,
                'refineD': self.model.refineD.state_dict(),
            }, model_path + f'/refineD_best.pth')
            print(f'saved best model in epoch: {epoch+1}')

    def plot(self, images):
        fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        names = [["input", "recon", "ref_recon"], ["saliency", "anomaly", "mask"]]
        cmap_i = ["gray", "hot"]
        for x in range(2):
            for y in range(3):
                if x == 1 and y == 2:
                    cmap_i[1] = "binary"
                axs[x, y].imshow(images[names[x][y]].detach().cpu().numpy().squeeze(), cmap=cmap_i[x])
                axs[x, y].set_title(names[x][y])
                axs[x, y].axis("off")
        plt.tight_layout()
        return fig