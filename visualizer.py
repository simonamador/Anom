# Code written by @GuillermoTafoya

from utils.config import load_model, val_loader
from utils.loss import ssim_loss, l1_loss, l2_loss
from models.anomaly import Anomaly
from models.framework import Framework
from models.csgan.cycle_GAN import CycleGANModel

from models.framework_new import Framework as GANFramework
from utils.config import load_model_new as load_GAN_model

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stts
import seaborn as sns
import pandas as pd
import os
import numpy as np
from utils.debugging_printers import *
from bunch_py3 import *


class Visualizer:
    def __init__(self, parameters, task = 'base'):
                 
                 
                 #, path, model_path, base, model, view, method, z_dim, name, n, device, training_folder, ga_n, raw, th = 99, cGAN = False):
            
            # visualizer = Visualizer(args.path, model_path, args.VAE_model_type, args.type, args.view, args.ga_method, 
            # args.z_dim, args.name, args.slice_size, device, args.training_folder, args.ga_n, args.raw, args.th, args.cGAN)

            # Determine if model inputs GA
            self.ga =  parameters['VAE_model_type'] == 'ga_VAE'
            model_name = parameters['name'] + '_' + parameters['view']
            print(f'{self.ga=}')
            print(f'{parameters["z_dim"]=}')
            

            self.view = parameters['view']
            self.device = parameters['device']
            self.raw = parameters['raw']
            
            prGreen(f'{parameters["device"]=}')

            self.th = parameters['th'] if parameters['th'] else 99

            model_loader = {
                'base':load_model,
                'gan': load_GAN_model
            }

            frameworks = {
                'base': Framework,
                'gan': GANFramework
            }

            # Generate and load model
            print(parameters['model_path'])
            self.model = frameworks[task](parameters['slice_size'], parameters['z_dim'], 
                                   parameters['ga_method'], parameters['device'], 
                                   parameters['model_path'], self.ga, parameters['ga_n'], 
                                   th = self.th, BOE_form = parameters['BOE_type'])
            
            # self.model.encoder, self.model.decoder, self.model.refineG = load_model(model_path, base, method, 
            #                                                     n, n, z_dim, model=model, pre = 'full', ga_n=ga_n)

            self.model.encoder, self.model.decoder, self.model.refineG = model_loader[task](parameters['model_path'], parameters['VAE_model_type'], 
                                                                                parameters['ga_method'], parameters['slice_size'], 
                                                                                parameters['slice_size'], parameters['z_dim'], 
                                                                                model=parameters['type'], pre = 'full', 
                                                                                ga_n = parameters['ga_n'], BOE_form = parameters['BOE_type'])
            

            if task in ['cycle']:
            

                opt = Bunch({
                'lambda_identity': 0.5,     # First, try using identity loss `--lambda_identity 1.0` or `--lambda_identity 0.1`. 
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
                'n_epochs': 200,            # number of epochs with the initial learning rate
                'n_epochs_decay': 200,      # number of epochs to linearly decay learning rate to zero
                'verbose': True,
                'preprocess': 'crop', # scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
                'load_size': 160,
                'crop_size': 160
                    }
                )

                ### Cycle GAN ###
                self.cycle_GAN = CycleGANModel(opt)
                self.cycle_GAN.setup(opt) 
            
            # Visualization paths
            self.hist_path = parameters['path']+'Results' + model_name + '/history.txt'
            self.vis_path = parameters['path']+'Results/Visualization/'+model_name+'/'

            
            os.makedirs(self.vis_path, exist_ok=True)
            os.makedirs(self.vis_path+'Whole/', exist_ok=True)
            os.makedirs(self.vis_path+'Whole/TD/', exist_ok=True)
            os.makedirs(self.vis_path+'Whole/VM/', exist_ok=True)
            os.makedirs(self.vis_path+'Process/', exist_ok=True)
            os.makedirs(self.vis_path+'Process/TD/', exist_ok=True)
            os.makedirs(self.vis_path+'Process/VM/', exist_ok=True)

            self.vm_path = parameters['path']+ ('/VM_dataset/Raw/' if not parameters['raw'] else '/VM_symposium/Raw/')
            self.vm_images = os.listdir(self.vm_path)
            self.td_path = parameters['path']+parameters['training_folder']+'/test/'
            self.td_images= os.listdir(self.td_path)

    


    def visualize_age_effect(self, delta_ga=5):
        print('----- BEGINNING VISUALIZATION -----')
        
            
        self.model = self.model.to(self.device)

        loader = val_loader(self.td_path, self.td_images, self.view, raw = self.raw)

        prev = ''
        reconstructed = 0

        for id, slice in enumerate(loader):
            if self.td_images[int(id/30)][:-4] == prev:
                continue
            prev = self.td_images[int(id/30)][:-4]
            prCyan(f'Working with id={id}')
            img = slice['image'].to(self.device)
            if self.ga:
                for validation in range(3):
                    ga = slice['ga'].to(self.device)
                    ga_copy = ga.clone().detach().cpu().numpy()
                    ### GA Conditioning ###
                    # if ga > 25:
                    #     continue
                    ga_variation = np.arange(ga_copy - delta_ga, ga_copy + delta_ga + 1, 1)

                    # Define the layout for subplot_mosaic
                    n_images = len(ga_variation)
                    layout = [['recon' + str(i) for i in range(n_images)],
                            ['refine' + str(i) for i in range(n_images)]]

                    # Create the figure with the specified layout
                    fig, axd = plt.subplot_mosaic(layout, figsize=(64, 32), dpi=80)

                    # Set the main title
                    fig.suptitle(f'GA Effect On Reconstruction for {self.td_images[int(id/30)][:-4]} GA({float(ga_copy):.2f})', fontsize=64)

                    
                    # Calculate the top of the subplots for placing the row titles
                    top_of_subplots = max(ax.get_position().ymax for ax in axd.values())

                    # Set the row titles
                    fig.text(0.5, top_of_subplots - 0.1, 'Reconstruction', ha='center', va='bottom', fontsize=48, transform=fig.transFigure)
                    fig.text(0.5, top_of_subplots / 2 - 0.02, 'Refinement', ha='center', va='bottom', fontsize=48, transform=fig.transFigure)

                    # Adjust the layout
                    plt.subplots_adjust(top=top_of_subplots - 0.01)  


                    for idx, ga_val in enumerate(ga_variation):
                        ga_alt = torch.tensor([[ga_val]], dtype=torch.float).to(self.device)
                        recon_ref, rec_dic = self.model(img, ga_alt)

                        # recon = rec_dic["x_recon"][0].detach().cpu().numpy().squeeze() #np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), -1)
                        # refinement = recon_ref[0].detach().cpu().numpy().squeeze() #np.rot90(recon_ref[0].detach().cpu().numpy().squeeze(), -1)
                        recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), 1)
                        refinement = np.rot90(recon_ref[0].detach().cpu().numpy().squeeze(), 1)
                        

                        # Access the axes using the unique labels we created
                        ax_recon = axd['recon' + str(idx)]
                        ax_refine = axd['refine' + str(idx)]

                        ax_recon.imshow(recon, cmap='gray')
                        ax_refine.imshow(refinement, cmap='gray')

                        # Set GA values as titles for each subplot
                        title = f'{"+" if 0 < ga_val-float(ga_copy) else ""}{ga_val-float(ga_copy): .2f}'
                        ax_recon.set_title(title, fontsize=32)
                        ax_refine.set_title(title, fontsize=32)

                        # Turn off the axes
                        ax_recon.axis('off')
                        ax_refine.axis('off')

                    # Hide the tick labels for the recon images
                    for i in range(n_images):
                        axd['recon' + str(i)].set_xticklabels([])
                        axd['recon' + str(i)].set_yticklabels([])

                    plt.tight_layout(pad=4.0)

                    fig.savefig(self.vis_path+'TD/'+self.td_images[int(id/30)][:-4]+'_'+str(id-30*int(id/30))+'_'+str(validation)+'.png')
                    plt.close(fig)  # Close the figure to free memory

                    #plt.show()

                #if id == 0:  # Break after the first image for demonstration purposes
                #    break
                if reconstructed >= 15:  # Remove or modify this condition as needed
                    break
                reconstructed += 1


    def find_nonzero_bounding_box(slice_2d, percentile=98, file_name='slice_with_bbox.png'):
        import matplotlib.patches as patches
        def rotate(image):
            return np.rot90(image, 1)
    
        slice_2d = rotate(slice_2d)

        # Determine the threshold based on the specified percentile of the non-zero values
        threshold = np.percentile(slice_2d[slice_2d > 0], percentile)
        
        # Apply the threshold to create a binary array: True for values above the threshold, False otherwise
        binary_slice = slice_2d > threshold

        # Find the indices of the True elements
        rows, cols = np.nonzero(binary_slice)
        if len(rows) == 0 or len(cols) == 0:  # If the slice is effectively empty after thresholding
            return 0, 0  # No bounding box

        # Calculate the bounding box dimensions
        min_col, max_col = cols.min(), cols.max()
        min_row, max_row = rows.min(), rows.max()
        width = max_col - min_col + 1
        height = max_row - min_row + 1

        # Plotting
        fig, ax = plt.subplots()
        ax.imshow(slice_2d, cmap='gray')
        # Add a rectangle patch for the bounding box
        rect = patches.Rectangle((min_col, min_row), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        voxel_size = 0.859375
        plt.title(f'Bounding Box: {width*voxel_size}x{height*voxel_size} mm (width x height)')  # Display the size of the bounding box
        plt.savefig(file_name)  # Save the figure
        plt.close(fig)  # Close the figure to free up memory

        return width, height
    
    # find_nonzero_bounding_box_3(normalized_slice, percentile=80, file_name=f'{case}_central_{view}_normalized_slice_with_bbox.png')

    def save_reconstruction_images(self, delta_ga=10, TD = True):
        model = self.model.to(self.device)
        path = self.td_path if TD else self.vm_path
        images = self.td_images if TD else self.vm_images

        loader = val_loader(path, images, self.view, raw = self.raw, data='healthy' if TD else 'VM')

        prev = ''
        reconstructed = 0

        sv_path = self.vis_path + ''
        

        for id, slice in enumerate(loader):
            if images[int(id / 30)][:-4] == prev:
                continue
            prev = images[int(id / 30)][:-4]
            img = slice['image'].to(self.device)
            ga = slice['ga'].to(self.device)
            original_ga = ga.clone().detach().cpu().numpy().item()  # Get original GA as a float
            ga_variation = np.arange(20, 41, 1)  # Original range of gestational ages

            # Append the original GA to the range and ensure all values are unique
            ga_variation = np.unique(np.append(ga_variation, original_ga))

            for ga_val in ga_variation:
                ga_alt = torch.tensor([[ga_val]], dtype=torch.float).to(self.device)
                _, rec_dic = model(img, ga_alt)
                recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), 1)
                #recon = rec_dic["x_recon"][0].detach().cpu().numpy().squeeze()

                # Save the reconstruction image
                fig, ax = plt.subplots()
                ax.imshow(recon, cmap='gray')
                ax.set_title(f'GA: {ga_val:.2f}')
                ax.axis('off')

                folder_path = self.vis_path + 'Whole/' + ('TD/' if TD else 'VM/') + images[int(id / 30)][:-4]
                os.makedirs(folder_path, exist_ok=True)

                fig.savefig(folder_path+'/'+str(id-30*int(id/30))+'_'+str(ga_val)+'.png')

                # Optional: Print out a status message
                print(f'Saved Reconstruction image for GA value {ga_val:.2f}')

            if reconstructed >= 15:  # Remove or modify this condition as needed
                break
            reconstructed += 1

    def save_reconstruction_images_GAN(self, TD = True):
        model = self.model.to(self.device)
        path = self.td_path if TD else self.vm_path
        images = self.td_images if TD else self.vm_images

        loader = val_loader(path, images, self.view, raw = self.raw, data='healthy' if TD else 'VM')

        prev = ''
        reconstructed = 0

        for id, slice in enumerate(loader):
            if images[int(id / 30)][:-4] == prev:
                continue
            prev = images[int(id / 30)][:-4]
            img = slice['image'].to(self.device)
            ga = slice['ga'].to(self.device)
            original_ga = ga.clone().detach().cpu().numpy().item()  # Get original GA as a float
            ga_variation = np.arange(20, 41, 1)  # Original range of gestational ages

            # Append the original GA to the range and ensure all values are unique
            ga_variation = np.unique(np.append(ga_variation, original_ga))

            for ga_val in ga_variation:
                ga_alt = torch.tensor([[ga_val]], dtype=torch.float).to(self.device)
                final, rec_dic = model(img, ga_alt)
                recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), 1)
                final_image = np.rot90(final[0].detach().cpu().numpy().squeeze(), 1)
                original_img = np.rot90(img[0].detach().cpu().numpy().squeeze(), 1)

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.imshow(original_img, cmap='gray')
                ax1.set_title(f'Original Input GA:{original_ga:.2f}')
                ax1.axis('off')

                ax2.imshow(recon, cmap='gray')
                ax2.set_title(f'x_recon GA: {ga_val:.2f}')
                ax2.axis('off')

                ax3.imshow(final_image, cmap='gray')
                ax3.set_title(f'Final GA: {ga_val:.2f}')
                ax3.axis('off')

                # os.makedirs(self.vis_path+'Whole/TD/'+self.td_images[int(id/30)][:-4], exist_ok=True)
                # fig.savefig(self.vis_path+'Whole/TD/'+self.td_images[int(id/30)][:-4]+'/'+str(id-30*int(id/30))+'_'+str(ga_val)+'.png')


                folder_path = self.vis_path + 'Whole/' + ('TD/' if TD else 'VM/') + images[int(id / 30)][:-4]
                os.makedirs(folder_path, exist_ok=True)

                fig.savefig(folder_path+'/'+str(id-30*int(id/30))+'_'+str(ga_val)+'.png')

                plt.close(fig)

                print(f'Saved Reconstruction image for GA value {ga_val:.2f}')

            if reconstructed >= 15:
                break
            reconstructed += 1

    def save_whole_range_plus_refined(self, TD = True):
        model = self.model.to(self.device)
        path = self.td_path if TD else self.vm_path
        images = self.td_images if TD else self.vm_images

        loader = val_loader(path, images, self.view, raw = self.raw, data='healthy' if TD else 'VM')

        prev = ''
        reconstructed = 0

        for id, slice in enumerate(loader):
            if images[int(id / 30)][:-4] == prev:
                continue
            prev = images[int(id / 30)][:-4]
            img = slice['image'].to(self.device)
            original_img = np.rot90(img.detach().cpu().numpy().squeeze(), 1)  # Assuming img is a single-channel image
            ga = slice['ga'].to(self.device)
            original_ga = ga.clone().detach().cpu().numpy().item()  # Get original GA as a float
            ga_variation = np.arange(20, 41, 1)  # Original range of gestational ages

            # Append the original GA to the range and ensure all values are unique
            ga_variation = np.unique(np.append(ga_variation, original_ga))

            for ga_val in ga_variation:
                ga_alt = torch.tensor([[ga_val]], dtype=torch.float).to(self.device)
                recon_ref, rec_dic = model(img, ga_alt)

                # Extracting the images from the model output
                refined_recon = np.rot90(recon_ref.detach().cpu().numpy().squeeze(), 1)
                vae_recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), 1)
                saliency_map_raw = np.rot90(rec_dic["saliency"][0].detach().cpu().numpy().squeeze(), 1)
                refinement_mask = np.rot90(-rec_dic["mask"][0].detach().cpu().numpy().squeeze(), 1)
                anomap_vae = np.rot90(abs(rec_dic["x_recon"]-img).detach().cpu().numpy().squeeze()* self.model.anomap.saliency_map(rec_dic["x_recon"], img).detach().cpu().numpy().squeeze(), 1) 
                saliency_map_refined = self.model.anomap.saliency_map(recon_ref, img).detach().cpu().numpy().squeeze()
            
                anomap_refined = abs(recon_ref-img).detach().cpu().numpy().squeeze() * saliency_map_refined

                # Create a figure with subplots for both rows
                fig, axs = plt.subplots(3, 3, figsize=(15, 15))

                # First row: Original, VAE reconstructed, Refined reconstructed
                axs[0, 0].imshow(original_img, cmap='gray')
                axs[0, 0].set_title(f'Original GA: {original_ga:.2f}')

                axs[0, 1].imshow(vae_recon, cmap='gray')
                axs[0, 1].set_title(f'VAE GA: {ga_val:.2f}')

                axs[0, 2].imshow(refined_recon, cmap='gray')
                axs[0, 2].set_title(f'Refined GA: {ga_val:.2f}')

                # Second row
                axs[1, 0].imshow(original_img, cmap='gray')  # Base image for saliency map
                axs[1, 0].imshow(saliency_map_raw, cmap='hot', alpha=0.8)  # Saliency map overlaid
                axs[1, 0].set_title(f'Saliency map VAE')

                axs[1, 1].imshow(original_img, cmap='gray')  # Base image for anomaly map
                axs[1, 1].imshow(np.rot90(saliency_map_refined, 1), cmap='hot', alpha=0.8)  # Anomaly map overlaid
                axs[1, 1].set_title(f'Saliency map refined')

                axs[1, 2].imshow(original_img, cmap='gray')  # Base image for refinement mask
                axs[1, 2].imshow(refinement_mask, cmap='Blues', alpha=0.5)  # Refinement mask overlaid
                axs[1, 2].set_title(f'Refining mask')

                # Third row
                axs[2, 0].imshow(original_img, cmap='gray')  # Base image for saliency map
                axs[2, 0].imshow(anomap_vae, cmap='hot', alpha=0.9)  # Saliency map overlaid
                axs[2, 0].set_title(f'Anomaly map VAE')
                
                axs[2, 1].imshow(original_img, cmap='gray')  # Base image for refinement mask
                axs[2, 1].imshow(np.rot90(anomap_refined, 1), cmap='hot', alpha=0.9)  # Refinement mask overlaid
                axs[2, 1].set_title(f'Anomaly map refined')

                #axs[2, 2].imshow(original_img, cmap='gray')  # Base image for refinement mask
                #axs[2, 2].imshow(np.rot90(anomap_refined, 1), cmap='hot', alpha=0.9)  # Refinement mask overlaid
                #axs[2, 2].set_title(f'Anomaly map refined')

                MSE = l2_loss(img, recon_ref).item()
                MAE = l1_loss(img, recon_ref).item()
                SSIM = 1-ssim_loss(img, recon_ref)
                anomaly_metric = torch.mean(torch.tensor(anomap_refined)).item()
                
                x = 0.7
                y = 0.11
                fig.text(x, y, f'{MSE=:.4f}', fontsize=15)
                fig.text(x, y+0.02, f'{MAE=:.4f}', fontsize=15)
                fig.text(x, y+0.04, f'{SSIM=:.4f}', fontsize=15)
                fig.text(x, y+0.06, f'{anomaly_metric=:.4f}', fontsize=15)

                for i in range(3):
                    for j in range(3):
                        axs[i, j].axis('off')  # Remove axes for all plots

                # Adjust subplot parameters and save the figure
                fig.subplots_adjust(hspace=0.1, wspace=0.1)
                folder_path = self.vis_path + 'Process/' + ('TD/' if TD else 'VM/') + images[int(id / 30)][:-4]
                os.makedirs(folder_path, exist_ok=True)
                comparison_filename = f'{str(id - 30 * int(id / 30))}_{str(ga_val)}.png'
                fig.savefig(os.path.join(folder_path, comparison_filename))
            
                plt.close(fig)  # Close the figure to free memory

                # Optional: Print out a status message
                print(f'Saved multi-image comparison for GA value {ga_val:.2f}')

            print(f'Processed {folder_path}')
            if reconstructed >= 15:  
                break
            reconstructed += 1