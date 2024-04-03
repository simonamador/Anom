# Code written by @simonamador

from utils.config import load_model, val_loader
from utils.loss import ssim_loss, l1_loss, l2_loss
from models.anomaly import Anomaly
from models.framework import Framework

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stts
import seaborn as sns
import pandas as pd
import os
import numpy as np
import scipy.stats as stts

class Validator:
    def __init__(self, parameters):
                 
                 #path, model_path, base, model, view, method, z_dim, name, n, device, training_folder, ga_n, raw, th = 99):

        # validator = Validator(args.path, model_path, args.model, args.type, args.view, args.ga_method, 
        #             args.z_dim, args.name, args.slice_size, device, args.training_folder, args.ga_n, args.raw, args.th, args.cGAN)

        # Determine if model inputs GA
        if parameters['VAE_model_type'] == 'ga_VAE':
            self.ga = True
            model_name = parameters['name'] + '_' + parameters['view']
        else:
            self.ga = False
            model_name = parameters['name'] + '_' + parameters['view']

        self.view = parameters['view']
        self.device = parameters['device']
        self.raw = parameters['raw']
        self.th = parameters['th'] if parameters['th'] else 99

        # Generate and load model
        print(parameters['model_path'])
        self.model = Framework(parameters['slice_size'], parameters['z_dim'], 
                               parameters['ga_method'], parameters['device'], 
                               parameters['model_path'], self.ga, 
                               parameters['ga_n'], th=self.th, BOE_form = parameters['BOE_type'])
        
        self.model.encoder, self.model.decoder, self.model.refineG = load_model(parameters['model_path'], parameters['VAE_model_type'], 
                                                                                parameters['ga_method'], parameters['slice_size'], 
                                                                                parameters['slice_size'], parameters['z_dim'], 
                                                                                model=parameters['model_path'], pre = 'full', 
                                                                                ga_n = parameters['ga_n'], BOE_form = parameters['BOE_type'])
        
        #self.model.encoder, self.model.decoder, self.model.refineG = load_model(model_path, base, method, 
        #                                                    parameters['slice_size'], parameters['slice_size'], z_dim, model=model, pre = 'full', ga_n=ga_n)
       
        # Validation paths
        self.hist_path = parameters['path']+'Results' + model_name + '/history.txt'
        self.val_path = parameters['path']+'Results/Validations/'+model_name+'/'

        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)
            os.mkdir(self.val_path+'TD/')
            os.mkdir(self.val_path+'VM/')

        self.vm_path = parameters['path']+ ('/VM_dataset/Raw/' if not parameters['raw'] else '/VM_symposium/Raw/') #path+'/VM_dataset/Raw/'
        self.vm_images = os.listdir(self.vm_path)
        self.td_path = parameters['path']+parameters['training_folder']+'test/'
        self.td_images= os.listdir(self.td_path)

    def validation(self,):

        # Create logger
        writer = open(self.val_path+'validation.txt', 'w')
        writer.write('Class, Case_ID, Slide_ID, MAE, MSE, SSIM, Anomaly'+'\n')

        # Load model to device
        self.model = self.model.to(self.device)

        print('----- BEGINNING VALIDATION -----')
        print('.')
        print('.')

        '''
        --------------------- VALIDATION FOR VENTRICULOMEGALY SUBJECTS ---------------------
        '''

        print('-----VM Dataset-----')

        # Validation loader
        loader = val_loader(self.vm_path, self.vm_images, self.view, data='vm', raw = self.raw)

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                #if ga != 29.29:
                #    continue
                
                recon_ref, rec_dic = self.model(img, ga)
            else:
                recon_ref, rec_dic = self.model(img)

            # Extract metrics
            MSE = l2_loss(img, recon_ref).item()
            MAE = l1_loss(img, recon_ref).item()
            SSIM = 1-ssim_loss(img, recon_ref)
            anomap0 = abs(rec_dic["x_recon"]-img)*self.model.anomap.saliency_map(rec_dic["x_recon"],img)
            anomap = abs(recon_ref-img)*self.model.anomap.saliency_map(recon_ref,img)

            # Log: Class | Image # | Mean Average Error | Mean Square Error | Structural-Similarity | Anomaly Metric
            writer.write('vm, '+
                                self.vm_images[int(id/30)][:-4]+', '+
                                str(id-30*int(id/30)+1)+', '+
                                str(MAE)+', '+
                                str(MSE*158)+', '+ #### MULTIPLIED BY SIZE (PIXEL N)
                                str(SSIM.item())+', '
                                +str(torch.mean(anomap.flatten()).item())+'\n')
            
            # Visualize images
            images = {"input": img[0][0], "recon": rec_dic["x_recon"][0], "coarse_anomaly": anomap0,"saliency": rec_dic["saliency"][0],
                    "mask": -rec_dic["mask"][0], "ref_recon": recon_ref[0], "anomaly": anomap[0]} 
            fig = self.plot(images, self.vm_images[int(id/30)][:-4], str(id-30*int(id/30)+1), 
                                     [MAE, MSE, SSIM.item(), torch.mean(anomap.flatten()).item()])
            fig.savefig(self.val_path+'VM/'+self.vm_images[int(id/30)][:-4]+'_'+str(id-30*int(id/30))+'.png')
            plt.close()

            # Print current subject
            if id%30 == 0:
                print('Subject {0} of {1}'.format(int(id/30)+1, len(self.vm_images)))
                
        
        #exit(0)
        '''
        --------------------- VALIDATION FOR TD SUBJECTS ---------------------
        '''

        print('-----TD Dataset-----')

        loader = val_loader(self.td_path, self.td_images, self.view, raw = self.raw)

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                #if ga != 30.57:
                #    continue
                recon_ref, rec_dic = self.model(img, ga)
            else:
                recon_ref, rec_dic = self.model(img)

            # Extract metrics
            MSE = l2_loss(img, recon_ref).item()
            MAE = l1_loss(img, recon_ref).item()
            SSIM = 1-ssim_loss(img, recon_ref)
            anomap0 = abs(rec_dic["x_recon"]-img)*self.model.anomap.saliency_map(rec_dic["x_recon"],img)
            anomap = abs(recon_ref-img)*self.model.anomap.saliency_map(recon_ref,img)

            # Log: Class | Image # | Mean Average Error | Mean Square Error | Structural-Similarity | Anomaly Metric
            writer.write('td, '+
                                self.td_images[int(id/30)][:-4]+', '+
                                str(id-30*int(id/30))+', '+
                                str(MAE)+', '+
                                str(MSE)+', '+
                                str(SSIM.item())+', '
                                +str(torch.mean(anomap.flatten()).item())+'\n')
            
            # Visualize images
            images = {"input": img[0][0], "recon": rec_dic["x_recon"][0], "coarse_anomaly": anomap0,"saliency": rec_dic["saliency"][0],
                    "mask": -rec_dic["mask"][0], "ref_recon": recon_ref[0], "anomaly": anomap[0]} 
            fig = self.plot(images,
                             self.td_images[int(id/30)][:-4],
                             str(id-30*int(id/30)+1), 
                            [MAE, MSE, SSIM.item(), torch.mean(anomap.flatten()).item()])
            fig.savefig(self.val_path+'TD/'+self.td_images[int(id/30)][:-4]+'_'+str(id-30*int(id/30))+'.png')
            plt.close()
            
            # Print current subject
            if id%30 == 0:
                print('Subject {0} of {1}'.format(int(id/30)+1, len(self.td_images)))

        print('.') 
        writer.close()
        print('.')
        print('Finished validation.')

    def GA_testing(self):
        pass
                
   

    def plot(self, images, subject, sliceid, metrics):
        def rotate(image):
            return np.rot90(image -1)
        
        fig, axs = plt.subplots(2,3, figsize=(20, 14))
        names = [["input", "recon", "ref_recon"], ["input", "input", "input"]]
        labels = [["Input MRI", "VAE Recon", "AOT-GAN Inpainting"], ["Saliency Map", "Mask", "Anomaly Map"]]
        overlays = ["", "", "", "saliency", "mask", "anomaly"]
        cmap_i = ["gray", "gray", "gray", "hot", "Blues", "hot"]
        alpha_values = [1, 1, 1, 0.8, 0.5, 0.9]

        for x in range(2):
            for y in range(3):
                base_img = images[names[x][y]].detach().cpu().numpy().squeeze()
                axs[x, y].imshow(rotate(base_img), cmap='gray')
                axs[x, y].set_title(labels[x][y], fontsize=34)
                axs[x, y].axis("off")
                
                if overlays[x*3 + y]:
                    overlay_img = rotate(images[overlays[x*3 + y]].detach().cpu().numpy().squeeze())
                    # Create a masked array where the mask is applied for zero values
                    masked_overlay = np.ma.masked_where(overlay_img == 0, overlay_img)
                    cmap = plt.cm.get_cmap(cmap_i[x*3 + y])
                    cmap.set_bad(alpha=0)  # Set the 'bad' values (masked) to be transparent
                    axs[x, y].imshow(masked_overlay, cmap=cmap, alpha=alpha_values[x*3 + y])

        #label = f'{subject}, Slice {sliceid}, MAE: {metrics[0]:.4f}, MSE: {metrics[1]:.4f}, SSIM: {metrics[2]:.4f}, Anomaly: {metrics[3]:.4f}'
        #fig.text(0.05, 1, label, fontsize=12)
        fig.subplots_adjust(hspace=0.3) 
        plt.tight_layout()
        #plt.subplots_adjust()
        return fig
    
    def tc(self,):
        #Pending (tc = training curves)
        df = pd.read_csv(self.hist_path)

    def severity(self,):
        #Pending dev
        a = 0

    def age_differential(self,delta_ga=10):
        # Create logger
        
        writer = open(self.val_path+'ga_validation.txt', 'w')
        writer.write('Case_ID, Slide_ID, GA_dif, MAE'+'\n')

        # Load model to device
        self.model = self.model.to(self.device)

        print('----- BEGINNING VALIDATION -----')
        print('.')
        print('.')

        '''
        --------------------- VALIDATION FOR TD SUBJECTS ---------------------
        '''

        print('-----GA Differential-----')

        loader = val_loader(self.td_path, self.td_images, self.view, raw = self.raw)

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                ga_copy = ga.clone().detach().cpu()
                
                #if ga_copy < 30:
                #    continue
                
                ga_variation = np.arange(ga_copy-delta_ga, ga_copy+delta_ga+0.5, 0.5)
                
                
                for ga_alt in ga_variation:
                    ga_alt = torch.tensor(np.expand_dims([ga_alt], axis = 0), dtype = torch.float).to(self.device)
                    # z = self.model.encoder(img, ga_alt)
                    z, _, _, _ = self.model.encoder(img, ga_alt)
                    recon = self.model.decoder(z)
                    # Extract metrics
                    MAE = l1_loss(img, recon).item()

                    # Log: GA dif | Image # | Mean Square Error 
                    writer.write(self.td_images[int(id/30)][:-4]+', '+
                                str(id-30*int(id/30))+', '+
                                f'{ga_alt.item()-ga.item():.1f}' +', '+
                                str(MAE)+'\n')
            else:
                raise NameError('Cannot validate age difference on a non-ga model.')
            
            # Print current subject
            if id%30 == 0:
                print('Subject {0} of {1}'.format(int(id/30)+1, len(self.td_images)))

        print('.') 
        writer.close()
        print('.')
        print('Finished validation.')

        df = pd.read_csv(self.val_path+'ga_validation.txt', sep=', ')

        mean = df.groupby('GA_dif').mean(numeric_only=True)['MAE']
        print(f'{mean=}')
        plt.plot([x/10.0 for x in range(-(delta_ga*10),delta_ga*10+5,5)],mean, color='orange')
        plt.xlabel('GA Difference')
        plt.ylabel('MAE')

        plt.savefig(self.val_path+f'ga_error_{delta_ga}.png')
        plt.close()

        
        plt.scatter(df['GA_dif'], df['MAE'])

        plt.plot([x/10.0 for x in range(-(delta_ga*10),delta_ga*10+5,5)],mean, color='orange')
        plt.xlabel('GA Difference')
        plt.ylabel('MAE')

        plt.savefig(self.val_path+f'ga_error_Dist_{delta_ga}.png')
        plt.close()

        print('Saved graphic.')

    def mannwhitneyu(self):
        print('-----BEGINNING STATS-----')
        print('.')
        print('.')

        df = pd.read_csv(self.val_path+'validation.txt', sep=', ')
        n_df = pd.melt(df, id_vars='Class', value_vars=['MSE', 'MAE', 'Anomaly'],
                    var_name='Metric', value_name='Value')

        td_df = df.where(df['Class']=='td').dropna()
        td_df = df.where(td_df['Slide_ID']==15).dropna()
        vm_df = df.where(df['Class']=='vm').dropna()
        vm_df = df.where(vm_df['Slide_ID']==15).dropna()

        numeric_td_df = td_df.select_dtypes(include=[np.number])
        numeric_vm_df = vm_df.select_dtypes(include=[np.number])

        # Compute statistics
        td_means = numeric_td_df.mean()
        td_stds = numeric_td_df.std()
        vm_means = numeric_vm_df.mean()
        vm_stds = numeric_vm_df.std()

        # Mann-Whitney U Test for your metrics
        stat_mse, p_mse = stts.mannwhitneyu(td_df['MSE'], vm_df['MSE'])
        stat_mae, p_mae = stts.mannwhitneyu(td_df['MAE'], vm_df['MAE'])
        stat_ssim, p_ssim = stts.mannwhitneyu(td_df['SSIM'], vm_df['SSIM'])
        stat_anom, p_anom = stts.mannwhitneyu(td_df['Anomaly'], vm_df['Anomaly'])

        stats = [[str(stat_mse), str(stat_mae), str(stat_ssim), str(stat_anom)],
                [str(p_mse), str(p_mae), str(p_ssim), str(p_anom)],
                [str(td_means['MSE']), str(td_means['MAE']), str(td_means['SSIM']), str(td_means['Anomaly'])],
                [str(td_stds['MSE']), str(td_stds['MAE']), str(td_stds['SSIM']), str(td_stds['Anomaly'])],
                [str(vm_means['MSE']), str(vm_means['MAE']), str(vm_means['SSIM']), str(vm_means['Anomaly'])],
                [str(vm_stds['MSE']), str(vm_stds['MAE']), str(vm_stds['SSIM']), str(vm_stds['Anomaly'])]]

        # Plotting code
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].axis('tight')
        axs[0].axis('off')
        axs[0].table(cellText=stats, rowLabels=['U1', 'P-Value', 'TD-Mean', 'TD-SD', 'VM-Mean', 'VM-SD'], 
                    colLabels=['MSE', 'MAE', 'SSIM', 'Anomaly'], loc='center')

        sns.set_style("whitegrid", {'axes.grid': False})
        axs[1] = sns.violinplot(data=n_df, x="Metric", y="Value", hue="Class", split=True, inner="quart")
        if self.view == 'L':
            view = 'Sagittal'
        elif  self.view == 'A':
            view = 'Coronal'
        else:
            view = 'Axial'
        plt.title(view + ' Mann-Whitney U Test')
        plt.savefig(self.val_path + view + '_mann-whitman.png')

        print('Finished stats.')

    # AUROC Computing
    def AUROC(self):
        from sklearn.metrics import roc_auc_score, roc_curve, auc #accuracy_score, confusion_matrix, roc_auc_score, roc_curve
        from scipy import interp
        from numpy import linspace
        print('-----BEGINNING AUROC-----')
        print()
        print()
        

        # Assuming 'self.val_path' contains the path to your 'validation.txt' file
        # And 'self.view' contains the view type ('L' for Sagittal, 'A' for Coronal, etc.)
        df = pd.read_csv(self.val_path + 'validation.txt', sep=', ')

        # Filter for Slide_ID 15
        td_df = df[(df['Class'] == 'td') & (df['Slide_ID'] == 15)]
        vm_df = df[(df['Class'] == 'vm') & (df['Slide_ID'] == 15)]

        # Define view based on self.view
        if self.view == 'L':
            view = 'Sagittal'
        elif self.view == 'A':
            view = 'Coronal'
        else:
            view = 'Axial'

        # Assuming 'Anomaly' scores are what you want to use for AUROC calculation
        # Prepare labels and scores
        y_true = np.concatenate((np.zeros(len(td_df)), np.ones(len(vm_df))))
        scores = np.concatenate((td_df['Anomaly'].values, vm_df['Anomaly'].values))

        # Compute AUROC
        auroc = roc_auc_score(y_true, scores)
        print(f"{view} AUROC: {auroc}")

        # Generate and plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        # This will create a high-resolution set of false positive rates
        base_fpr = linspace(0, 1, 101)

        # Interpolate the true positive rates at these points
        smooth_tpr = interp(base_fpr, fpr, tpr)
        smooth_tpr[0] = 0.0  # Ensuring the curve starts at 0

        # Plot the smoothed ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(base_fpr, smooth_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title(F'Receiver Operating Characteristic for {view} View')
        plt.legend(loc="lower right")
        
        plt.savefig(self.val_path+view+'_AUROC.png')
        #plt.show()



    def calculate_multiview_auroc_and_mannwhitneyu(self):
        from sklearn.metrics import roc_auc_score, roc_curve, auc #accuracy_score, confusion_matrix, roc_auc_score, roc_curve
        from scipy import interp
        from numpy import linspace
        import altair as alt

        # views = ['L', 'A', 'S']  # Sagittal, Coronal, Axial
        # view_labels = ['Sagittal', 'Coronal', 'Axial']
        views = ['L', 'A', 'S']  # Sagittal, Coronal, Axial
        view_labels = ['Sagittal', 'Coronal', 'Axial']
        combined_base_fpr = linspace(0, 1, 101)
        combined_tpr = []
        auroc_scores = []

        all_stats = []
        view_data = []

        for index, view in enumerate(views):
            # Modify val_path to replace the last character (view letter) and remove final slash if needed
            base_val_path = self.val_path[:-2]  # Remove view letter and slash
            current_val_path = f"{base_val_path}{view}/"  # Append current view letter with a slash

            # Load data for current view
            df = pd.read_csv(current_val_path + 'validation.txt', sep=', ')  # Adjust based on your actual data file
            df['View'] = view_labels[index]  # Add a 'View' column to label the current view
            view_data.append(df)

            # Filter for each class
            # td_df = df[df['Class'] == 'td']
            # vm_df = df[df['Class'] == 'vm']

            # Filter for Slide_ID 15
            td_df = df[(df['Class'] == 'td') & (df['Slide_ID'] == 15)]
            vm_df = df[(df['Class'] == 'vm') & (df['Slide_ID'] == 15)]

            # Mann-Whitney U Tests
            stat_mse, p_mse = stts.mannwhitneyu(td_df['MSE'], vm_df['MSE'])
            stat_mae, p_mae = stts.mannwhitneyu(td_df['MAE'], vm_df['MAE'])
            stat_anom, p_anom = stts.mannwhitneyu(td_df['Anomaly'], vm_df['Anomaly'])

            

            n_td = len(td_df)
            n_vm = len(vm_df)
            max_U_mse = n_td * n_vm
            max_U_mae = n_td * n_vm
            max_U_anom = n_td * n_vm

            print("Maximum possible U value for MSE:", max_U_mse)
            print("Maximum possible U value for MAE:", max_U_mae)
            print("Maximum possible U value for Anomaly:", max_U_anom)

            # Compute means and standard deviations for TD
            td_means = td_df[['MSE', 'MAE', 'Anomaly']].mean().tolist()
            td_stds = td_df[['MSE', 'MAE', 'Anomaly']].std().tolist()

            # Compute means and standard deviations for VM
            vm_means = vm_df[['MSE', 'MAE', 'Anomaly']].mean().tolist()
            vm_stds = vm_df[['MSE', 'MAE', 'Anomaly']].std().tolist()

            # Aggregate stats for Mann-Whitney U for each view
            # stats = {
            #     'View': view_labels[index],
            #     'U_Stats': [stat_mse, stat_mae, stat_anom],
            #     'P_Values': [p_mse, p_mae, p_anom]
            # }
            stats = {
                'View': view_labels[index],
                'TD_Means': td_means,
                'TD_Stds': td_stds,
                'VM_Means': vm_means,
                'VM_Stds': vm_stds,
                'U_Stats': [stat_mse, stat_mae, stat_anom],
                'P_Values': [p_mse, p_mae, p_anom]
            }
            all_stats.append(stats)

            # AUROC calculations
            y_true = np.concatenate((np.zeros(len(td_df)), np.ones(len(vm_df))))
            scores = np.concatenate((td_df['Anomaly'].values, vm_df['Anomaly'].values))
            auroc = roc_auc_score(y_true, scores)
            print(f"{view} AUROC: {auroc}")
            fpr, tpr, _ = roc_curve(y_true, scores)
            smooth_tpr = interp(combined_base_fpr, fpr, tpr)
            smooth_tpr[0] = 0.0  # Ensure it starts at 0
            combined_tpr.append(smooth_tpr)
            auroc_scores.append(auroc)

        plt.figure(figsize=(10, 10))
        for i, view_label in enumerate(view_labels):
            plt.plot(combined_base_fpr, combined_tpr[i], label=f'{view_label} View AUROC = {auroc_scores[i]:.2f}', linewidth=2)  # Increased line width for better visibility
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('1 - Specificity', fontsize=14)
        plt.ylabel('Sensitivity', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc='lower right', fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax = plt.gca()  # Get the current Axes instance
        ax.set_facecolor('white')  
        plt.tight_layout()
        # plt.savefig(base_val_path + 'Combined_AUROC_test.png', bbox_inches='tight')  # Save the plot with less whitespace
        plt.savefig(base_val_path + 'Combined_AUROC.png', bbox_inches='tight', transparent=True, facecolor=ax.get_facecolor())
        plt.close()


        def format_p_value(p): return ("<0.01", True) if p < 0.01 else (f"{p:.4f}", False)



        # TD Table Data Initialization
        td_table_data = [['View', 'MSE Mean', 'MSE SD', 'MAE Mean', 'MAE SD', 'Anomaly Mean', 'Anomaly SD']]

        # VM Table Data Initialization
        vm_table_data = [['View', 'MSE Mean', 'MSE SD', 'MAE Mean', 'MAE SD', 'Anomaly Mean', 'Anomaly SD']]

        # Populate the tables
        for stat in all_stats:
            # TD data
            td_row = [stat['View']] + [f"{x:.4f}" for x in stat['TD_Means']] + [f"{x:.4f}" for x in stat['TD_Stds']]
            td_table_data.append(td_row)

            # VM data
            vm_row = [stat['View']] + [f"{x:.4f}" for x in stat['VM_Means']] + [f"{x:.4f}" for x in stat['VM_Stds']]
            vm_table_data.append(vm_row)

        def create_and_save_table(data, filename, title, figsize=(10, 2)):
            from matplotlib.font_manager import FontProperties

            # Set up the figure and axis
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis('off')  # Hide the axes
            ax.set_title(title, fontweight="bold") 

            # Initialize cell text and formatting arrays
            cell_text = []
            cell_text_format = []  # Boolean array for whether cell should be bold

            # Populate cell text and format arrays
            for row in data:
                cell_row = []
                format_row = []
                for item in row:
                    # Assuming item is just data if not involving p-value comparison, or a tuple (text, needs_bold)
                    if isinstance(item, tuple):
                        print(item)
                        cell_row.append(item[0])
                        format_row.append(item[1])
                    else:
                        cell_row.append(item)
                        format_row.append(False)
                cell_text.append(cell_row)
                cell_text_format.append(format_row)

            from pprint import pprint
            pprint(cell_text)
            pprint(cell_text_format)

            # Create the table
            table = ax.table(cellText=cell_text, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)

            # Apply bold formatting based on cell_text_format
            for i, row in enumerate(cell_text_format):  # Iterate over data rows
                for j, should_bold in enumerate(row):  # Iterate over each cell in the row
                    if should_bold:
                        # Get the cell from the table
                        cell = table[(i, j)]
                        # Set the font properties to bold
                        cell.get_text().set_fontproperties(FontProperties(family='sans', weight='bold'))
                        cell.get_text().set_fontweight('bold')
                        cell.get_text().set_color('red')  # Change text color to red for testing
                        cell.get_text().set_size(14)      # Increase text size for testing
                        print('bolded', cell.get_text())
            
            plt.tight_layout()
            plt.savefig(filename + ".png")
            plt.close()
        def create_and_save_table_checked(data, filename, title):
            # Check data consistency
            expected_columns = len(data[0])  # Assumes the first row is the header and has correct number of columns
            for i, row in enumerate(data):
                if len(row) != expected_columns:
                    raise ValueError(f"Row {i} has incorrect length: {len(row)} elements (expected {expected_columns})")

            # Create and save the table if data is consistent
            create_and_save_table(data, filename, title)

        # Create and save the TD table
        create_and_save_table_checked(td_table_data, 'TD_Statistics', 'Typically Developing Subjects Statistics')

        # Create and save the VM table
        create_and_save_table_checked(vm_table_data, 'VM_Statistics', 'Ventriculomegaly Subjects Statistics')


        # print("Problematic row data:", u_p_values_table_data[1])

        # # Create and save the U & P Values table
        # create_and_save_table_checked(u_p_values_table_data, 'U_P_Values', 'U & P Values')


        # Print or handle the aggregated statistics as needed
        for stat in all_stats:
            print(f"{stat['View']} View:")
            print(f"U Statistics (MSE, MAE, Anomaly): {stat['U_Stats']}")
            print(f"P-values (MSE, MAE, Anomaly): {stat['P_Values']}")
            print("\n")

        

        # Initialize table data with headers
        table_data = [['View', 'U (MSE)', 'U (MAE)', 'U (Anomaly)', 'P (MSE)', 'P (MAE)', 'P (Anomaly)']]

        # Populate the table data with your statistics
        for stat in all_stats:
            # Process each row's p-values
            formatted_p_values = [format_p_value(p) for p in stat['P_Values']]
            # Append the row to your table data
            table_data.append([stat['View']] + stat['U_Stats'] + formatted_p_values)

        create_and_save_table_checked(table_data, 'U_P_Values', 'U & P Values')

        




        # Combine data from all views into a single DataFrame
        combined_df = pd.concat(view_data, ignore_index=True)

        # Transform data to long format for Altair
        long_df = combined_df.melt(id_vars=['View', 'Class', 'Slide_ID'], 
                                value_vars=['MSE', 'MAE', 'Anomaly'], 
                                var_name='Metric', value_name='Value')

        # Create violin plots for each metric, layered by view
        violin_plots = alt.Chart(long_df).transform_density(
            'Value',
            as_=['Value', 'density'],
            extent=[long_df['Value'].min(), long_df['Value'].max()],
            groupby=['View', 'Metric']
        ).mark_area(orient='horizontal').encode(
            y=alt.Y('Value:Q'),
            color=alt.Color('View:N', legend=alt.Legend(title="View")),
            x=alt.X(
                'density:Q',
                stack='center',
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            column=alt.Column('Metric:N', header=alt.Header(title="Mann-Whitney U Test"))
        ).properties(
            width=500,
            height=500
        )

        # Save the violin plot
        violin_plots.save(f'{self.val_path}layered_violin_plots.html')



    def multiview_AUROC_AS(self):
        from sklearn.metrics import roc_auc_score, roc_curve, auc
        from scipy import interp
        from numpy import linspace, mean
        
        base_path = self.val_path[:-2]  
        print(f'{base_path=}')

        # Load validation data for both views
        df_S = pd.read_csv(f'{base_path}S/validation.txt', sep=', ')
        df_A = pd.read_csv(f'{base_path}A/validation.txt', sep=', ')

        # Filter for Slide_ID 15 for both views
        td_df_S = df_S[(df_S['Class'] == 'td') & (df_S['Slide_ID'] == 15)]
        vm_df_S = df_S[(df_S['Class'] == 'vm') & (df_S['Slide_ID'] == 15)]
        td_df_A = df_A[(df_A['Class'] == 'td') & (df_A['Slide_ID'] == 15)]
        vm_df_A = df_A[(df_A['Class'] == 'vm') & (df_A['Slide_ID'] == 15)]

        # Ensure the data is aligned if necessary (e.g., by sorting by an identifier if the rows don't match up)

        # Combine anomaly scores from both views by averaging
        scores_L = np.concatenate((td_df_S['Anomaly'].values, vm_df_S['Anomaly'].values))
        scores_A = np.concatenate((td_df_A['Anomaly'].values, vm_df_A['Anomaly'].values))
        multiview_scores = (scores_L + scores_A) / 2

        # Prepare labels (same for both views)
        y_true = np.concatenate((np.zeros(len(td_df_S)), np.ones(len(vm_df_S))))  # Assuming both have same length

        # Compute AUROC for multiview scores
        multiview_auroc = roc_auc_score(y_true, multiview_scores)
        print(f"Multiview AUROC (Axial & Coronal): {multiview_auroc}")

        # Generate and plot ROC curve for multiview scores
        fpr, tpr, thresholds = roc_curve(y_true, multiview_scores)
        base_fpr = linspace(0, 1, 101)
        smooth_tpr = interp(base_fpr, fpr, tpr)
        smooth_tpr[0] = 0.0

        plt.figure(figsize=(8, 6))
        plt.plot(base_fpr, smooth_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {multiview_auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Receiver Operating Characteristic for Coronal-Axial Multiview')
        plt.legend(loc="lower right")
        plt.savefig(f'{base_path}Axial_Coronal_Multiview_AUROC.png')


    def multiview_AUROC_ASL(self):

        from sklearn.metrics import roc_auc_score, roc_curve, auc
        from scipy import interp
        from numpy import linspace, mean
        import numpy as np  # Ensure numpy is imported
        import pandas as pd  # Ensure pandas is imported
        import matplotlib.pyplot as plt  # Ensure matplotlib is imported
        
        base_path = self.val_path[:-2]  # Assume the path ends with 'L/', 'A/', or 'S/'

        # Load validation data for all three views
        df_L = pd.read_csv(f'{base_path}L/validation.txt', sep=', ')
        df_A = pd.read_csv(f'{base_path}A/validation.txt', sep=', ')
        df_S = pd.read_csv(f'{base_path}S/validation.txt', sep=', ')

        # Filter for Slide_ID 15 for all views
        td_df_L = df_L[(df_L['Class'] == 'td') & (df_L['Slide_ID'] == 15)]
        vm_df_L = df_L[(df_L['Class'] == 'vm') & (df_L['Slide_ID'] == 15)]
        td_df_A = df_A[(df_A['Class'] == 'td') & (df_A['Slide_ID'] == 15)]
        vm_df_A = df_A[(df_A['Class'] == 'vm') & (df_A['Slide_ID'] == 15)]
        td_df_S = df_S[(df_S['Class'] == 'td') & (df_S['Slide_ID'] == 15)]
        vm_df_S = df_S[(df_S['Class'] == 'vm') & (df_S['Slide_ID'] == 15)]

        # Combine anomaly scores from all three views by averaging
        scores_L = np.concatenate((td_df_L['Anomaly'].values, vm_df_L['Anomaly'].values))
        scores_A = np.concatenate((td_df_A['Anomaly'].values, vm_df_A['Anomaly'].values))
        scores_S = np.concatenate((td_df_S['Anomaly'].values, vm_df_S['Anomaly'].values))
        multiview_scores = (scores_L + scores_A + scores_S) / 3

        # Prepare labels (assumed same for all views)
        y_true = np.concatenate((np.zeros(len(td_df_L)), np.ones(len(vm_df_L))))  # Assuming all have same length

        # Compute AUROC for multiview scores
        multiview_auroc = roc_auc_score(y_true, multiview_scores)
        print(f"Multiview AUROC (Sagittal, Coronal, & Axial): {multiview_auroc}")

        # Generate and plot ROC curve for multiview scores
        fpr, tpr, thresholds = roc_curve(y_true, multiview_scores)
        base_fpr = linspace(0, 1, 101)
        smooth_tpr = interp(base_fpr, fpr, tpr)
        smooth_tpr[0] = 0.0

        plt.figure(figsize=(8, 6))
        plt.plot(base_fpr, smooth_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {multiview_auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Receiver Operating Characteristic for ASL Multiview')
        plt.legend(loc="lower right")
        plt.savefig(f'{base_path}Multiview_AUROC.png')

    # TODO Cross Validation

    def crossValidation():
        from sklearn.model_selection import KFold, cross_val_score 
        pass
        