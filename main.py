# Code written by @simonamador & @GuillermoTafoya

# from train_framework import Trainer
from train_framework_new import Trainer
from train_framework_new_new import Trainer as  GANTrainer
from train_cycle import Trainer as CycleTrainer
from validation import Validator
from visualizer import Visualizer
from utils.config import *

import os, torch



if __name__ == "__main__":
    # Obtain all configs from parser
    parser = settings_parser()
    args = parser.parse_args()

    # Establish CUDA GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print(f'{torch.cuda.is_available()=}')
    print(f'{torch.cuda.device_count()=}')

    # Obtain all paths needed for training/validation from config
    source_path, model_path, tensor_path, image_path, pre_path = path_generator(args)

    # Convert args to a dictionary and add device and paths dynamically
    parameters = vars(args).copy()
    parameters['device'] = device.type  
    parameters['source_path'] = source_path
    parameters['model_path'] = model_path
    parameters['tensor_path'] = tensor_path
    parameters['image_path'] = image_path
    parameters['pretrained_path'] = pre_path


    if args.task == 'Train':
        trainer = Trainer(parameters)
        trainer.train(args.epochs, args.loss)
        # trainer = Trainer(source_path, model_path, tensor_path,
        #                 image_path, device, args.batch, args.z_dim, args.ga_method, args.type, 
        #                 args.VAE_model_type, args.view, args.slice_size, args.pretrained, pre_path, args.ga_n, args.raw, args.th)
        # trainer.train(args.epochs, args.loss)
    elif args.task == 'Train_CycleGAN':
        trainer = CycleTrainer(parameters)
        trainer.train(args.epochs, args.loss)

    elif args.task == 'Train_GAN':
        trainer = GANTrainer(parameters)
        trainer.train(args.epochs, args.loss)

    elif args.task == 'Validate':
        # validator = Validator(args.path, model_path, args.model, args.type, args.view, args.ga_method, 
        #             args.z_dim, args.name, args.slice_size, device, args.training_folder, args.ga_n, args.raw, args.th, args.cGAN)


        validator = Validator(parameters)
        validator.validation()
        if parameters['VAE_model_type'] == 'ga_VAE':
           validator.age_differential(delta_ga=5)
           validator.age_differential(delta_ga=10)
        validator.mannwhitneyu()
        validator.AUROC()
        # validator.multiview_AUROC_ASL()
        # validator.multiview_AUROC_AS()
        validator.calculate_multiview_auroc_and_mannwhitneyu()
    elif args.task == 'Visualize':
        visualizer = Visualizer(parameters, task='gan')
        # visualizer.visualize_age_effect()
        # visualizer.save_reconstruction_images()
        # visualizer.save_whole_range_plus_refined(TD=True)
        # visualizer.save_whole_range_plus_refined(TD=False)
        visualizer.save_reconstruction_images_GAN(TD=True)
        visualizer.save_reconstruction_images_GAN(TD=False)
