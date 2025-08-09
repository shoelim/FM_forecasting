import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from model.model_endtoend import Model_EndtoEnd 
from model.model import Model
from model.model import metric
from utils.get_data import SimpleFlowDataset, EvalSimpleFlowDataset, ShalloWaterDataset, EvalShallowWaterDataset, ReacDiffDataset, EvalReacDiffDataset, NavierStokesDataset, EvalNavierStokesDataset

import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from transformers import get_polynomial_decay_schedule_with_warmup
from argparse import ArgumentParser, Namespace as ArgsNamespace
import scipy.stats
import time


def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='simpleflow', help="Name of dataset.")
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")
    parser.add_argument("--probpath_option", type=str, default='ours', help="Options for choosing probability path and vector field.")
    parser.add_argument("--train_option", type=str, default='end-to-end', help="Options for choosing training scheme, either end-to-end, separate or no_ae")
    parser.add_argument("--ae_option", type=str, default='ae', help="Options for choosing autoencoders.")
    parser.add_argument("--solver", type=str, default='rk4', help="Options for choosing sampler scheme.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Sigma for our method.")
    parser.add_argument("--sigma_min", type=float, default=0.001, help="Sigma_min for our method.")
    parser.add_argument("--sigma_sam", type=float, default=0.0, help="Sigma_sam for our method.")

    # AE parameters (for end-to-end training) 
    parser.add_argument("--enc_mid_channels", type=int, default=64, help="Number of channels in the encoder.")
    parser.add_argument("--dec_mid_channels", type=int, default=128, help="Number of channels in the decoder.")

    # paths 
    parser.add_argument("--path_to_ae_checkpoints", type=str, default='checkpoints/', help="Path to the checkpoints of pre-trained ae.")
    parser.add_argument("--path_to_results", type=str, default='', help="Path to save the results.")

    # optimizer parameters
    parser.add_argument("--learning-rate", type=float, default=0.00005, help="Learning rate.") 
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay.")
    parser.add_argument("--grad_acc", type=bool, default=False, help="Gradient accumulation.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Test batch size.")
    parser.add_argument("--snapshots-per-sample", type=int, default=25, help="Number of snapshots per sample.")

    # evaluation options
    parser.add_argument("--sampling_steps", type=int, default=100, help="Number of integration steps.")    
    parser.add_argument("--eval-freq", type=int, default=500, help="Number of snapshots per sample.")    
    parser.add_argument("--condition-snapshots", type=int, default=5, help="Number of snapshots per sample.")
    parser.add_argument("--snapshots-to-generate", type=int, default=20, help="Number of snapshots per sample.") 
    parser.add_argument("--N_test", type=int, default=5, help="Number of samples to generate for computation of evaluation metrics.") 
    
    return parser.parse_args()


def train(args, device):
    if args.dataset == 'simpleflow':
        in_channels = 1 
        out_channels = 1
        state_size = 4
        enc_mid_channels = args.enc_mid_channels
        dec_mid_channels = args.dec_mid_channels
        state_res = [8,8]
        datasets = {}
        for key in ["train", "val"]:
            datasets[key] = SimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets["test"] = EvalSimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)

        train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    elif args.dataset == 'shallowwater':
        in_channels = 1 
        out_channels = 1
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_res = [16,16]
        state_size = 4
        datasets = {}
        for key in ["train", "val"]:
            datasets[key] = ShalloWaterDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets["test"] = EvalShallowWaterDataset(snapshots_per_sample=args.snapshots_per_sample)

        train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    elif args.dataset == 'reacdiff':
        in_channels = 2 
        out_channels = 2
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_res = [16,16]
        state_size = 4
        datasets = {}
        for key in ["train", "val"]:
            datasets[key] = ReacDiffDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets["test"] = EvalReacDiffDataset(snapshots_per_sample=args.snapshots_per_sample)

        train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    elif args.dataset == 'navierstokes':
        in_channels = 2
        out_channels = 2
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_res = [64,64]
        state_size = 8
        datasets = {}
        for key in ["train", "val"]:
            datasets[key] = NavierStokesDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets["test"] = EvalNavierStokesDataset(snapshots_per_sample=args.snapshots_per_sample)
        
        train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)   
    else:
        raise ValueError('Invalid dataset option!')


    # Setup losses
    flow_matching_mse_loss_fun = nn.MSELoss()
    ae_mse_loss_fun = nn.MSELoss()  
    val_mse_loss_fun = nn.MSELoss()  
    test_mse_loss_fun = nn.MSELoss()


    # Setup model and distribute across gpus
    if args.train_option == "end-to-end":
        model = Model_EndtoEnd(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('... using: ', args.probpath_option)
        print('... with (for ours) sigma=', args.sigma)
        print('... with (for ours) sigma_min=', args.sigma_min)
        print('... with sampler=', args.solver)
        print('... with sampling steps=', args.sampling_steps)
        print('... with (for ours) sigma_sam=', args.sigma_sam)
    elif args.train_option == "separate":
        print("loading ae model...")
        if args.enc_mid_channels == 64 and args.dec_mid_channels == 128:
            AE_PATH = args.path_to_ae_checkpoints + args.dataset + ".pt"
        else:
            AE_PATH = args.path_to_ae_checkpoints + args.dataset + "_enc" + str(args.enc_mid_channels) + "_dec" + str(args.dec_mid_channels) + ".pt"
        
        ae_model = torch.load(AE_PATH, weights_only=False)
        ae_model.eval()
        if args.ae_option == "ae":
            model = Model(ae_model.encoder, ae_model.decoder, state_size=state_size, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
            model = model.to(device)
        else:
            raise ValueError('Invalid ae option!')

        total_params_ae = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
        print('# AE parameters: ', total_params_ae)

        total_params_flow = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('# trainable parameters for flow matching model: ', total_params_flow)
        print('... using: ', args.probpath_option)
        print('... with (for ours) sigma=', args.sigma)
        print('... with (for ours) sigma_min=', args.sigma_min)
        print('... with sampler=', args.solver)
        print('... with sampling steps=', args.sampling_steps)
        print('... with (for ours) sigma_sam=', args.sigma_sam)
    elif args.train_option == "no_ae":
        print("working on original data space...")
        #will run out of memory if the spatial resolution is very large
        model = Model(state_size=in_channels, state_res=[4,4], ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
        model = model.to(device)

        total_params_flow = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('# trainable parameters for flow matching model: ', total_params_flow)
        print('... using: ', args.probpath_option)
        print('... with (for ours) sigma=', args.sigma)
        print('... with (for ours) sigma_min=', args.sigma_min)
        print('... with sampler=', args.solver)
        print('... with sampling steps=', args.sampling_steps)
        print('... with (for ours) sigma_sam=', args.sigma_sam)
    else:
        raise ValueError('Invalid training option!')


    # Setup optimizer and scheduler if not yet
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999))

    # Gradient accumulation if needed
    if args.grad_acc: #rarely needed unless the model is very large
        accumulation_steps = 32
        num_update_steps_per_epoch = len(train_loader) // accumulation_steps
        total_training_steps = num_update_steps_per_epoch * args.epochs 
        num_warmup_steps = int(0.1 * total_training_steps)
        effective_training_steps = (len(train_loader) * args.epochs) // accumulation_steps
        effective_warmup_steps = (len(train_loader) * 5) // accumulation_steps

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=effective_warmup_steps,
            num_training_steps=effective_training_steps
        )
    else: 
        accumulation_steps = 1
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=len(train_loader) * 5, # we need only a very shot warmup phase for our data
        num_training_steps=(len(train_loader) * args.epochs))

    # Start training
    train_loss = []
    val_loss = []
    test_loss = []
    start_time = time.time()

    if args.grad_acc:
        print(f"Using gradient accumulation with {accumulation_steps} steps.")
        for epoch in range(args.epochs):
            model.train()
            train_gen = tqdm(train_loader, desc="Training")

            optimizer.zero_grad() 

            for step, batch in enumerate(train_gen):
                observations = batch.cuda()
                batch_size = observations.size(0)  

                if args.train_option == "end-to-end":
                    input_snapshots, reconstructed_snapshots, target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
                    flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
                    ae_loss = ae_mse_loss_fun(input_snapshots, reconstructed_snapshots)
                    loss = flow_matching_loss / accumulation_steps + ae_loss
                elif args.train_option == "separate":
                    target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
                    flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
                    ae_loss = 0.0
                    loss = flow_matching_loss / accumulation_steps
                elif args.train_option == "no_ae":
                    target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
                    flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)            
                    ae_loss = 0.0
                    loss = flow_matching_loss / accumulation_steps

                loss.backward()

                # Optimizer step + lr scheduler step every accumulation_steps batches
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Handle leftover batches at epoch end
            if (step + 1) % accumulation_steps != 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            print(f"epoch={epoch}, mse_total={loss * accumulation_steps}, ae_loss={ae_loss}, fm_loss={flow_matching_loss}")
            train_loss.append((loss * accumulation_steps).item())

            # --------------------
            # Validation
            # --------------------
              # Evaluate the model
            if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:  
                model.eval()

                # Setup loading bar
                val_gen = tqdm(val_loader, desc="Validation")
                for i, batch in enumerate(val_gen):
                    if i >= 1:
                        break

                    # Fetch data
                    observations = batch.cuda()
                    targets = observations

                    # Log media
                    generated_observations = model.generate_snapshots(
                        observations=observations[:, :args.condition_snapshots],
                        num_condition_snapshots=args.condition_snapshots, 
                        num_snapshots=args.snapshots_to_generate, 
                        steps=args.sampling_steps, 
                        option=args.probpath_option)
                
                    val_mse = val_mse_loss_fun(targets, generated_observations)  
                    
                    # plotting
                    nrow = 2; ncol = args.condition_snapshots + args.snapshots_to_generate;
                    obs = observations.cpu().numpy()
                    gen = generated_observations.cpu().numpy()
                    
        
                    f, axarr = plt.subplots(nrow, ncol, figsize=(25, 8))
                    for i in range(ncol): 
                        axarr[0,i].imshow(obs[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
                        axarr[0,i].set_xticks([])
                        axarr[0,i].set_yticks([])
                        if i < args.condition_snapshots:
                            axarr[0,i].title.set_text("GT (cond.)")  
                        else:
                            axarr[0,i].title.set_text("GT (pred.)")  

                    
                    for i in range(ncol): 
                        axarr[1,i].imshow(gen[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
                        axarr[1,i].set_xticks([])
                        axarr[1,i].set_yticks([])
                        
                    plt.tight_layout()   
                    directory = args.path_to_results + args.run_name
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    savename = args.run_name + '/evaluation_epoch=' + str(epoch) + '.pdf'
                    plt.savefig(args.path_to_results + savename)

                print(f"epoch={epoch}, val_mse={val_mse}")
                val_loss.append(val_mse.item())

                # Setup loading bar     
                test_gen = tqdm(test_loader, desc="Testing")
                for i, batch in enumerate(test_gen):
                    if i >= 1:
                        break

                    # Fetch data
                    test_observations = batch.cuda()
                    targets = test_observations 

                    # Log media
                    generated_observations_test = model.generate_snapshots(
                        observations=test_observations[:, :args.condition_snapshots],
                        num_condition_snapshots=args.condition_snapshots, 
                        num_snapshots=args.snapshots_to_generate, 
                        steps=args.sampling_steps, 
                        solver=args.solver, 
                        option=args.probpath_option)
                    test_mse = test_mse_loss_fun(targets, generated_observations_test) 

                    nrow_test = 2; ncol_test = args.condition_snapshots + args.snapshots_to_generate;
                    obs_test = test_observations.cpu().numpy()
                    gen_test = generated_observations_test.cpu().numpy()
                    
                    f, axarr = plt.subplots(nrow_test, ncol_test, figsize=(25, 8))
                    for i in range(ncol_test): 
                        axarr[0,i].imshow(obs_test[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
                        axarr[0,i].set_xticks([])
                        axarr[0,i].set_yticks([])
                        if i < args.condition_snapshots:
                            axarr[0,i].title.set_text("GT (cond.)")  
                        else:
                            axarr[0,i].title.set_text("GT (pred.)")  
                        
                    for i in range(ncol_test): 
                        axarr[1,i].imshow(gen_test[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
                        axarr[1,i].set_xticks([])
                        axarr[1,i].set_yticks([])
        
                    plt.tight_layout()   
                    directory = args.path_to_results + args.run_name
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    savename = args.run_name + '/test_epoch=' + str(epoch) + '.pdf'
                    plt.savefig(args.path_to_results + savename)
                    plt.close('all')
                
                print(f"epoch={epoch}, test_mse={test_mse}")
                test_loss.append(test_mse.item())

    else:
        print("Not using gradient accumulation.")
        for epoch in range(args.epochs): 
            model.train()
            train_gen = tqdm(train_loader, desc="Training")
            for batch in train_gen:
                # Fetch data
                observations = batch.cuda()
                batch_size = observations.size(0)
            
                if args.train_option == "end-to-end":
                    input_snapshots, reconstructed_snapshots, target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
                    flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
                    ae_loss = ae_mse_loss_fun(input_snapshots, reconstructed_snapshots)
                    loss = flow_matching_loss/accumulation_steps + ae_loss
                elif args.train_option == "separate":
                    target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
                    flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
                    ae_loss = 0.0
                    loss = flow_matching_loss/accumulation_steps 
                elif args.train_option == "no_ae":
                    target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
                    flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)            
                    ae_loss = 0.0
                    loss = flow_matching_loss/accumulation_steps
            
                # Backward pass
                model.zero_grad()
                loss.backward()
            
                # Optimizer step
                optimizer.step()
            
            # update learning rate
            lr_scheduler.step()

            print(f"epoch={epoch}, mse_total={loss}, ae_loss={ae_loss}, fm_loss={flow_matching_loss}")
            train_loss.append(loss.item())


            # Evaluate the model
            if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:  
                model.eval()

                # Setup loading bar
                val_gen = tqdm(val_loader, desc="Validation")
                for i, batch in enumerate(val_gen):
                    if i >= 1:
                        break

                    # Fetch data
                    observations = batch.cuda()
                    targets = observations 

                    # Log media
                    generated_observations = model.generate_snapshots(
                        observations=observations[:, :args.condition_snapshots],
                        num_condition_snapshots=args.condition_snapshots, 
                        num_snapshots=args.snapshots_to_generate, 
                        steps=args.sampling_steps, 
                        option=args.probpath_option)
                
                    val_mse = val_mse_loss_fun(targets, generated_observations) 
                    
                    # plotting
                    nrow = 2; ncol = args.condition_snapshots + args.snapshots_to_generate;
                    obs = observations.cpu().numpy()
                    gen = generated_observations.cpu().numpy()
                    
        
                    f, axarr = plt.subplots(nrow, ncol, figsize=(25, 8))
                    for i in range(ncol): 
                        axarr[0,i].imshow(obs[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
                        axarr[0,i].set_xticks([])
                        axarr[0,i].set_yticks([])
                        if i < args.condition_snapshots:
                            axarr[0,i].title.set_text("GT (cond.)")  
                        else:
                            axarr[0,i].title.set_text("GT (pred.)")  

                    
                    for i in range(ncol): 
                        axarr[1,i].imshow(gen[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
                        axarr[1,i].set_xticks([])
                        axarr[1,i].set_yticks([])
                        
                    plt.tight_layout()   
                    directory = args.path_to_results + args.run_name
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    savename = args.run_name + '/evaluation_epoch=' + str(epoch) + '.pdf'
                    plt.savefig(args.path_to_results + savename)

                print(f"epoch={epoch}, val_mse={val_mse}")
                val_loss.append(val_mse.item())

                # Setup loading bar     
                test_gen = tqdm(test_loader, desc="Testing")
                for i, batch in enumerate(test_gen):
                    if i >= 1:
                        break

                    # Fetch data
                    test_observations = batch.cuda()
                    targets = test_observations

                    # Log media
                    generated_observations_test = model.generate_snapshots(
                        observations=test_observations[:, :args.condition_snapshots],
                        num_condition_snapshots=args.condition_snapshots, 
                        num_snapshots=args.snapshots_to_generate, 
                        steps=args.sampling_steps, 
                        solver=args.solver, 
                        option=args.probpath_option)
                    test_mse = test_mse_loss_fun(targets, generated_observations_test)  

                    nrow_test = 2; ncol_test = args.condition_snapshots + args.snapshots_to_generate;
                    obs_test = test_observations.cpu().numpy()
                    gen_test = generated_observations_test.cpu().numpy()
                    
                    f, axarr = plt.subplots(nrow_test, ncol_test, figsize=(25, 8))
                    for i in range(ncol_test): 
                        axarr[0,i].imshow(obs_test[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
                        axarr[0,i].set_xticks([])
                        axarr[0,i].set_yticks([])
                        if i < args.condition_snapshots:
                            axarr[0,i].title.set_text("GT (cond.)")  
                        else:
                            axarr[0,i].title.set_text("GT (pred.)")  
                        
                    for i in range(ncol_test): 
                        axarr[1,i].imshow(gen_test[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
                        axarr[1,i].set_xticks([])
                        axarr[1,i].set_yticks([])
        
                    plt.tight_layout()   
                    directory = args.path_to_results + args.run_name
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    savename = args.run_name + '/test_epoch=' + str(epoch) + '.pdf'
                    plt.savefig(args.path_to_results + savename)
                    plt.close('all')
                
                print(f"epoch={epoch}, test_mse={test_mse}")
                test_loss.append(test_mse.item())

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    
    save_path = args.path_to_results + args.run_name 
    with open(save_path + "/train_loss.txt", "w") as f:
        for loss in train_loss:
            f.write(f"{loss}\n")
    with open(save_path + "/val_loss.txt", "w") as f:
        for loss in val_loss:
            f.write(f"{loss}\n")
    with open(save_path + "/test_loss.txt", "w") as f:
        for loss in test_loss:
            f.write(f"{loss}\n")
    
    
    for sampling_steps in [1, 2, 5, 10, 20, 50, 100]:
        model.eval()
        # testing
        print('-------------------------------------------------------------------------------')
        print('computing test results using sampling steps=', sampling_steps)  
        N_test = args.N_test
        gen_observations_list = []
        test_mse_list = [] 
        ssim_list = []
        psnr_list = []
        for i in range(N_test):
            targets = test_observations 
            generated_observations_new = model.generate_snapshots(
                observations=test_observations[:, :args.condition_snapshots],
                num_condition_snapshots=args.condition_snapshots, 
                num_snapshots=args.snapshots_to_generate, 
                steps=sampling_steps, 
                solver=args.solver, 
                option=args.probpath_option)
            gen_observations_list.append(generated_observations_new.cpu().numpy()) 
            test_mse_list.append(test_mse_loss_fun(targets, generated_observations_new).item())  
            ssim_, psnr_ = metric(generated_observations_new.cpu().numpy(), targets.cpu().numpy())    
            ssim_list.append(ssim_)
            psnr_list.append(psnr_)


        print('--------- computing test results ------------')
        print(f"test_mse={test_mse_list}")
        print(f"test_mse mean={np.mean(test_mse_list)}")
        print(f"test_mse std={np.std(test_mse_list)}")
        print(f"ssim={ssim_list}")
        print(f"ssim mean={np.mean(ssim_list)}")
        print(f"ssim std={np.std(ssim_list)}")
        print(f"psnr={psnr_list}")
        print(f"psnr mean={np.mean(psnr_list)}")
        print(f"psnr std={np.std(psnr_list)}")

        print(' ***** RUN IS FINISHED ***** ')


        #compute correlation and rfne
        batch_size = gen_observations_list[0].shape[0] 
        num_snapshots = gen_observations_list[0].shape[1]  

        # Function to compute the Pearson correlation for a given batch and snapshot index
        def compute_correlation(generated_observations_new_np, targets_np, b, j):
            return scipy.stats.pearsonr(
                generated_observations_new_np[b, j, 0, :, :].flatten(),
                targets_np[b, j, 0, :, :].flatten())[0]

        def compute_rfne(generated_observations_new_np, targets_np, b, j):
            error_norm = np.linalg.norm(generated_observations_new_np[b, j, 0, :, :] - targets_np[b, j, 0, :, :])
            target_norm = np.linalg.norm(targets_np[b, j, 0, :, :])
            return error_norm / target_norm


        # Initialize an array to store the cumulative average correlations across all N_test
        average_correlations_all_tests = np.zeros(num_snapshots)
        average_rfne_all_tests = np.zeros(num_snapshots)

        rfne_list = []
        corr_list = []
        # Loop over each test set in generated_observations_new_np (list of N_test items)
        for t in range(N_test):
            current_test_set = gen_observations_list[t]  # Extract the current test set
            
            # Initialize a list to store correlations for each batch and snapshot for the current test set
            correlations_per_batch = [
                [compute_correlation(current_test_set, targets.cpu().numpy(), b, j) for j in range(num_snapshots)]
                for b in range(batch_size)
            ]
            rfne_per_batch = [
                [compute_rfne(current_test_set, targets.cpu().numpy(), b, j) for j in range(num_snapshots)]
                for b in range(batch_size)
            ]
            
            # Convert to a numpy array for easier manipulation
            correlations_per_batch = np.array(correlations_per_batch)
            rfne_per_batch = np.array(rfne_per_batch)
            
            # Compute the average correlation for each snapshot across batches in the current test set
            average_correlations_per_snapshot = np.mean(correlations_per_batch, axis=0)
            average_rfne_per_snapshot = np.mean(rfne_per_batch, axis=0)
            corr_list.append(average_correlations_per_snapshot)
            rfne_list.append(average_rfne_per_snapshot)
            # Add the result to the cumulative average (across all N_test sets)
            average_correlations_all_tests += average_correlations_per_snapshot
            average_rfne_all_tests += average_rfne_per_snapshot

        # Divide by N_test to get the final average correlations across all test sets
        average_correlations_all_tests /= N_test
        average_rfne_all_tests /= N_test


        print(f"average_correlations={average_correlations_all_tests}")
        print(f"mean of average_correlations={np.mean(average_correlations_all_tests)}")
        print(f"std of average_correlations={np.std(corr_list)}")
       
        print(f"average_rfne={average_rfne_all_tests}")
        print(f"mean of average_rfne={np.mean(average_rfne_all_tests)}")
        print(f"std of average_rfne={np.std(rfne_list)}")
       

def main():
    args = parse_args()

    # Launch processes.
    print('Launching processes...')

    # Initialize
    np.random.seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    train(args, device)


if __name__ == "__main__":
    main() 