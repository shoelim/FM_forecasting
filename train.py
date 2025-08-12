import os
import io
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup
from transformers import get_polynomial_decay_schedule_with_warmup
from argparse import ArgumentParser, Namespace as ArgsNamespace
import scipy.stats

from model.model_endtoend import Model_EndtoEnd 
from model.model import Model
from model.model import metric
from utils.get_data import (
    SimpleFlowDataset, EvalSimpleFlowDataset, 
    ShalloWaterDataset, EvalShallowWaterDataset, 
    ReacDiffDataset, EvalReacDiffDataset, 
    NavierStokesDataset, EvalNavierStokesDataset
)


def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    
    # Basic configuration
    parser.add_argument("--run-name", type=str, required=True, 
                       help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='simpleflow', 
                       help="Name of dataset.")
    parser.add_argument("--random-seed", type=int, default=1543, 
                       help="Random seed.")
    
    # Model options
    parser.add_argument("--probpath_option", type=str, default='ours', 
                       help="Options for choosing probability path and vector field.")
    parser.add_argument("--train_option", type=str, default='end-to-end', 
                       help="Options for choosing training scheme, either end-to-end, separate or no_ae")
    parser.add_argument("--ae_option", type=str, default='ae', 
                       help="Options for choosing autoencoders.")
    parser.add_argument("--solver", type=str, default='rk4', 
                       help="Options for choosing sampler scheme.")
    
    # Sigma parameters
    parser.add_argument("--sigma", type=float, default=0.01, 
                       help="Sigma for our method.")
    parser.add_argument("--sigma_min", type=float, default=0.001, 
                       help="Sigma_min for our method.")
    parser.add_argument("--sigma_sam", type=float, default=0.0, 
                       help="Sigma_sam for our method.")

    # AE parameters (for end-to-end training) 
    parser.add_argument("--enc_mid_channels", type=int, default=64, 
                       help="Number of channels in the encoder.")
    parser.add_argument("--dec_mid_channels", type=int, default=128, 
                       help="Number of channels in the decoder.")

    # Paths 
    parser.add_argument("--path_to_ae_checkpoints", type=str, default='checkpoints/', 
                       help="Path to the checkpoints of pre-trained ae.")
    parser.add_argument("--path_to_results", type=str, default='', 
                       help="Path to save the results.")

    # Optimizer parameters
    parser.add_argument("--learning-rate", type=float, default=0.00005, 
                       help="Learning rate.") 
    parser.add_argument("--weight-decay", type=float, default=0, 
                       help="Weight decay.")
    parser.add_argument("--grad_acc", type=bool, default=False, 
                       help="Gradient accumulation.")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of epochs.")
    parser.add_argument("--warmup_steps", type=int, default=500, 
                       help="Warmup steps.")
    parser.add_argument("--train_batch_size", type=int, default=32, 
                       help="Train batch size.")
    parser.add_argument("--test_batch_size", type=int, default=32, 
                       help="Test batch size.")
    parser.add_argument("--snapshots-per-sample", type=int, default=25, 
                       help="Number of snapshots per sample.")

    # Evaluation options
    parser.add_argument("--sampling_steps", type=int, default=10, 
                       help="Number of integration steps.")     
    parser.add_argument("--condition-snapshots", type=int, default=5, 
                       help="Number of snapshots per sample.")
    parser.add_argument("--snapshots-to-generate", type=int, default=20, 
                       help="Number of snapshots per sample.") 
    parser.add_argument("--N_test", type=int, default=5, 
                       help="Number of samples to generate for computation of evaluation metrics.") 
        
    # Plotting and saving options
    parser.add_argument("--save-plots", action='store_true', 
                       help="Enable saving of evaluation plots as PDFs.")
    parser.add_argument("--save-gif", action='store_true', 
                       help="Enable saving of evaluation animations as GIFs.")

    return parser.parse_args()


def setup_dataset_and_model(args):
    """Setup dataset, data loaders, and model based on dataset type."""
    
    if args.dataset == 'simpleflow':
        in_channels = 1 
        out_channels = 1
        state_size = 4
        enc_mid_channels = args.enc_mid_channels
        dec_mid_channels = args.dec_mid_channels
        state_res = [8, 8]
        
    elif args.dataset == 'shallowwater':
        in_channels = 1 
        out_channels = 1
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_res = [16, 16]
        state_size = 4
        
    elif args.dataset == 'reacdiff':
        in_channels = 2 
        out_channels = 2
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_res = [16, 16]
        state_size = 4
        
    elif args.dataset == 'navierstokes':
        in_channels = 2
        out_channels = 2
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_res = [64, 64]
        state_size = 8
        
    else:
        raise ValueError('Invalid dataset option!')
    
    # Setup datasets
    datasets = {}
    if args.dataset == 'simpleflow':
        datasets['train'] = SimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets['val'] = SimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets['test'] = EvalSimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)
    elif args.dataset == 'shallowwater':
        datasets['train'] = ShalloWaterDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets['val'] = ShalloWaterDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets['test'] = EvalShallowWaterDataset(snapshots_per_sample=args.snapshots_per_sample)
    elif args.dataset == 'reacdiff':
        datasets['train'] = ReacDiffDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets['val'] = ReacDiffDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets['test'] = EvalReacDiffDataset(snapshots_per_sample=args.snapshots_per_sample)
    elif args.dataset == 'navierstokes':
        datasets['train'] = NavierStokesDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets['val'] = NavierStokesDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets['test'] = EvalNavierStokesDataset(snapshots_per_sample=args.snapshots_per_sample)

    # Setup data loaders
    train_loader = DataLoader(
        dataset=datasets['train'], 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=datasets['val'], 
        batch_size=args.train_batch_size, 
        shuffle=False, 
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=datasets['test'], 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return (in_channels, out_channels, state_size, enc_mid_channels, 
            dec_mid_channels, state_res, train_loader, val_loader, test_loader)


def setup_model(args, device, in_channels, out_channels, state_size, 
                enc_mid_channels, dec_mid_channels, state_res):
    """Setup model based on training option."""
    
    if args.train_option == "end-to-end":
        model = Model_EndtoEnd(
            state_size=state_size, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            enc_mid_channels=enc_mid_channels, 
            dec_mid_channels=dec_mid_channels, 
            state_res=state_res, 
            ours_sigma=args.sigma, 
            sigma_min=args.sigma_min, 
            sigma_sam=args.sigma_sam
        )
        model = model.to(device)
        
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
            AE_PATH = (args.path_to_ae_checkpoints + args.dataset + 
                      "_enc" + str(args.enc_mid_channels) + 
                      "_dec" + str(args.dec_mid_channels) + ".pt")
        
        ae_model = torch.load(AE_PATH, weights_only=False)
        ae_model.eval()
        
        if args.ae_option == "ae":
            model = Model(
                ae_model.encoder, 
                ae_model.decoder, 
                state_size=state_size, 
                state_res=state_res, 
                ours_sigma=args.sigma, 
                sigma_min=args.sigma_min, 
                sigma_sam=args.sigma_sam
            )
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
        # will run out of memory if the spatial resolution is very large
        model = Model(
            state_size=in_channels, 
            state_res=[4, 4], 
            ours_sigma=args.sigma, 
            sigma_min=args.sigma_min, 
            sigma_sam=args.sigma_sam
        )
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
    
    return model


def setup_optimizer_and_scheduler(args, model, train_loader):
    """Setup optimizer and learning rate scheduler."""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Gradient accumulation if needed
    if args.grad_acc:  # rarely needed unless the model is very large
        accumulation_steps = 32
        effective_warmup_steps = (len(train_loader) * 5) // accumulation_steps
        effective_training_steps = (len(train_loader) * args.epochs) // accumulation_steps

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=effective_warmup_steps,
            num_training_steps=effective_training_steps
        )
    else: 
        accumulation_steps = 1
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=len(train_loader) * 5,  # very short warmup phase for our data
            num_training_steps=(len(train_loader) * args.epochs)
        )
    
    return optimizer, lr_scheduler, accumulation_steps


def create_gif_frames(observations, generated_observations, args, prefix="evaluation"):
    """Create frames for GIF animation."""
    frames = []
    ncol = args.condition_snapshots + args.snapshots_to_generate
    
    for j in range(ncol):
        fig_gif, ax_gif = plt.subplots(1, 2, figsize=(6, 3))
        
        # Ground Truth
        ax_gif[0].imshow(observations[0, j, 0, :, :], cmap='RdBu')
        ax_gif[0].set_title(f'Ground Truth (t={j})')
        ax_gif[0].set_xticks([])
        ax_gif[0].set_yticks([])
        
        # Generated
        ax_gif[1].imshow(generated_observations[0, j, 0, :, :], cmap='RdBu')
        ax_gif[1].set_title(f'Generated (t={j})')
        ax_gif[1].set_xticks([])
        ax_gif[1].set_yticks([])
        
        plt.tight_layout()
        
        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig_gif)
    
    return frames


def save_plots_and_gifs(args, epoch, observations, generated_observations, prefix="evaluation"):
    """Save plots and GIFs for evaluation."""
    
    # Create directory if it doesn't exist
    directory = args.path_to_results + args.run_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Plotting
    if args.save_plots:
        nrow = 2
        ncol = args.condition_snapshots + args.snapshots_to_generate
        obs = observations.cpu().numpy()
        gen = generated_observations.cpu().numpy()

        f, axarr = plt.subplots(nrow, ncol, figsize=(25, 8))
        
        for i in range(ncol): 
            axarr[0, i].imshow(obs[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
            axarr[0, i].set_xticks([])
            axarr[0, i].set_yticks([])
            if i < args.condition_snapshots:
                axarr[0, i].title.set_text("GT (cond.)")  
            else:
                axarr[0, i].title.set_text("GT (pred.)")  

        for i in range(ncol): 
            axarr[1, i].imshow(gen[0, i, 0, :, :], cmap=mpl.colormaps['RdBu'])
            axarr[1, i].set_xticks([])
            axarr[1, i].set_yticks([])
            
        plt.tight_layout()
        savename = f'{args.run_name}/{prefix}_epoch={epoch}.pdf'
        plt.savefig(args.path_to_results + savename)
        plt.close('all')
    
    # Create GIF
    if args.save_gif:
        frames = create_gif_frames(obs, gen, args, prefix)
        gif_savename = os.path.join(directory, f'{prefix}_epoch_{epoch}.gif')
        imageio.mimsave(gif_savename, frames, duration=0.2, loop=0)


def train_epoch_with_grad_acc(args, model, train_loader, optimizer, lr_scheduler, 
                             flow_matching_mse_loss_fun, ae_mse_loss_fun, accumulation_steps):
    """Train one epoch with gradient accumulation."""
    model.train()
    train_gen = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()

    for step, batch in enumerate(train_gen):
        observations = batch.cuda()

        if args.train_option == "end-to-end":
            input_snapshots, reconstructed_snapshots, target_vectors, reconstructed_vectors = model(
                observations, option=args.probpath_option
            )
            flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
            ae_loss = ae_mse_loss_fun(input_snapshots, reconstructed_snapshots)
            if args.grad_acc:
                loss = flow_matching_loss / accumulation_steps + ae_loss
            else:
                loss = flow_matching_loss + ae_loss
            
        elif args.train_option == "separate":
            target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
            flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
            ae_loss = 0.0
            if args.grad_acc:
                loss = flow_matching_loss / accumulation_steps 
            else:
                loss = flow_matching_loss
            
        elif args.train_option == "no_ae":
            target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
            flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)            
            ae_loss = 0.0
            if args.grad_acc:
                loss = flow_matching_loss / accumulation_steps 
            else:
                loss = flow_matching_loss
            
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
    
    return loss * accumulation_steps, ae_loss, flow_matching_loss


def train_epoch_regular(args, model, train_loader, optimizer, lr_scheduler, 
                       flow_matching_mse_loss_fun, ae_mse_loss_fun, accumulation_steps):
    """Train one epoch without gradient accumulation."""
    model.train()
    train_gen = tqdm(train_loader, desc="Training")
    
    for batch in train_gen:
        observations = batch.cuda()
    
        if args.train_option == "end-to-end":
            input_snapshots, reconstructed_snapshots, target_vectors, reconstructed_vectors = model(
                observations, option=args.probpath_option
            )
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
    
        # Backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    lr_scheduler.step()
    
    return loss, ae_loss, flow_matching_loss


def evaluate_model(args, model, data_loader, val_mse_loss_fun, epoch, prefix="evaluation"):
    """Evaluate model on validation or test set."""
    model.eval()
    
    # Setup loading bar
    eval_gen = tqdm(data_loader, desc=f"{prefix.capitalize()}")
    for i, batch in enumerate(eval_gen):
        if i >= 1:
            break

        # Fetch data
        observations = batch.cuda()
        targets = observations

        # Generate snapshots
        generated_observations = model.generate_snapshots(
            observations=observations[:, :args.condition_snapshots],
            num_condition_snapshots=args.condition_snapshots, 
            num_snapshots=args.snapshots_to_generate, 
            steps=args.sampling_steps, 
            solver=args.solver,
            option=args.probpath_option
        )
    
        eval_mse = val_mse_loss_fun(targets, generated_observations)
        
        # Save plots and GIFs
        save_plots_and_gifs(args, epoch, observations, generated_observations, prefix)
        
        print(f"epoch={epoch}, {prefix}_mse={eval_mse}")
        return eval_mse.item()


def compute_correlation(generated_observations_new_np, targets_np, b, j):
    """Compute the Pearson correlation for a given batch and snapshot index."""
    return scipy.stats.pearsonr(
        generated_observations_new_np[b, j, 0, :, :].flatten(),
        targets_np[b, j, 0, :, :].flatten()
    )[0]


def compute_rfne(generated_observations_new_np, targets_np, b, j):
    """Compute relative Frobenius norm error."""
    error_norm = np.linalg.norm(generated_observations_new_np[b, j, 0, :, :] - targets_np[b, j, 0, :, :])
    target_norm = np.linalg.norm(targets_np[b, j, 0, :, :])
    return error_norm / target_norm


def compute_test_metrics(args, model, test_observations, test_mse_loss_fun, sampling_steps):
    """Compute comprehensive test metrics for different sampling steps."""
    model.eval()
    
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
            option=args.probpath_option
        )
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

    # Compute correlation and RFNE
    batch_size = gen_observations_list[0].shape[0] 
    num_snapshots = gen_observations_list[0].shape[1]  

    # Initialize arrays to store the cumulative average correlations across all N_test
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


def save_losses(args, train_loss, val_loss, test_loss):
    """Save loss lists to text files."""
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


def train(args, device):
    """Main training function."""
    
    # Setup dataset and model
    (in_channels, out_channels, state_size, enc_mid_channels, 
     dec_mid_channels, state_res, train_loader, val_loader, test_loader) = setup_dataset_and_model(args)

    # Setup losses
    flow_matching_mse_loss_fun = nn.MSELoss()
    ae_mse_loss_fun = nn.MSELoss()  
    val_mse_loss_fun = nn.MSELoss()  
    test_mse_loss_fun = nn.MSELoss()

    # Setup model
    model = setup_model(args, device, in_channels, out_channels, state_size, 
                       enc_mid_channels, dec_mid_channels, state_res)

    # Setup optimizer and scheduler
    optimizer, lr_scheduler, accumulation_steps = setup_optimizer_and_scheduler(args, model, train_loader)

    # Start training
    train_loss = []
    val_loss = []
    test_loss = []
    start_time = time.time()

    if args.grad_acc:
        print(f"Using gradient accumulation with {accumulation_steps} steps.")
        
        for epoch in range(args.epochs):
            loss, ae_loss, flow_matching_loss = train_epoch_with_grad_acc(
                args, model, train_loader, optimizer, lr_scheduler, 
                flow_matching_mse_loss_fun, ae_mse_loss_fun, accumulation_steps
            )
            
            print(f"epoch={epoch}, mse_total={loss}, ae_loss={ae_loss}, fm_loss={flow_matching_loss}")
            train_loss.append(loss.item())

            # Evaluation
            if epoch == args.epochs - 1:
                val_mse = evaluate_model(args, model, val_loader, val_mse_loss_fun, epoch, "evaluation")
                val_loss.append(val_mse)
                
                test_mse = evaluate_model(args, model, test_loader, test_mse_loss_fun, epoch, "test")
                test_loss.append(test_mse)

    else:
        print("Not using gradient accumulation.")
        
        for epoch in range(args.epochs):
            loss, ae_loss, flow_matching_loss = train_epoch_regular(
                args, model, train_loader, optimizer, lr_scheduler, 
                flow_matching_mse_loss_fun, ae_mse_loss_fun, accumulation_steps
            )
            
            print(f"epoch={epoch}, mse_total={loss}, ae_loss={ae_loss}, fm_loss={flow_matching_loss}")
            train_loss.append(loss.item())

            # Evaluation
            if epoch == args.epochs - 1:
                val_mse = evaluate_model(args, model, val_loader, val_mse_loss_fun, epoch, "evaluation")
                val_loss.append(val_mse)
                
                test_mse = evaluate_model(args, model, test_loader, test_mse_loss_fun, epoch, "test")
                test_loss.append(test_mse)

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    
    # Save loss curves
    save_losses(args, train_loss, val_loss, test_loss)
    
    # Get test observations for comprehensive evaluation
    test_gen = tqdm(test_loader, desc="Getting test observations")
    for i, batch in enumerate(test_gen):
        if i >= 1:
            break
        test_observations = batch.cuda()
    
    # Comprehensive testing with different sampling steps
    for sampling_steps in [1, 2, 5, 10, 20, 50, 100]:
        compute_test_metrics(args, model, test_observations, test_mse_loss_fun, sampling_steps)

    print(' ***** RUN IS FINISHED ***** ')


def main():
    """Main function."""
    args = parse_args()

    # Launch processes
    print('Launching processes...')

    # Initialize random seeds
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