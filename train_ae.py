import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from model.model_ae import Model_AE
from utils.get_data import SimpleFlowDataset, EvalSimpleFlowDataset, ShalloWaterDataset, EvalShallowWaterDataset, ReacDiffDataset, EvalReacDiffDataset, NavierStokesDataset, EvalNavierStokesDataset

import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from transformers import get_polynomial_decay_schedule_with_warmup
from argparse import ArgumentParser, Namespace as ArgsNamespace


def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='simpleflow', help="Name of dataset.")
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")
    parser.add_argument("--ae_option", type=str, default='ae', help="Options for choosing autoencoders.")
    parser.add_argument("--enc_mid_channels", type=int, default=64, help="Number of channels in the encoder.")
    parser.add_argument("--dec_mid_channels", type=int, default=128, help="Number of channels in the decoder.")
    parser.add_argument("--ae_learning_rate", type=float, default=0.00005, help="Learning rate for AE.") 
    parser.add_argument("--ae_weight_decay", type=float, default=0, help="Weight decay for AE.")
    parser.add_argument("--ae_lr_scheduler", type=str, default='exponential', help="Options for learning rate scheduler for AE.")
    parser.add_argument("--ae_epochs", type=int, default=10001, help="Number of epochs for training ae, if trained separately.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Test batch size.")
    parser.add_argument("--snapshots-per-sample", type=int, default=25, help="Number of snapshots per sample.")

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False


def train(args, device=None):
    if args.dataset == 'simpleflow':
        in_channels = 1 
        out_channels = 1
        state_size = 4
        enc_mid_channels = args.enc_mid_channels 
        dec_mid_channels = args.dec_mid_channels 
        datasets = {}
        for key in ["train", "val"]:
            datasets[key] = SimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets["test"] = EvalSimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)

        train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size,
                                shuffle=True, num_workers=4)

        val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size, 
                shuffle=False, num_workers=4)

        test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size, 
                shuffle=False, num_workers=4)
    elif args.dataset == 'shallowwater':
        in_channels = 1 
        out_channels = 1
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_size = 4
        datasets = {}
        for key in ["train", "val"]:
            datasets[key] = ShalloWaterDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets["test"] = EvalShallowWaterDataset(snapshots_per_sample=args.snapshots_per_sample)

        train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size, 
                                shuffle=True, num_workers=4)

        val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size,
                shuffle=False, num_workers=4)

        test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size,
                shuffle=False, num_workers=4)
    elif args.dataset == 'reacdiff':
        in_channels = 2 
        out_channels = 2
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_size = 4
        datasets = {}
        for key in ["train", "val"]:
            datasets[key] = ReacDiffDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets["test"] = EvalReacDiffDataset(snapshots_per_sample=args.snapshots_per_sample)

        train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size, 
                                shuffle=True, num_workers=4)

        val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size,
                shuffle=False, num_workers=4)

        test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size,
                shuffle=False, num_workers=4)
    elif args.dataset == 'navierstokes':
        in_channels = 2
        out_channels = 2
        enc_mid_channels = 128
        dec_mid_channels = 256
        state_size = 8
        datasets = {}
        for key in ["train", "val"]:
            datasets[key] = NavierStokesDataset(snapshots_per_sample=args.snapshots_per_sample)
        datasets["test"] = EvalNavierStokesDataset(snapshots_per_sample=args.snapshots_per_sample)
        
        train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size, 
                                shuffle=True, num_workers=4)

        val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size,
                shuffle=False, num_workers=4)

        test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size,
                shuffle=False, num_workers=4)   
    else:
        raise ValueError('Invalid dataset option!')


    # Setup losses
    ae_mse_loss_fun = nn.MSELoss()  

    ae_epochs = args.ae_epochs 
    if args.ae_option == "ae":
        ae_model = Model_AE(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels)
        ae_model = ae_model.to(device)
    else:
        raise ValueError('Invalid ae option!')

    total_params_ae = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    print('# trainable parameters: ', total_params_ae)

    ae_optimizer = torch.optim.AdamW(
            ae_model.parameters(),
            lr=args.ae_learning_rate,
            weight_decay=args.ae_weight_decay,
            betas=(0.9, 0.999))

    if args.ae_lr_scheduler == "exponential":
        ae_lr_scheduler = lr_scheduler.ExponentialLR(ae_optimizer, gamma=0.999)
    elif args.ae_lr_scheduler == "cosine":
        ae_lr_scheduler = get_cosine_schedule_with_warmup(ae_optimizer, 
        num_warmup_steps=len(train_loader) * 5, # we need only a very shot warmup phase for our data
        num_training_steps=(len(train_loader) * ae_epochs))

    for epoch in range(ae_epochs):
        ae_model.train()
        train_gen = tqdm(train_loader, desc="Training")
        for batch in train_gen:
            observations = batch.to(device)
            batch_size = observations.size(0)

            if args.ae_option == "ae":
                input_snapshots, reconstructed_snapshots = ae_model(observations)
                ae_loss = ae_mse_loss_fun(input_snapshots, reconstructed_snapshots)

            else:
                raise ValueError('Invalid ae option!')

            # Backward pass
            ae_model.zero_grad()
            ae_loss.backward()
        
            # Optimizer step
            ae_optimizer.step()
        
        # update learning rate
        ae_lr_scheduler.step()
        
        print(f"epoch={epoch}, ae_loss={ae_loss}")

    print('Done with training AE...')

    # Specify a path
    if args.enc_mid_channels == 64 and args.dec_mid_channels == 128:
        PATH = "checkpoints/" + args.dataset + ".pt"
    else:
        PATH = "checkpoints/" + args.dataset + "_enc" + str(args.enc_mid_channels) + "_dec" + str(args.dec_mid_channels) + ".pt"

    # Save the model
    torch.save(ae_model, PATH)
    print('trained AE model is saved...')

if __name__ == "__main__":
    args = parse_args()

    # Launch processes.
    print('Launching processes...')

    set_seed(args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(args, device=device)
    