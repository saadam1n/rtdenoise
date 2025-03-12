from ..models.base_denoiser import *

from . import prebatched_dataset

import torch
from torch.utils.data import DataLoader

def unsqueeze_inputs(data):
    inputs, reference = data

    if inputs.dim() < 4:
        inputs = inputs.unsqueeze(0)
        reference = reference.unsqueeze(0)
    elif inputs.dim() > 4:
        inputs = inputs.flatten(start_dim=0, end_dim=1)
        reference = reference.flatten(start_dim=0, end_dim=1)

    return inputs, reference

def train_model(dataset : prebatched_dataset.PrebatchedDataset, dataloader : torch.utils.data.DataLoader, model: BaseDenoiser, optimizer : torch.optim.Optimizer, scheduler, num_epochs, device):
    # placeholder
    loss_fn = torch.nn.L1Loss()
    losses = []

    for epoch in range(num_epochs):
        print(f"Processing epoch {epoch}")

        # Training code
        total_loss = 0.0
        num_batches = 0

        dataset.switch_mode(training=True, fullres=False)

        model.train()
        for batch_idx, data in enumerate(dataloader):
            print(f"\tProcessing training batch {batch_idx}")

            seq_in, seq_ref = unsqueeze_inputs(data)
            seq_in = seq_in.to(device)
            seq_ref = seq_ref.to(device)

            optimizer.zero_grad()

            seq_out = model(seq_in)

            loss = loss_fn(seq_out, seq_ref)
            loss.backward()

            optimizer.step()

            total_loss += loss.item() *  seq_ref.shape[0]
            num_batches  += seq_ref.shape[0]

        epoch_training_loss = total_loss / num_batches

        scheduler.step()

        with torch.no_grad():

            total_loss = 0.0
            num_batches = 0

            dataset.switch_mode(training=False, fullres=False)

            model.eval()
            for batch_idx, data in enumerate(dataloader):
                print(f"\tProcessing eval batch {batch_idx}")

                seq_in, seq_ref = unsqueeze_inputs(data)
                seq_in = seq_in.to(device)
                seq_ref = seq_ref.to(device)

                seq_out = model(seq_in)

                loss = loss_fn(seq_out, seq_ref)
                
                # Update statistics
                total_loss += loss.item() *  seq_ref.shape[0]
                num_batches += seq_ref.shape[0]

            epoch_loss = total_loss / num_batches
            print(f"Training and epoch loss in epoch {epoch} was {epoch_training_loss}\t{epoch_loss}\n\n")

            losses.append((epoch_training_loss, epoch_loss))

    return (model, losses)
            


        



            
