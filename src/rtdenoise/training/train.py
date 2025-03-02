from ..models.base_denoiser import *

import torch
from torch.utils.data import DataLoader

def unsqueeze_inputs(data):
    inputs, reference = data

    if inputs.dim() < 4:
        inputs = inputs.unsqueeze(0)
        reference = reference.unsqueeze(0)

    return inputs, reference

def train_model(training_dataset : DataLoader, eval_dataset : DataLoader, model: BaseDenoiser, optimizer : torch.optim.Optimizer, scheduler, num_epochs):
    # placeholder
    loss_fn = torch.nn.L1Loss()
    losses = []

    for epoch in range(num_epochs):
        print(f"Processing epoch {epoch}")

        # Training code

        model.train()
        for batch_idx, data in enumerate(training_dataset):
            print(f"\tProcessing training batch {batch_idx}")

            seq_in, seq_ref = unsqueeze_inputs(data)

            optimizer.zero_grad()

            seq_out = model(seq_in)

            loss = loss_fn(seq_out, seq_ref)
            loss.backward()

            optimizer.step()

        scheduler.step()

        with torch.no_grad():

            total_loss = 0.0
            num_batches = 0

            model.eval()
            for batch_idx, data in enumerate(eval_dataset):
                print(f"\tProcessing eval batch {batch_idx}")

                seq_in, seq_ref = unsqueeze_inputs(data)
                seq_out = model(seq_in)

                loss = loss_fn(seq_out, seq_ref)
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1

            epoch_loss = total_loss / num_batches
            print(f"Loss in epoch {epoch} was {epoch_loss}\n\n")

            losses.append(epoch_loss)

    return (model, losses)
            


        



            
