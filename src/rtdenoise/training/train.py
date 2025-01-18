from ..models.base_denoiser import *

import torch
from torch.utils.data import DataLoader

def unsqueeze_inputs(data):
    inputs, reference = data

    if inputs.dim() < 4:
        inputs = inputs.unsqueeze(0)
        reference = reference.unsqueeze(0)

    return inputs, reference

def train_model(training_dataset : DataLoader, model: BaseDenoiser, optimizer : torch.optim.Optimizer, scheduler, num_epochs):
    # placeholder
    loss_fn = torch.nn.L1Loss()
    losses = []

    for epoch in range(num_epochs):
        print(f"Processing epoch {epoch}")

        # Training code
        model.train()

        total_loss = 0
        num_batches = 0
        for batch_idx, data in enumerate(training_dataset):
            print(f"\tProcessing training batch {batch_idx}")

            inputs, reference = unsqueeze_inputs(data)

            optimizer.zero_grad()

            output = model(inputs)

            loss = loss_fn(output, reference)
            loss.backward()

            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1


        scheduler.step()

        # Eval code (coming soon)

        epoch_loss = total_loss / num_batches
        print(f"Loss in epoch {epoch} was {epoch_loss}\n\n")

        losses.append(epoch_loss)

    return (model, losses)
            


        



            
