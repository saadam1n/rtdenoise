from ..models.base_denoiser import *
from .frame_dataset import *

import torch

def train_model(training_dataset : FrameDataSet, model: BaseDenoiser, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler, num_epochs):
    # placeholder
    loss_fn = torch.nn.L1Loss()
    losses = []

    for epoch in range(num_epochs):
        # Training code
        model.train()

        total_loss = 0
        num_batches = 0
        for batch_idx, (inputs, reference) in enumerate(training_dataset):
            optimizer.zero_grad()

            output = model(inputs)

            loss = loss_fn(output, reference)
            loss.backward()

            optimizer.step()
            scheduler.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches
        losses.append(epoch_loss)

    return (model, losses)
            


        



            
