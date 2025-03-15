from ..models.base_denoiser import *

from . import prebatched_dataset

import torch
from torch.utils.data import DataLoader

import os
import openexr_numpy as exr
import time

def unsqueeze_inputs(data):
    inputs, reference = data

    if inputs.dim() < 4:
        inputs = inputs.unsqueeze(0)
        reference = reference.unsqueeze(0)
    elif inputs.dim() > 4:
        inputs = inputs.flatten(start_dim=0, end_dim=1)
        reference = reference.flatten(start_dim=0, end_dim=1)

    return inputs, reference

class ModelTimer():
    def __init__(self, name):
        self.name = name
        self.start = time.time()

    def stop_and_str(self):
        delta = time.time() - self.start

        return f"\tmodel timer {self.name}:\t{delta}"

def train_model(
        dataset : prebatched_dataset.PrebatchedDataset,
        dataloader : torch.utils.data.DataLoader, 
        models : list[BaseDenoiser], 
        optimizers : list[torch.optim.Optimizer], 
        schedulers, 
        names : list[str],
        num_epochs, 
        device
    ):
    num_models = len(models)

    loss_fn = torch.nn.L1Loss()
    losses = []

    for epoch in range(num_epochs):
        print(f"Processing epoch {epoch}")

        # Training code
        accum_training_loss = [0.0] * num_models
        num_training_samples = 0


        dataset.switch_mode(training=True, fullres=False)

        for model in models:
            model.train()
        
        for batch_idx, data in enumerate(dataloader):
            print(f"\tProcessing training batch {batch_idx}")

            seq_in, seq_ref = unsqueeze_inputs(data)
            seq_in = seq_in.to(device)
            seq_ref = seq_ref.to(device)

            with torch.no_grad():
                seq_in.clamp_max_(max=32.0)

            for i in range(num_models):
                mt = ModelTimer(names[i])

                optimizers[i].zero_grad()

                seq_out = models[i](seq_in)

                loss = loss_fn(seq_out, seq_ref)
                loss.backward()

                optimizers[i].step()

                print(f"\t\tLoss for model {names[i]}\twas {loss.item()}\t{mt.stop_and_str()}")

                accum_training_loss[i] += loss.item() * seq_in.shape[0]
                
            num_training_samples += seq_in.shape[0]

        for scheduler in schedulers:
            scheduler.step()

        with torch.no_grad():

            accum_test_loss = [0.0] * num_models
            num_eval_samples = 0

            dataset.switch_mode(training=False, fullres=False)

            for model in models:
                model.eval()

            for batch_idx, data in enumerate(dataloader):
                print(f"\tProcessing eval batch {batch_idx}")

                seq_in, seq_ref = unsqueeze_inputs(data)
                seq_in = seq_in.to(device)
                seq_ref = seq_ref.to(device)

                with torch.no_grad():
                    seq_in.clamp_max_(max=32.0)

                for i in range(num_models):
                    seq_out = models[i](seq_in)

                    loss = loss_fn(seq_out, seq_ref)
                
                    # Update statistics
                    accum_test_loss[i] += loss.item() *  seq_ref.shape[0]
                    num_eval_samples += seq_ref.shape[0]

            training_loss = [loss / num_training_samples for loss in accum_training_loss]
            test_loss = [loss / num_training_samples for loss in accum_test_loss]

            print(f"Training and epoch loss in epoch {epoch}:")
            for i in range(num_models):
                print(f"\t{names[i]}\t{training_loss[i]}\t{test_loss[i]}")

            losses.append((training_loss, test_loss))

            dataset.switch_mode(training=False, fullres=True)

            for seq_idx, data in enumerate(dataloader):
                print(f"Processing test sequence {seq_idx}")

                seq_in, seq_ref = unsqueeze_inputs(data)

                seq_in = seq_in.to(device)
                seq_ref = seq_ref.to(device)

                with torch.no_grad():
                    seq_in.clamp_max_(max=32.0)

                for i in range(num_models):
                    seq_out = models[i](seq_in)

                    loss = loss_fn(seq_out, seq_ref)
                
                    print(f"\tTotal loss on test sequence {seq_idx} was {loss.item()}\n\n")

                    dump_path = f"{os.environ['RTDENOISE_OUTPUT_PATH']}/test/epoch{epoch}/seq{seq_idx}/"
                    os.makedirs(dump_path, exist_ok=True)

                    with open(dump_path + f"loss-{names[i]}.csv", "w") as f:
                        f.write("frameidx, loss\n")
                        f.write(f"-1, {loss.item()}\n")

                        for j in range(0, seq_out.shape[1], 3):
                            still_frame = seq_out[0, j:j+3, :, :]
                            still_loss = loss_fn(still_frame, seq_ref[0, j:j+3, :, :])

                            f.write(f"{j // 3}, {still_loss.item()}\n")

                            exr.imwrite(dump_path + f"{names[i]}{j // 3}.exr", still_frame.permute(1, 2, 0).cpu().numpy())

    return (model, losses)
            


        



            
