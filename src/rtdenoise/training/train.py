from ..models.base_denoiser import *

from . import prebatched_dataset
from . import loss_functions

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import openexr_numpy as exr
import time

def proc_img(r : torch.Tensor, a : torch.Tensor):
    # if something is zero just replace it with 1
    a = torch.where(a == 0, torch.ones_like(a), a)

    # now, how do we resolve very dark images?
    # let's try clamping
    # 0.01 is going to be a 100x multiplier
    # let's try 0.01
    a = torch.clamp(a, min=1.0/100.0)

    # relu to make sure we don't have any negative values for any reason
    r = torch.where(torch.isnan(r), 0.0, r).relu()

    r = r / a

    # gamma transform
    r = torch.pow(r, 1.0 / 2.2)

    return r, a

def reformat_inputs(data : tuple[torch.Tensor, torch.Tensor], device):
    # ([N], L, C, H, W)
    inputs, reference = data

    # create batch dimension if it doesn't exist already
    if inputs.dim() < 5:
        inputs = inputs.unsqueeze(0)
        reference = reference.unsqueeze(0)

    inputs = inputs.to(device)
    reference = reference.to(device)

    # apply gamma correction on the reference
    # we expect the denoiser to output gamma corrected values by itself
    with torch.no_grad():
        reference = torch.where(
            torch.isnan(reference), 0.0, reference
        ).relu().pow(1.0 / 2.2)
        
        mcol, malb = proc_img(r=inputs[:, :, 0:3], a=inputs[:, :, 3:6])

        inputs[:, :, 0:3] = mcol
        inputs[:, :, 3:6] = malb

        if True:
            inputs = inputs[:, :1]
            reference = reference[:, :1]

    print(f"INP shape {inputs.shape}")
    print(f"REF shape {reference.shape}")

    return inputs, reference

def dump_test_sequence(folder : str, name, seq_out : torch.Tensor, seq_ref : torch.Tensor, loss_fn):
    os.makedirs(folder, exist_ok=True)

    if seq_ref is not None:
        loss = loss_fn(seq_out, seq_ref)
        print(f"\tTotal loss for model {name}\twas {loss.item()}")

        f = open(os.path.join(folder, f"loss-{name}.csv"), "w")
        f.write("frameidx, loss\n")
        f.write(f"-1, {loss.item()}\n")
    else:
        f = None

    # (N, L, 3, H, W)

    for j in range(0, seq_out.shape[1]):
        still_frame = seq_out[0, j, :, :, :]

        if f is not None:
            still_loss = loss_fn(still_frame, seq_ref[0, j, :, :, :])

            f.write(f"{j}, {still_loss.item()}\n")

        exr.imwrite(
            os.path.join(folder, f"{name}{j}.exr"),
            still_frame.detach().permute(1, 2, 0).cpu().numpy()
        )

    if f is not None:
        f.close()

class ModelTimer():
    def __init__(self, name):
        self.name = name
        self.start = time.time()

    def stop_and_str(self):
        delta = time.time() - self.start

        return f"\tmodel timer {self.name}:\t{delta}"


def run_epoch(
        dataset : prebatched_dataset.PrebatchedDataset,
        dataloader : torch.utils.data.DataLoader, 
        models : list[BaseDenoiser], 
        optimizers : list[torch.optim.Optimizer], 
        schedulers, 
        names : list[str],
        loss_fn : nn.Module,
        scaler : torch.GradScaler,
        device,
    ):

    num_models = len(models)
    accum_training_loss = [0.0] * num_models
    num_training_samples = 0


    for model in models:
        model.train()
    

    dataset.switch_mode(training=True, fullres=False)
    for batch_idx, data in enumerate(dataloader):
        print(f"\tProcessing training batch {batch_idx}")

        seq_in, seq_ref = reformat_inputs(data, device)

        for repeat in range(100):
            print(f"REPEATING SAMPLE {repeat}/100 times!")

            for i in range(num_models):
                mt = ModelTimer(names[i])

                optimizers[i].zero_grad()

                with torch.autocast(device_type=device, dtype=torch.float16):
                    seq_out = models[i](seq_in)
                    loss = loss_fn(seq_out, seq_ref)
                    scaler.scale(loss).backward()
                    scaler.step(optimizers[i])

                scaler.update()

                print(f"\t\tLoss for model {names[i]}\twas {loss.item()}\t{mt.stop_and_str()}")

                accum_training_loss[i] += loss.item() * seq_in.shape[0]
            
            num_training_samples += seq_in.shape[0]



    for scheduler in schedulers:
        scheduler.step()



    training_loss = [loss / num_training_samples for loss in accum_training_loss]

    return training_loss

def run_eval(
        dataset : prebatched_dataset.PrebatchedDataset,
        dataloader : torch.utils.data.DataLoader, 
        models : list[BaseDenoiser], 
        loss_fn : nn.Module,
        scaler : torch.GradScaler,
        device,
    ):

    num_models = len(models)
    accum_test_loss = [0.0] * num_models
    num_eval_samples = 0

    for model in models:
        model.eval()

    dataset.switch_mode(training=False, fullres=False)
    for batch_idx, data in enumerate(dataloader):
        print(f"\tProcessing eval batch {batch_idx}")

        seq_in, seq_ref = reformat_inputs(data, device)

        for i in range(num_models):

            with torch.autocast(device_type=device, dtype=torch.float16):
                seq_out = models[i](seq_in)
                loss = loss_fn(seq_out, seq_ref)
        
            # Update statistics
            accum_test_loss[i] += loss.item() *  seq_ref.shape[0]
            num_eval_samples += seq_ref.shape[0]



    test_loss = [loss / num_eval_samples for loss in accum_test_loss]
    return test_loss

def run_test(
        dataset : prebatched_dataset.PrebatchedDataset,
        dataloader : torch.utils.data.DataLoader, 
        models : list[BaseDenoiser], 
        names : list[str],
        loss_fn : nn.Module,
        scaler : torch.GradScaler,
        dump_prefix,
        device,
    ):
    num_models = len(models)

    dataset.switch_mode(training=False, fullres=True)
    for seq_idx, data in enumerate(dataloader):
        print(f"Processing test sequence {seq_idx}")

        seq_in, seq_ref = reformat_inputs(data, device)

        folder = os.path.join(dump_prefix, f"seq{seq_idx}")
        for i in range(num_models):
            with torch.autocast(device_type=device, dtype=torch.float16):
                seq_out = models[i](seq_in)

            dump_test_sequence(
                folder=folder, 
                name=names[i], 
                seq_out=seq_out, 
                seq_ref=seq_ref, 
                loss_fn=loss_fn
            )

        # dump the reference... for reference
        # a bit heavy on the disk usage side but whatever
        dump_test_sequence(
            folder=folder, 
            name="Reference", 
            seq_out=seq_ref, 
            seq_ref=None, 
            loss_fn=None
        )
            

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

    scaler = torch.GradScaler()

    loss_fn = F.l1_loss
    losses = []

    for epoch in range(num_epochs):
        print(f"Processing epoch {epoch}")

        # Training code
        training_loss = run_epoch(
            dataset=dataset,
            dataloader=dataloader,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            names=names,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device
        )

        with torch.no_grad():

            test_loss = run_eval(
                dataset=dataset,
                dataloader=dataloader,
                models=models,
                loss_fn=loss_fn,
                scaler=scaler,
                device=device
            )
            

            print(f"Training and epoch loss in epoch {epoch}:")
            for i in range(num_models):
                print(f"\t{names[i]}\t{training_loss[i]}\t{test_loss[i]}")

            losses.append((training_loss, test_loss))

            print("Full loss history:")
            for loss in losses:
                training_loss, eval_loss = loss

                console_row = f"\tEpoch {epoch}:\t"

                for j in range(len(models)):
                    console_row = f"{console_row},\t{training_loss[j]} ({names[j]})"

                for j in range(len(models)):
                    console_row = f"{console_row},\t{eval_loss[j]} ({names[j]})"

                print(console_row)

            run_test(
                dataset=dataset,
                dataloader=dataloader,
                models=models,
                names=names,
                loss_fn=loss_fn,
                scaler=scaler,
                dump_prefix=os.path.join(os.environ['RTDENOISE_OUTPUT_PATH'], f"test/epoch{epoch}"),
                device=device
            )

    return (models, losses)
            


        



            
