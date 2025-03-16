import rtdenoise

import torch
from torch.utils.data import DataLoader

import time
import openexr_numpy as exr

import os

# comment to force recration of docker image
# please force recreation
if __name__ == "__main__":
    # this package is meant to be GPU-only, unless you are a crazy person
    if torch.cuda.is_available():
        device = "cuda"

        print(f"Utilizing GPU {torch.cuda.get_device_name(0)} for training and inference.")
    else:
        raise RuntimeError("Unable to find suitable GPU for training!")

    dataset = rtdenoise.PrebatchedDataset(os.environ['RTDENOISE_DATASET_PATH'], buffers=["color", "albedo", "normal", "motionvec"], truncated_batch_size=10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, prefetch_factor=4)

    models = [
        rtdenoise.SingleFrameDiffusion(),
        rtdenoise.LaplacianDenoiser()
    ]

    names = [
        type(model).__name__ for model in models
    ]

    parallel_models = [
        torch.nn.DataParallel(model.to(device)) for model in models
    ]

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.005) for model in parallel_models
    ]

    schedulers = [
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.85) for optimizer in optimizers
    ]

    model, losses = rtdenoise.train_model(dataset, dataloader, models=parallel_models, optimizers=optimizers, schedulers=schedulers, names=names, num_epochs=32, device=device)

    print("Losses over time:")
    with open(f"{os.environ['RTDENOISE_OUTPUT_PATH']}/latest-losses.csv", "w") as f:
        header = "\tEpoch Index"
        for model in models:
            header = f"{header}, {type(model).__name__} training loss"
        for model in models:
            header = f"{header}, {type(model).__name__} eval loss"
        
        print(header)
        f.write(f"{header}\n")
    
        for i, loss in enumerate(losses):
            csv_row = f"{i}"
            console_row = f"\tEpoch {i}:\t"

            training_loss, eval_loss = loss

            for j in range(len(models)):
                csv_row = f"{csv_row},\t{training_loss[j]}"
                console_row = f"{console_row},\t{training_loss[j]} ({type(models[j]).__name__})"

            for j in range(len(models)):
                csv_row = f"{csv_row},\t{eval_loss[j]}"
                console_row = f"{console_row},\t{eval_loss[j]} ({type(models[j]).__name__})"
            
            print(console_row)
            f.write(f"{csv_row}\n")
            

    print("Done!")
    