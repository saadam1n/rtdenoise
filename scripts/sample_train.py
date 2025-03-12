import rtdenoise

import torch
from torch.utils.data import DataLoader

import time
import openexr_numpy as exr

import os

def unsqueeze_inputs(data):
    inputs, reference = data

    if inputs.dim() < 4:
        inputs = inputs.unsqueeze(0)
        reference = reference.unsqueeze(0)
    elif inputs.dim() > 4:
        inputs = inputs.flatten(start_dim=0, end_dim=1)
        reference = reference.flatten(start_dim=0, end_dim=1)

    return inputs, reference

# comment to force recration of docker image
# please force recreation
if __name__ == "__main__":
    # this package is meant to be GPU-only, unless you are a crazy person
    if torch.cuda.is_available():
        device = "cuda"

        print(f"Utilizing GPU {torch.cuda.get_device_name(0)} for training and inference.")
    else:
        raise RuntimeError("Unable to find suitable GPU for training!")

    dataset = rtdenoise.PrebatchedDataset(os.environ['RTDENOISE_DATASET_PATH'], ["color", "albedo", "normal", "motionvec"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16, prefetch_factor=2)

    model = torch.nn.DataParallel(rtdenoise.LaplacianPyramidUNet().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

    model, losses = rtdenoise.train_model(dataset, dataloader, model=model, optimizer=optimizer, scheduler=scheduler, num_epochs=2, device=device)

    print("Losses over time:")
    f = open(f"{os.environ['RTDENOISE_OUTPUT_PATH']}/latest-losses.csv", "w")
    f.write("Epoch, Training Loss, Eval Loss\n")
    for i, loss in enumerate(losses):
        train, eval = loss

        print(f"\tEpoch {i}:\t{train}\t{eval}")
        f.write(f"{i}, {train}, {eval}\n")

    # show results on full image
    with torch.no_grad():
        loss_fn = torch.nn.L1Loss()

        dataset.switch_mode(training=False, fullres=True)

        model.eval()

        for seq_idx, data in enumerate(dataloader):
            print(f"Processing test sequence {seq_idx}")

            seq_in, seq_ref = unsqueeze_inputs(data)

            seq_in = seq_in.to(device)
            seq_ref = seq_ref.to(device)

            seq_out = model(seq_in)

            loss = loss_fn(seq_out, seq_ref)
            
            print(f"\tTotal loss on test sequence {seq_idx} was {loss.item()}\n\n")

            dump_path = f"{os.environ['RTDENOISE_OUTPUT_PATH']}/test/seq{seq_idx}/"
            os.makedirs(dump_path, exist_ok=True)

            with open(dump_path + "loss.csv", "w") as f:
                f.write("frameidx, loss\n")
                f.write(f"-1, {loss.item()}\n")

                for i in range(0, seq_out.shape[1], 3):
                    still_frame = seq_out[0, i:i+3, :, :]
                    still_loss = loss_fn(still_frame, seq_ref[0, i:i+3, :, :])

                    f.write(f"{i // 3}, {still_loss.item()}\n")

                    exr.imwrite(dump_path + f"frame{i // 3}.exr", still_frame.permute(1, 2, 0).cpu().numpy())

            

    print("Done!")
    