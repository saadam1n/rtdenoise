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

    return inputs, reference

if __name__ == "__main__":
    # this package is meant to be GPU-only, unless you are a crazy person
    if torch.cuda.is_available():
        device = "cuda"

        print(f"Utilizing GPU {torch.cuda.get_device_name(0)} for training and inference.")
    else:
        raise RuntimeError("Unable to find suitable GPU for training!")

    training_dataloader = DataLoader(
        rtdenoise.FrameDataset(dataset_folder="/home/saada/Datasets/mini_local_dataset/rt_train", device=device, seq_len=8),
        batch_size=1, shuffle=True
    )

    eval_dataloader = DataLoader(
        rtdenoise.FrameDataset(dataset_folder="/home/saada/Datasets/mini_local_dataset/rt_test", device=device, seq_len=8),
        batch_size=32, shuffle=False
    )

    test_dataloader = DataLoader(
        rtdenoise.FrameDataset(dataset_folder="/home/saada/Datasets/mini_local_dataset/test_fullres_dir", device=device, seq_len=8),
        batch_size=1, shuffle=False
    )

    model = rtdenoise.LaplacianPyramidUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    model, losses = rtdenoise.train_model(training_dataloader, eval_dataloader, model=model, optimizer=optimizer, scheduler=scheduler, num_epochs=1)

    print("Losses over time:")
    f = open("/tmp/latest-losses.csv", "w")
    f.write("Epoch, Loss\n")
    for i, loss in enumerate(losses):
        print(f"\tEpoch {i}:\t{loss}")
        f.write(f"{i}, {loss}\n")

    # show results on full image
    with torch.no_grad():
        loss_fn = torch.nn.L1Loss()

        model.eval()

        for seq_idx, data in enumerate(test_dataloader):
            print(f"Processing test sequence {seq_idx}")

            seq_in, seq_ref = unsqueeze_inputs(data)
            seq_out = model(seq_in)

            loss = loss_fn(seq_out, seq_ref)
            
            print(f"\tTotal loss on test sequence {seq_idx} was {loss.item()}\n\n")

            dump_path = f"/tmp/rtdenoise/test/seq{seq_idx}/"
            os.makedirs(dump_path, exist_ok=True)

            with open(dump_path + "loss.csv", "w") as f:
                f.write("frameidx, loss\n")
                f.write(f"-1, {loss.item()}\n")

                for i in range(0, seq_out.shape[1], 3):
                    still_frame = seq_out[0, i:i+3, :, :]
                    still_loss = loss_fn(still_frame, seq_ref[0, i:i+3, :, :])

                    f.write(f"{i // 3}, {still_loss.item()}\n")

                    exr.imwrite(dump_path + f"frame{i // 3}.exr", still_frame.permute(1, 2, 0).cpu().numpy())

            


    while True:
        time.sleep(1.0)
    