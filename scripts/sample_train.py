import rtdenoise

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # this package is meant to be GPU-only, unless you are a crazy person
    if torch.cuda.is_available():
        device = "cuda"

        print(f"Utilizing GPU {torch.cuda.get_device_name(0)} for training and inference.")
    else:
        raise RuntimeError("Unable to find suitable GPU for training!")

    dataset = rtdenoise.FrameDataset("/home/saada/Datasets", "Dataset0", device, 20)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = rtdenoise.FastKPCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    model, losses = rtdenoise.train_model(dataloader, model=model, optimizer=optimizer, scheduler=scheduler, num_epochs=1)


    print("Losses over time:")
    f = open("results/losses.csv", "w")
    f.write("Epoch, Loss\n")
    for i, loss in enumerate(losses):
        print(f"\tEpoch {i}:\t{loss}")
        f.write(f"{i}, {loss}\n")
    f.close()