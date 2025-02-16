import rtdenoise

import torch
from torch.utils.data import DataLoader

import cv2
import time

if __name__ == "__main__":
    # this package is meant to be GPU-only, unless you are a crazy person
    if torch.cuda.is_available():
        device = "cuda"

        print(f"Utilizing GPU {torch.cuda.get_device_name(0)} for training and inference.")
    else:
        raise RuntimeError("Unable to find suitable GPU for training!")

    dataset = rtdenoise.FrameDataset("data/", "Dataset0", device, 1)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = rtdenoise.LaplacianPyramidUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    model, losses = rtdenoise.train_model(dataloader, model=model, optimizer=optimizer, scheduler=scheduler, num_epochs=1)

    print("Losses over time:")
    f = open("results/latest-losses.csv", "w")
    f.write("Epoch, Loss\n")
    for i, loss in enumerate(losses):
        print(f"\tEpoch {i}:\t{loss}")
        f.write(f"{i}, {loss}\n")

    # show results on full image
    model.eval()
    loss_fn = torch.nn.L1Loss()
    with torch.no_grad():
        input, target = dataset.get_full_img()
        input = input[None, :]
        target = target[None, :]

        model = model.to(device)
        output = model(input)

        loss = loss_fn(output, target)
        print(f"Loss on entire image was {loss.item()}")

        image = output.detach()
        image = image[0].squeeze().permute((1, 2, 0)).cpu().numpy()
        image = image[:, :, -3:]

        cv2.imwrite("results/output.exr", image)
        f.write(f"-1, {loss.item()}")

    f.close()

    while True:
        time.sleep(1.0)
    