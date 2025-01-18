import rtdenoise

import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"

        print(f"Utilizing GPU {torch.cuda.get_device_name(0)} for training and inference.")
    else:
        raise RuntimeError("Unable to find suitable GPU for training!")
