import openexr_numpy as exr
import os

exr.set_default_channel_names(2, ["r", "g"])
os.makedirs(f"{os.environ['RTDENOISE_OUTPUT_PATH']}/cache", exist_ok=True)
os.makedirs(f"{os.environ['RTDENOISE_OUTPUT_PATH']}/test", exist_ok=True)
