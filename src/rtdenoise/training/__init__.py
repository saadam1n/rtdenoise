import openexr_numpy as exr
import os

exr.set_default_channel_names(2, ["r", "g"])
os.makedirs("/tmp/rtdenoise/cache", exist_ok=True)
os.makedirs("/tmp/rtdenoise/test", exist_ok=True)
