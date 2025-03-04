import torch
from torch.utils.data import Dataset

import numpy as np
import openexr_numpy as exr

import os
import multiprocessing
import time
import hashlib
import shutil
import tarfile

def get_cache_path(path):
    cache_name = hashlib.sha256(path.encode('utf-8')).hexdigest()
    cache_path = f"{os.environ['RTDENOISE_OUTPUT_PATH']}/cache/" + cache_name
    return cache_path

def load_exr_image(image_path, manager_dict):
    # replace everything with alpha numeric 
    cache_path = get_cache_path(image_path)
    
    per_image_caching = False
    if os.path.exists(cache_path) and per_image_caching:
        image = torch.load(cache_path, weights_only=True)
    else:
        image = torch.tensor(exr.imread(image_path))

        if per_image_caching:
            torch.save(image, cache_path)

    filename = os.path.basename(image_path)
    manager_dict[filename] = image

class FrameDataset(Dataset):
    def __init__(self, dataset_folder, device, seq_len):
        self.dataset_dir = dataset_folder + "/"
        self.seq_len=seq_len

        all_tarballs = os.listdir(dataset_folder)
        self.samples = [self.dataset_dir + tarball for tarball in all_tarballs]
        
        print(f"Dataset at {self.dataset_dir} has {len(self.samples)} samples")

        self.device=device


    def __len__(self):
        # length here is defined by the number of frame sequneces we have,
        # not the number of frames
        return len(self.samples)

    # this needs to manually convert things to a tensor
    def __getitem__(self, idx):
        # basically, I want to output it in this format for each element of each batch
        # (B, N, C, H, W)
        # B = batch size
        # N = number of frames
        # C = channels for each frame (color, albedo, world norm)

        sample_folder = self.fetch_sample_folder(idx)

        cached_sample = get_cache_path(sample_folder)
        seq_in_cached = cached_sample + "-seq-in.pt"
        seq_ref_cached = cached_sample + "-seq-ref.pt"

        load_start = time.time()

        per_seq_caching = True
        if os.path.exists(seq_ref_cached) and per_seq_caching:
            seq_in = torch.load(seq_in_cached, weights_only=True)
            seq_ref = torch.load(seq_ref_cached, weights_only=True)
        else:
            parallel_load = False
            if parallel_load:
                # load all images in parallel
                with multiprocessing.Manager() as manager:
                    manager_dict = manager.dict()
                    pool = multiprocessing.Pool()

                    all_images_paths = [
                        sample_folder + bufname + str(i) + ".exr" 
                        for i in range(self.seq_len)
                        for bufname in ["color", "albedo", "normal", "motionvec", "reference"]
                    ]

                    with multiprocessing.Pool(processes=len(all_images_paths)) as pool:
                        for image_path in all_images_paths:
                            pool.apply_async(load_exr_image, args=(image_path, manager_dict))
                        pool.close()
                        pool.join()

                    seq_in_list = [
                        manager_dict[bufname + str(i) + ".exr"].to(self.device)
                        for i in range(self.seq_len)
                        for bufname in ["color", "albedo", "normal", "motionvec"]
                    ]

                    seq_ref_list = [
                        manager_dict[bufname + str(i) + ".exr"].to(self.device)
                        for i in range(self.seq_len)
                        for bufname in ["reference"]
                    ]
            else:
                seq_in_list = [
                    torch.tensor(exr.imread(sample_folder + bufname + str(i) + ".exr"))
                    for i in range(self.seq_len)
                    for bufname in ["color", "albedo", "normal", "motionvec"]
                ]

                seq_ref_list = [
                    torch.tensor(exr.imread(sample_folder + bufname + str(i) + ".exr"))
                    for i in range(self.seq_len)
                    for bufname in ["reference"]
                ]

            seq_in = torch.cat(seq_in_list, dim=2).permute(2, 0, 1)
            seq_ref = torch.cat(seq_ref_list, dim=2).permute(2, 0, 1)

            if per_seq_caching:
                torch.save(seq_in, seq_in_cached)
                torch.save(seq_ref, seq_ref_cached)

        
        load_duration = time.time() - load_start

        loading_diagnostics = False
        if loading_diagnostics:
            print(f"\t\tLoading item {idx}\ttook {load_duration}\tseconds")

        self.cleanup_sample_folder(idx)

        return seq_in, seq_ref

    def fetch_sample_folder(self, idx):
        tarball_path = self.samples[idx]
        local_path = self.get_path_local_path(tarball_path)
        os.makedirs(local_path)


        local_file = local_path + ".tgz"
        shutil.copyfile(tarball_path, local_file)
        with tarfile.open(local_file, "r:gz") as tar:
            tar.extractall(path=local_path)
        os.remove(local_file)

        return local_path

    def cleanup_sample_folder(self, idx):
        local_path = self.get_path_local_path(self.samples[idx])
        shutil.rmtree(local_path)

    def get_path_local_path(self, path):
        local_path = f"/tmp/local-{hashlib.sha256(path.encode('utf-8')).hexdigest()}/"
        return local_path



