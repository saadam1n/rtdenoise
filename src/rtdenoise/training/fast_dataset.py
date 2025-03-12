"""
A faster alternative to the regular dataset loading. 
"""

import torch
import torch.multiprocessing as mp

import openexr_numpy as exr

import hashlib
import os
import random
import shutil
import tarfile

from . import frame_dataset

def load_exr_as_tensor(path):
    return torch.from_numpy(exr.imread(path)).permute(2, 0, 1)

def load_buffers(path, seq_len, buffers):
    return torch.cat(
        [
            load_exr_as_tensor(f"{path}/{bufname}{i}.exr")
            for i in range(seq_len)
            for bufname in buffers 
        ],
        dim=0 # not 1 because we are not batching yet
    )

def cachify_entry(args):
    seq_len, path = args

    print(f"\t\tBuilding cache entry for {path}")

    # ideally RTDENOISE_DOWNLOAD_CACHE should point to a ramdisk
    download_path = f"{os.environ['RTDENOISE_DOWNLOAD_CACHE']}/{frame_dataset.get_path_sha256(path)}/"
    tarball_path = download_path + "download.tgz"

    os.makedirs(download_path)
    shutil.copyfile(path, tarball_path)

    with tarfile.open(tarball_path, "r:gz") as t:
        t.extractall(path=download_path)

    seq_in = load_buffers(download_path, seq_len, ["color", "albedo", "normal", "motionvec"])
    seq_ref = load_buffers(download_path, seq_len, ["reference"])

    shutil.rmtree(download_path)

    pt_cache_path = frame_dataset.get_cache_path(path)

    torch.save(seq_in, pt_cache_path + "-seq-in.pt")
    torch.save(seq_ref, pt_cache_path + "-seq-ref.pt")

def copy_and_load_tensor(path):
    download_path = f"{os.environ['RTDENOISE_DOWNLOAD_CACHE']}/{frame_dataset.get_path_sha256(path)}.pt"
    
    shutil.copyfile(path, download_path)
    tensor = torch.load(path, weights_only=True)
    os.remove(download_path)

    return tensor


"""
This assumes that a cache has already been built.
"""
def copy_and_load(args):
    sample_queue, path = args

    pt_cache_path = frame_dataset.get_cache_path(path)
    
    seq_in = copy_and_load_tensor(pt_cache_path + "-seq-in.pt")
    seq_ref = copy_and_load_tensor(pt_cache_path + "-seq-ref.pt")

    sample_queue.put((seq_in, seq_ref))

"""
Take everyhing that has been loaded so far and aggregate it
"""
def aggregate_samples(args):
    batch_queue, sample_queue, batch_size = args 

    batch_in = []
    batch_ref = []

    while len(batch_in) < batch_size:
        seq_in, seq_ref = sample_queue.get()

        batch_in.append(seq_in)
        batch_ref.append(seq_ref)

    batch_queue.put((torch.stack(batch_in), torch.stack(batch_ref)))
        



class FastDataset:
    """
    Registers all files in path
    """
    def __init__(self, path, seq_len, batch_size, num_processes, prefetch_factor):
        self.path = path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor

        self.pool = mp.Pool(processes=num_processes)
        self.manager = mp.Manager()
        self.batch_queue = self.manager.Queue()

        if self.prefetch_factor < 1:
            raise ValueError("Cannot have a prefetch factor less than 1")

        files = os.listdir(self.path)

        num_files = len(files)
        if True: #exclude_incomplete_batches
            num_files = (num_files // self.batch_size) * self.batch_size

        self.samples = []
        for i in range(num_files):
            self.samples.append(os.path.join(self.path, files[i]))

        print(f"Dataset at {self.path}\thas {len(self.samples)} items.")
        self.build_cache()



    """
    Generate batch ordering
    """
    def __iter__(self):
        shuffled_samples = [i for i in range(len(self.samples))] 
        random.shuffle(shuffled_samples)

        # round up division
        num_batches = (len(self.samples) - 1) // self.batch_size + 1

        self.batches = []
        for i in range(num_batches):
            first_sample = i * self.batch_size
            last_sample = min(first_sample + self.batch_size, len(self.samples))

            current_batch = []
            for j in range(first_sample, last_sample):
                current_batch.append(shuffled_samples[j])

            self.batches.append(current_batch)
            
        self.next_batch = 0

        for i in range(self.prefetch_factor):
            if i < len(self.batches):
                self.enqueue_batch_for_load(i)

        return self 
    
    def __next__(self):
        if self.next_batch >= len(self.batches):
            raise StopIteration
        
        batch_tensor = self.fetch_next_available_batch()

        self.next_batch += 1
        if self.next_batch + self.prefetch_factor < len(self.batches):
            self.enqueue_batch_for_load(self.next_batch + self.prefetch_factor)

        return batch_tensor

    def enqueue_batch_for_load(self, batch_index):
        """
        We keep a seperate queue for each loading job
        """

        batch_to_load = self.batches[batch_index]
        sample_queue = self.manager.Queue()

        load_args = [
            (
                sample_queue,
                self.samples[i]
            ) 
        for i in batch_to_load]

        self.pool.map(copy_and_load, load_args)
        self.pool.map(aggregate_samples, [(self.batch_queue, sample_queue, self.batch_size)])

    def fetch_next_available_batch(self):
        return self.batch_queue.get()
    
    def build_cache(self):
        os.makedirs(frame_dataset.get_cache_prefix(), exist_ok=True)

        pt_cache_files = set(os.listdir(frame_dataset.get_cache_prefix()))

        uncached_files = [(self.seq_len, sample) for sample in self.samples if (frame_dataset.get_path_sha256(sample) + "-seq-in.pt") not in pt_cache_files]

        print(f"\tFound {len(uncached_files)}\tuncached entries in this dataset. Building cache for uncached entries...")

        self.pool.map(cachify_entry, uncached_files)
