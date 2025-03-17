import torch

import os
import shutil

from . import frame_dataset

class PrebatchedDataset(torch.utils.data.Dataset):
    """A prebatched dataset."""
    def __init__(self, folder, buffers, truncated_batch_size=None):
        super(PrebatchedDataset, self).__init__()

        with open(os.path.join(folder, "format.txt"), "r") as f:
            dataset_format = f.readlines()
            dataset_format = [named_buffer.strip() for named_buffer in dataset_format if named_buffer.strip() != None]

            if len(dataset_format) != len(buffers) or any([dataset_format[i] != buffers[i] for i in range(len(dataset_format))]):
                raise RuntimeError(f"Dataset has {dataset_format} buffers but {buffers} were requested.")

        self.switch_mode(training=True, fullres=False)

        self.rt_train = self.catalouge_samples(folder, "rt_train")
        self.rt_test = self.catalouge_samples(folder, "rt_test")
        self.test_fullres_dir = self.catalouge_samples(folder, "test_fullres_dir")

        self.truncated_batch_size = truncated_batch_size

    def switch_mode(self, training, fullres):
        if training and fullres:
            raise RuntimeError("RT Denoise datasets cannot be used in training with full-resolution datasets!")

        self.training = training
        self.fullres = fullres

    def get_samples(self):
        if self.training:
            return self.rt_train
        else:
            if self.fullres:
                return self.test_fullres_dir
            else:
                return self.rt_test

    def catalouge_samples(self, folder, subdir):
        full_path = os.path.join(folder, subdir)
        samples = [os.path.join(full_path, sample) for sample in os.listdir(full_path)]
        return samples

    def fast_load_tensors(self, path):
        download_path = os.path.join(os.environ['RTDENOISE_DOWNLOAD_CACHE'], frame_dataset.get_path_sha256(path))

        shutil.copyfile(path, download_path)
        tensors = torch.load(download_path, weights_only=True)
        os.remove(download_path)

        seq_in = tensors["seq_in"]
        seq_ref = tensors["seq_ref"]

        if (self.truncated_batch_size is not None) and (not self.fullres):
            seq_in = seq_in[:self.truncated_batch_size]
            seq_ref = seq_ref[:self.truncated_batch_size]

        return (seq_in, seq_ref)

    def __getitem__(self, index):
        return self.fast_load_tensors(self.get_samples()[index])

    def __len__(self):
        return len(self.get_samples())