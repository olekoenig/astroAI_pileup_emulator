import glob
import re
import torch
from astropy.io import fits
from torch.utils.data import Dataset, Subset, random_split
import numpy as np

import config

class PileupDataset(Dataset):
    def __init__(self, input_files, target_files, transform=None):
        """
        Args:
            input_files (list): List of file paths for input FITS files.
            target_files (list): List of file paths for target FITS files.
            transform (callable, optional): A function/transform to apply to the input data.
        """
        self.input_files = input_files
        self.target_files = target_files
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = fits.getdata(self.input_files[idx])
        target_data = fits.getdata(self.target_files[idx])

        input_counts = np.array(input_data["COUNTS"], dtype=np.float32)
        target_counts = np.array(target_data["COUNTS"], dtype=np.float32)

        input_tensor = torch.tensor(input_counts)
        target_tensor = torch.tensor(target_counts)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, target_tensor

    def get_filenames(self, idx):
        return self.input_files[idx], self.target_files[idx]


class SubsetWithFilenames(Subset):
    def get_filenames(self, idx):
        """Retrieve filenames using the parent dataset."""
        return self.dataset.get_filenames(self.indices[idx])


def get_src_flux_from_filename(fname):
    src_flux = re.findall("[0-9]+.[0-9]+Em10", fname)[0]
    return float(src_flux.split("Em10")[0]) * 1e-10

def load_and_split_dataset():
    piledup = glob.glob(config.SPECDIR + "*cgs*.fits")
    # piledup = [fname for fname in piledup if get_src_flux_from_filename(fname) <= 1e-10]
    nonpiledup = [pha.replace("piledup", "nonpiledup") for pha in piledup]

    torch.manual_seed(config.DATALOADER_RANDOM_SEED)

    dataset = PileupDataset(piledup, nonpiledup)

    if len(dataset[0][0]) != 1024:
        exit("Input dimension must be 1024")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size  # Ensure all samples are used

    train_dataset, val_dataset, test_dataset = [
    SubsetWithFilenames(dataset, split.indices) for split in random_split(dataset,[train_size, val_size, test_size])
    ]

    #train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset