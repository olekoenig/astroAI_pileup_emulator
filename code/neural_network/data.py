import glob
import re
import torch
from astropy.io import fits
from torch.utils.data import Dataset, Subset, random_split
import numpy as np

from config import SIXTEConfig,MLConfig

sixte_config = SIXTEConfig()
ml_config = MLConfig()

class PileupDataset(Dataset):
    def __init__(self, input_files, target_files, target_transform=None):
        self.input_files = input_files
        self.target_files = target_files
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data_circle0 = fits.getdata(self.input_files[idx])
        input_data_annulus1 = fits.getdata(self.input_files[idx].replace('circle0','annulus1'))
        input_data_annulus2 = fits.getdata(self.input_files[idx].replace('circle0','annulus2'))
        input_data_annulus3 = fits.getdata(self.input_files[idx].replace('circle0','annulus3'))

        # input_channels = torch.tensor(np.array(input_data["CHANNEL"], dtype=np.int32)); torch.stack([input_channels, input_counts], dim=0)

        input_counts_circle0 = np.array(input_data_circle0["COUNTS"], dtype=np.float32)
        input_counts_annulus1 = np.array(input_data_annulus1["COUNTS"], dtype=np.float32)
        input_counts_annulus2 = np.array(input_data_annulus2["COUNTS"], dtype=np.float32)
        input_counts_annulus3 = np.array(input_data_annulus3["COUNTS"], dtype=np.float32)
        counts = [torch.from_numpy(input_counts_circle0),
                  torch.from_numpy(input_counts_annulus1),
                  torch.from_numpy(input_counts_annulus2),
                  torch.from_numpy(input_counts_annulus3)]

        input_tensor = torch.stack(counts, dim=0)

        kt = fits.getval(self.input_files[idx], "KT", ext=1)
        src_flux = fits.getval(self.input_files[idx], "SRC_FLUX", ext=1) / 1e-12
        nh = fits.getval(self.input_files[idx], "NH", ext=1)

        target_tensor = torch.tensor([kt, src_flux, nh])

        return input_tensor, target_tensor

    def alt__getitem__(self, idx):
        input_data = fits.getdata(self.input_files[idx])
        target_data = fits.getdata(self.target_files[idx])

        input_counts = np.array(input_data["COUNTS"], dtype=np.float32)
        target_counts = np.array(target_data["COUNTS"], dtype=np.float32)

        input_tensor = torch.tensor(input_counts)
        target_tensor = torch.tensor(target_counts)

        # if self.target_transform:
        #     input_tensor = self.target_transform(input_tensor)

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
    piledup = glob.glob(sixte_config.SPECDIR + "*cgs*piledup_circle0.fits")

    # Perform filtering on source flux
    # piledup = [fname for fname in piledup if get_src_flux_from_filename(fname) <= 1e-10]

    nonpiledup = [pha.replace("piledup", "nonpiledup") for pha in piledup]

    torch.manual_seed(ml_config.dataloader_random_seed)

    #target_transform = lambda t: torch.clamp(t, min=1e-4)
    dataset = PileupDataset(piledup, nonpiledup, target_transform=None)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size  # Ensure all samples are used

    # Use class inheriting the PileupDataset in order to access the filenames
    # (which contain the spectral parameter and flux information)
    train_dataset, val_dataset, test_dataset = [
    SubsetWithFilenames(dataset, split.indices) for split in random_split(dataset,[train_size, val_size, test_size])
    ]
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def main():
    # Do a few tests for trouble-shooting when running this file individually
    def test1():
        from torch.utils.data import DataLoader
        train_dataset, val_dataset, test_dataset = load_and_split_dataset()
        full_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)  # treat as one big batch to get min/max
        _, targets = next(iter(full_train_loader))
        kt_vals = targets[:, 0]
        src_flux_vals = targets[:, 1]
        nh_vals = targets[:, 2]

        print(f"kT min: {kt_vals.min().item()}, max: {kt_vals.max().item()}")
        print(f"src_flux min: {src_flux_vals.min().item()}, max: {src_flux_vals.max().item()} e-12 cgs")
        print(f"nh min: {nh_vals.min().item()}, max: {nh_vals.max().item()}")

    def test2():
        piledup = glob.glob(sixte_config.SPECDIR + "*cgs*.fits")
        nonpiledup = [pha.replace("piledup", "nonpiledup") for pha in piledup]
        torch.manual_seed(ml_config.dataloader_random_seed)
        dataset = PileupDataset(piledup, nonpiledup)
        print(dataset.__getitem__(0))

    test1()

if __name__ == "__main__":
    main()