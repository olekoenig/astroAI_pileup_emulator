import matplotlib.pyplot as plt
import torch
from astropy.io import fits
import numpy as np
import os

from data import load_and_split_dataset
from neuralnetwork import pileupNN
import config

plt.rcParams['text.usetex'] = True

def evaluate_test_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for inputs, targets in test_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)
    return avg_loss

def setup_plot():
    fig, axes = plt.subplots(2, 1, figsize=(10/2.54, 7/2.54), dpi=300,
                             sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    axes[0].set_ylabel('Counts')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(20, 300)

    axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=1)  # Reference line at ratio = 1
    axes[1].set_xlabel('Channel')
    axes[1].set_xscale('log')
    axes[1].set_xlim(20, 300)

    return fig, axes

def plot_ratio(axis, data1, data2, data1_label = "Target", data2_label = "predicted"):
    ratio = torch.where(data2 != 0, data1 / data2, torch.nan)
    axis.plot(ratio, color='black')
    axis.set_ylabel(fr'$\frac{{\mathrm{{{data1_label}}}}}{{\mathrm{{{data2_label}}}}}$')
    axis.set_ylim(0, 2)
    return axis

def plot_testdata(model, test_dataset):
    test_idx = -1
    input_data, target_data = test_dataset[test_idx]
    input_fname, target_fname = test_dataset.get_filenames(test_idx)

    model.eval()
    with torch.no_grad():  # Disable gradient calculation for inference
        predicted_output = model(input_data)

    fix, axes = setup_plot()

    axes[0].set_title(fr"\small {os.path.basename(input_fname)}")
    axes[0].plot(input_data, label="Input")
    axes[0].plot(target_data, label="Target")
    axes[0].plot(predicted_output, label="Predicted")
    axes[0].legend()

    axes[1] = plot_ratio(axes[1], target_data, predicted_output, data1_label="Target")

    plt.tight_layout()
    plt.savefig("testdata.png")

def plot_realdata(predicted_output, real_dataset):

    fix, axes = setup_plot()
    axes[0].plot(real_dataset, label="eROSITA spectrum")
    axes[0].plot(predicted_output, label="Predicted by NN")
    axes[0].legend()
    axes[1] = plot_ratio(axes[1], real_dataset, predicted_output, data1_label="Real", data2_label="Predicted")

    plt.tight_layout()
    plt.show()

def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()

    model = pileupNN(input_size=1024, hidden_size=256, output_size=1024)
    model.load_state_dict(torch.load(config.DATA_NEURAL_NETWORK + "model_weights.pth", map_location="cpu"))

    plot_testdata(model, test_dataset)
    exit(0)
    phafile = "/pool/burg1/novae4ole/V1710Sco_em04_PATall_820_SourceSpec_00001.fits"
    with fits.open(phafile) as hdulist:
        header = hdulist[1].header.copy()
        count_spectrum = hdulist[1].data["COUNTS"]

        real_dataset = torch.tensor(np.array(count_spectrum, dtype=np.float32))

        model.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            predicted_output = model(real_dataset)

        plot_realdata(predicted_output, real_dataset)

        de_piledup_spectrum = predicted_output.numpy().astype(count_spectrum.dtype)

        primary_hdu = fits.PrimaryHDU()
        new_header = fits.Header()
        new_header["BACKFILE"] = "NONE"
        new_header["RESPFILE"] = header["RESPFILE"]
        new_header["ANCRFILE"] = header["ANCRFILE"]

        new_columns = fits.ColDefs([
            fits.Column(name="CHANNEL", format="J", array=hdulist[1].data["CHANNEL"]),
            fits.Column(name="COUNTS", format="J", array=de_piledup_spectrum)
        ])
        new_table_hdu = fits.BinTableHDU.from_columns(new_columns, header=new_header, name="SPECTRUM")

        #new_hdu = fits.BinTableHDU(data=hdulist[1].data.copy(), header=new_header, name="SPECTRUM")
        #new_hdu.data["COUNTS"] = de_piledup_spectrum

        new_hdulist = fits.HDUList([primary_hdu, new_table_hdu])

        new_hdulist.writeto("test.fits", overwrite=True)


if __name__ == "__main__":
    main()