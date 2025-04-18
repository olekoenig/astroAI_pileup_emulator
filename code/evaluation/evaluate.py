import matplotlib.pyplot as plt
import torch
from astropy.io import fits
import numpy as np
import os
import random

from data import load_and_split_dataset
# from neuralnetwork import pileupNN
from nn_parameter_prediction import pileupNN
import config

plt.rcParams['text.usetex'] = True


def setup_plot():
    fig, axes = plt.subplots(2, 1, figsize=(10/2.54, 7/2.54), dpi=300,
                             sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    axes[0].set_ylabel('Counts')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(20, 500)

    axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=1)  # Reference line at ratio = 1
    axes[1].set_xlabel('Channel')
    axes[1].set_xscale('log')
    axes[1].set_xlim(20, 500)

    return fig, axes

def plot_ratio(axis, data1, data2, data1_label = "Target", data2_label = "predicted"):
    ratio = torch.where(data2 != 0, data1 / data2, torch.nan)
    axis.plot(ratio, color='black')
    axis.set_ylabel(fr'$\frac{{\mathrm{{{data1_label}}}}}{{\mathrm{{{data2_label}}}}}$')
    axis.set_ylim(0, 2)
    return axis

def evaluate_on_test_spectrum(model, test_dataset):
    indices = [int(random.uniform(0, len(test_dataset)-1)) for _ in range(20)]
    outfiles = []

    for index in indices:
        input_data, target_data = test_dataset[index]
        input_fname, target_fname = test_dataset.get_filenames(index)

        model.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            predicted_output = model(input_data)

        # Bug in normalization! Hack: rescale for now
        predicted_output *= max(input_data) / max(predicted_output)
        target_data *= max(input_data) / max(target_data)

        ymax = max(predicted_output)

        fix, axes = setup_plot()
        axes[0].set_title(fr"\small {os.path.basename(input_fname)}")
        axes[0].plot(input_data, label="Input", linewidth=1)
        axes[0].plot(predicted_output, label="Predicted (rescaled)", linewidth=1)
        axes[0].plot(target_data, label="Target (rescaled)", linewidth=1)
        axes[0].set_ylim(0.7, ymax + 0.1 * ymax)
        axes[0].legend()
        axes[1] = plot_ratio(axes[1], target_data, predicted_output, data1_label="Target")
        outfile = f"outfiles/testdata_{index}.pdf"
        outfiles.append(outfile)
        plt.tight_layout()
        plt.savefig(outfile)

    outstr = " ".join(outfiles)
    outfile = "testdata.pdf"
    os.system(f"pdfunite {outstr} {outfile}")
    print(f"Wrote {outfile}")
    [os.system(f"rm {fp}") for fp in outfiles]

def write_pha_file(channels, predicted_output, output_fname):
    de_piledup_spectrum = predicted_output.numpy()  # .astype(count_spectrum.dtype)

    primary_hdu = fits.PrimaryHDU()

    header = fits.Header()
    header["BACKFILE"] = "NONE"
    header["RESPFILE"] = config.MASTER_RMF
    header["ANCRFILE"] = config.MASTER_ARF

    columns = fits.ColDefs([
        fits.Column(name="CHANNEL", format="J", array=channels),
        fits.Column(name="COUNTS", format="J", array=de_piledup_spectrum)
    ])
    table_hdu = fits.BinTableHDU.from_columns(columns, header=header, name="SPECTRUM")

    hdulist = fits.HDUList([primary_hdu, table_hdu])
    hdulist.writeto(output_fname, overwrite=True)

def evaluate_on_real_spectrum(model, real_pha_filename, out_pha_file = None):
    with fits.open(real_pha_filename) as hdulist:
        channels = hdulist[1].data["CHANNEL"]
        counts = hdulist[1].data["COUNTS"]

        real_dataset = torch.tensor(np.array(counts, dtype=np.float32))

        model.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            predicted_spectrum = model(real_dataset)

        fix, axes = setup_plot()
        axes[0].plot(real_dataset, label="eROSITA spectrum")
        axes[0].plot(predicted_spectrum, label="Predicted by NN")
        axes[0].legend()
        axes[1] = plot_ratio(axes[1], real_dataset, predicted_spectrum, data1_label="Real", data2_label="Predicted")
        plt.tight_layout()
        plt.show()

        if out_pha_file is not None:
            write_pha_file(channels, predicted_spectrum, out_pha_file)

def evaluate_parameter_prediction(model, test_dataset):
    kt_target = []
    flux_target = []
    kt_predicted = []
    flux_predicted = []

    for index in range(len(test_dataset)):
        input_data, target_data = test_dataset[index]

        model.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            predicted_output = model(input_data)

        kt_predicted.append(predicted_output[0].item())
        flux_predicted.append(predicted_output[1].item())
        kt_target.append(target_data[0].item())
        flux_target.append(target_data[1].item())

    fig, axes = plt.subplots(ncols = 2, figsize=[10/2.54, 5/2.54])

    axes[0].scatter(kt_target, kt_predicted, alpha=0.2, s=2)
    axes[1].scatter(flux_target, flux_predicted, alpha=0.2, s=2)

    axes[0].set_xlabel("True kT [keV]")
    axes[0].set_ylabel("Predicted kT [keV]")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("True flux [e-12 cgs]")
    axes[1].set_ylabel("Predicted flux [e-12 cgs]")

    plt.tight_layout()
    plt.savefig("testdata.pdf")

def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()

    model = pileupNN()
    model.load_state_dict(torch.load(config.DATA_NEURAL_NETWORK + "model_weights.pth", map_location="cpu"))

    # evaluate_on_test_spectrum(model, test_dataset)
    # evaluate_on_real_spectrum(model, "/pool/burg1/novae4ole/V1710Sco_em04_PATall_820_SourceSpec_00001.fits", out_pha_file = "test.fits")
    evaluate_parameter_prediction(model, test_dataset)

if __name__ == "__main__":
    main()