import matplotlib.pyplot as plt
import torch
from astropy.io import fits
import numpy as np
import os
import random
from typing import Tuple

from data import load_and_split_dataset
from neuralnetwork import ConvSpectraNet
import config

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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
    axis.set_ylabel(fr'$\frac{{\text{{{data1_label}}}}}{{\text{{{data2_label}}}}}$')
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

def _get_params_from_output(output: torch.Tensor) -> Tuple[float, float, float, float, float, float]:
    """Split parameter means (first numbers) and log variance (last half of numbers).
    It does not quite correspond to the variance because I'm applying softplus
    instead of exp on the logarithm of the variance for numerical stability."""
    mu = output[:config.DIM_OUTPUT_PARAMETERS]
    raw_log_var = output[config.DIM_OUTPUT_PARAMETERS:]
    var = torch.nn.functional.softplus(raw_log_var) + 1e-6

    log_kt, flux, nh = mu.tolist()
    log_kt_err, flux_err, nh_err = torch.sqrt(var).tolist()

    kt = np.exp(log_kt)  # [keV]
    kt_err = np.exp(log_kt_err)  # [keV]

    flux *= 1e-12  # [erg/cm^2/s]
    flux_err *= 1e-12  # [erg/cm^2/s]

    return kt, flux, nh, kt_err, flux_err, nh_err

def evaluate_parameter_prediction(model, test_dataset):
    model.eval()

    kt_true, flux_true, nh_true = [], [], []
    kt_pred, flux_pred, nh_pred = [], [], []
    # kt_errs, flux_errs, nh_errs = [], [], []

    with torch.no_grad():
        for input, target in test_dataset:
            # output = model(input) # .squeeze(0)
            output = model(input.unsqueeze(0))[0]  # from [1024] to [1,1024] (for convolutional layer)

            # kt, flux, nh, kt_e, flux_e, nh_e = _get_params_from_output(output)
            kt, flux, nh = output.tolist()

            kt_pred.append(np.exp(kt))
            flux_pred.append(np.exp(flux) * 1e-12)
            nh_pred.append(np.exp(nh))

            # kt_errs.append(kt_e)
            # flux_errs.append(flux_e)
            # nh_errs.append(nh_e)

            kt_true.append(target[0].item())
            flux_true.append(target[1].item() * 1e-12)
            nh_true.append(target[2].item())

    fig, axes = plt.subplots(ncols = 3, figsize=[18/2.54, 6/2.54])
    #axes[0].set_xlim(min(kt_true), max(kt_true))
    #axes[0].set_ylim(min(kt_true)/10, max(kt_true)*10)
    # axes[1].set_xlim(min(flux_true), max(flux_true))
    # axes[1].set_ylim(min(flux_true), max(flux_true))
    #axes[2].set_xlim(min(nh_true), max(nh_true))
    #axes[2].set_ylim(min(nh_pred), max(nh_pred))

    # axes[0].errorbar(kt_true, kt_pred, yerr=kt_errs, alpha=0.1, ms=2, ecolor="silver", elinewidth=0.1, fmt=".")
    # axes[1].errorbar(flux_true, flux_pred, yerr=flux_errs, alpha=0.1, ms=2, ecolor="silver", elinewidth=0.1, fmt=".")
    # axes[2].errorbar(nh_true, nh_pred, yerr=nh_errs, alpha=0.1, ms=2, ecolor="silver", elinewidth=0.1, fmt=".")

    axes[0].scatter(kt_true, kt_pred, alpha=0.1, s=2)
    axes[1].scatter(flux_true, flux_pred, alpha=0.1, s=2)
    axes[2].scatter(nh_true, nh_pred, alpha=0.1, s=2)

    axes[0].axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
    axes[1].axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
    axes[2].axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)

    axes[0].set_xlabel(r"True $kT$ [keV]")
    axes[0].set_ylabel(r"Predicted $kT$ [keV]")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"True flux [$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$]")
    axes[1].set_ylabel(r"Predicted flux [$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$")

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"True $N_\mathrm{H}$ [$10^{22}\,\mathrm{cm}^{-2}$]")
    axes[2].set_ylabel(r"Predicted $N_\text{H}$ [$10^{22}\,\mathrm{cm}^{-2}$]")

    plt.tight_layout()
    plt.savefig("testdata.pdf")

def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()

    model = ConvSpectraNet()
    model.load_state_dict(torch.load(config.DATA_NEURAL_NETWORK + "model_weights.pth", map_location="cpu"))

    # evaluate_on_test_spectrum(model, test_dataset)
    # evaluate_on_real_spectrum(model, "/pool/burg1/novae4ole/V1710Sco_em04_PATall_820_SourceSpec_00001.fits", out_pha_file = "test.fits")
    evaluate_parameter_prediction(model, test_dataset)

if __name__ == "__main__":
    main()