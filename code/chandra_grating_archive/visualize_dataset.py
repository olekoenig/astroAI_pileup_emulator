from astropy.io import fits
from astropy.table import Table
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages


def get_first_order_spectra(fname):
    tbl = Table.read(fname, hdu=1)

    mask_heg_m1 = (tbl['TG_M'] == -1) & (tbl['TG_PART'] == 1)
    mask_heg_p1 = (tbl['TG_M'] == 1) & (tbl['TG_PART'] == 1)
    
    mask_meg_m1 = (tbl['TG_M'] == -1) & (tbl['TG_PART'] == 2)
    mask_meg_p1 = (tbl['TG_M'] == 1) & (tbl['TG_PART'] == 2)
    return tbl[mask_heg_m1], tbl[mask_heg_p1], tbl[mask_meg_m1], tbl[mask_meg_p1]

def get_zero_order_spectra(fname):
    if not os.path.exists(fname):
        print(f"{os.path.basename(fname)} does not exist!")
    else:
        return fits.getdata(fname, ext=1)

def main():
    # pha2_list = glob.glob("/pool/burg1/astroai/chandra_grating_archive/*/repro/acisf*_repro_pha2.fits")
    pha2_list = ["/pool/burg1/astroai/chandra_grating_archive/11044/repro/acisf11044_repro_pha2.fits"];
    zero_order_list = [os.path.dirname(fname).strip("repro") + "zeroth_order.pi" for fname in pha2_list]

    for index in range(len(pha2_list)):
        print(f"{pha2_list[index]}: {index+1}/{len(pha2_list)}")
        
        heg_m1, heg_p1, meg_m1, meg_p1 = get_first_order_spectra(pha2_list[index])
        zo = get_zero_order_spectra(zero_order_list[index])
        
        fig, ax = plt.subplots(nrows=3, ncols=1)
        
        ax[0].set_title(os.path.basename(pha2_list[index]))
        
        ax[0].step(zo['CHANNEL'], zo['COUNTS'], color="black", label="Zero order", where='mid')
        ax[1].step(heg_m1['CHANNEL'][0], heg_m1['COUNTS'][0], color="red", label="HEG -1", where='mid')
        ax[1].step(heg_p1['CHANNEL'][0], heg_p1['COUNTS'][0], color="green", label="HEG +1", where='mid')
        ax[2].step(heg_m1['CHANNEL'][0], heg_m1['COUNTS'][0], color="blue", label="MEG -1", where='mid')
        ax[2].step(heg_p1['CHANNEL'][0], heg_p1['COUNTS'][0], color="cyan", label="MEG +1", where='mid')
        
        # print(type(heg_m1['COUNTS']))
        # print(type(heg_m1['CHANNEL']))
        
        # ax[1].plot(heg_m1['CHANNEL'], heg_m1['COUNTS'], '-o', color="red", label="First order -1", alpha=0.5)
        # ax[1].plot(heg_p1['CHANNEL'], heg_p1['COUNTS'], '-o', color="blue", label="First order +1", alpha=0.5)
        
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        
        ax[0].set_xlim(10, max(zo['CHANNEL'])+1)
        ax[1].set_xlim(3e3, 8193)
        ax[2].set_xlim(3e3, 8193)
        
        ax[0].set_xscale('log', base=10)
        ax[1].set_xscale('log', base=10)
        ax[2].set_xscale('log', base=10)
        ax[0].set_yscale('log', base=10)
        ax[1].set_yscale('log', base=10)
        ax[2].set_yscale('log', base=10)
        
        ax[-1].set_xlabel("Channel")
        ax[0].set_ylabel("Counts")
        ax[1].set_ylabel("Counts")
        ax[2].set_ylabel("Counts")
        
        plt.tight_layout()
        plt.savefig(os.path.basename(pha2_list[index])+".pdf")
        
        exit(0)

        
if __name__ == "__main__":
    main()
    
# with PdfPages('visualize_dataset.pdf') as pdf:
#     for fig in figs:
#         pdf.savefig(fig, bbox_inches='tight') 
