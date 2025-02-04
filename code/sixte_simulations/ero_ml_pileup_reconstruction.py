from pileupsim import Pileupsim
from sim_ero_pileup import write_bbody_parfile,write_slurm_script

import numpy as np

# Define the fluxes you want to simulate, in erg/cm^2/s and in
# energyband given by EMIN, EMAX in config.py
fluxes = np.logspace(-10, -8, 20)

nhs = [1.0]  # [10^22 cm^-2]
# gammas = np.linspace(1.7, 2.5, 10)
kts = np.linspace(30, 100, 100)  # [eV]

def main():
    # cmds = []
    total = len(fluxes) * len(nhs) * len(kts)
    counter = 0

    for nh in nhs:
        for kt in kts:
            # Write the spectrum file. As the source flux is scaled up in
            # the SIMPUT via Src_Flux parameter, the normalization
            # doesn't matter, and we can use the same spectrum file for
            # all fluxes and only change it if we change N_H/kT
            parfile = write_bbody_parfile(nh, kt)

            for flux in fluxes:
                print(f"Counter: {counter+1}/{total}, flux: {flux} cgs, N_H = {nh} e22 cm-2, kt = {kt} eV")
                pileupsim = Pileupsim(flux = flux, parfile = parfile, background = "no",
                                      verbose = -1, clobber = "yes")

                pileupsim.create_simput()
                pileupsim.run_sixtesim()
                pileupsim.run_makespec()
                pileupsim.clean_up()

                #cmds.append(f"python3 {os.getcwd()}/run.py --flux={flux} --parfile={parfile}")
                counter += 1

    #write_slurm_script(cmds)

if __name__ == "__main__":
    main()
