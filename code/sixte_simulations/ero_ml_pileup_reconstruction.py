from pileupsim import Pileupsim
import sixtesoft
from sim_ero_pileup import write_bbody_parfile,write_slurm_script
import config

import numpy as np
import os

k = 8.617333262145e-5  # Boltzmann constant [eV/K]

# Define the fluxes you want to simulate, in erg/cm^2/s and in
# energyband given by EMIN, EMAX in config.py
fluxes = np.linspace(1.0e-10, 1.0e-7, 10)

# Define the simulated absorption columns in units of 10^22 cm^-2
nhs = [1.0]

# Define the simulated temperatures in units of eV
kts = np.linspace(20, 1000, 10)

# Parameters specific to the parallelization via Slurm
WALLTIME = "3-24:00:00"
SLURMFILE = "sim_ero_pileup.slurm"
JOBNAME = "ero_pileup"
QUEUE = "dlr"
MEMORY = "2G"



def main():
    cmds = []

    for nh in nhs:
        for kt in kts:
            
            # Write the spectrum file. As the source flux is scaled up in
            # the SIMPUT via Src_Flux parameter, the normalization
            # doesn't matter and we can use the same spectrum file for
            # all fluxes and only change it if we change N_H/kT
            parfile = write_bbody_parfile(nh, kt)

            for flux in fluxes:
                cmds.append(f"python3 {os.getcwd()}/run.py --flux={flux} --parfile={parfile}")

    write_slurm_script(cmds)


if __name__ == "__main__":
    main()
