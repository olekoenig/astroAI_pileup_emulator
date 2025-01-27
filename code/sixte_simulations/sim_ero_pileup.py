#!/usr/bin/env python3
"""Author: Ole Koenig (ole.koenig@fau.de)

This script runs simulations on a grid of different parameters using
the eROSITA pile-up simulation scripts.

Changelog:
2020-11-XX: Add eSASS pipeline
2020-11-26: Add option to simulate multiple spectra for chi2 minimization
2020-12-02: All output files except for spectra on scratch disks, mathpha for summing
2020-12-05: Re-write to call script directly and not through shell scripts

------
Notes: 
------

On the Remeis cluster, you'll need to 
   export HEADASNOQUERY= ; export HEADASPROMPT=/dev/null;
   module unload caldb; module load esass;
before submitting the jobs.

"""

import os
import numpy as np
import sys

sys.path.insert(1, '/home/koenig/work/erosita/pileup/pileupsoft')
# sys.path.insert(1, '/home/koenig/work/erosita/pileup/sim_apec')
from pileupsim import Pileupsim
import config


k = 8.617333262145e-5  # Boltzmann constant [eV/K]

# Define the fluxes you want to simulate, in erg/cm^2/s and in
# energyband given by EMIN, EMAX in config.py
fluxes = np.linspace(1.4e-8, 2.2e-8, 20)

# Define the simulated absorption columns in units of 10^22 cm^-2
nhs = np.linspace(0.001, 0.01, 10)

# Define the simulated temperatures in units of eV
kts = np.linspace(25, 30, 20)

# Number of spectra for each parameter combination, which are averaged over
n_sim = 1000

# Parameters specific to the parallelization via Slurm
WALLTIME = "3-24:00:00"
SLURMFILE = "sim_ero_pileup.slurm"
JOBNAME = "ero_pileup"
QUEUE = "dlr"
MEMORY = "2G"


def write_bbody_parfile(nh, ktbb):
    """
    :param nh: Absorption column (10^22 cm^-2)
    :param ktbb: Black body temperature (eV)
    :returns: Path to parameter file

    Write the spectrum file to be read in by simputfile. The flux is
    scaled by the parameter Src_Flux in the creation of the SIMPUT so
    the normalization of the black body is irrelevant.

    .. ToDo: Is there a way to do this without raw writing, e.g. with pyspec?

    """
    parfile = "{}{:f}e22_{:f}eV.par".format(config.ISISPARFILEDIR, nh, ktbb)
    with open(parfile, 'w') as fp:
        fp.write("tbnew_simple(1)*bbody(1)\n")
        fp.write(" idx  param           tie-to  freeze         value         min         max\n")
        fp.write("  1  tbnew_simple(1).nH   0     0         {:f}           0         100  10^22/cm^2\n".format(nh))
        fp.write("  2  bbody(1).norm        0     0                1           0       1e+10 \n")
        fp.write("  3  bbody(1).kT          0     0         {:f}        0.01         100  keV\n".format(ktbb/1000))
    return parfile

def write_tmap_parfile(nh, kt):
    """
    :param nh: Absorption column [10^22 cm^-2]
    :param kt: Atmosphere temperature [eV]
    :returns: Path to parameter file
    """
    temp = kt/k  # [K]
    parfile = "{}{:f}e22_{:f}K.par".format(config.ISISPARFILEDIR, nh, temp)
    with open(parfile, 'w') as fp:
        fp.write("tbnew_simple(1)*tmap(1)\n")
        fp.write(" idx  param           tie-to  freeze         value         min         max\n")
        fp.write("  1  tbnew_simple(1).nH   0     0         {:f}           0         100  10^22/cm^2\n".format(nh))
        fp.write("  2  tmap(1).norm         0     0                1           0       1e+10  \n")
        fp.write("  3  tmap(1).dlg          0     0         0.100001         0.1         2.2  \n")
        fp.write("  4  tmap(1).T            0     0         {:.1f}      100000     1000000  \n".format(temp))
    return parfile

def write_apec_parfile(nh, apec_kt):
    parfile = "{}apec_{:f}e22_{:f}eV.par".format(config.ISISPARFILEDIR, nh, apec_kt)
    with open(parfile, 'w') as fp:
        fp.write("tbnew_simple(1)*apec(1)\n")
        fp.write(" idx  param           tie-to  freeze         value         min         max\n")
        fp.write("  1  tbnew_simple(1).nH   0     0         {:f}           0         100  10^22/cm^2\n".format(nh))
        fp.write("  2  apec(1).norm         0     0                1           0       1e+10  \n")
        fp.write("  3  apec(1).kT           0     0         {:f}       0.008          64  keV\n".format(apec_kt/1000))
        fp.write("  4  apec(1).Abundanc     0     1                1           0           5  \n");
        fp.write("  5  apec(1).Redshift     0     1                0      -0.999          10  \n");
    return parfile

def write_slurm_script(cmds):
    """
    Writes a slurm script which can be submitted with sbatch.

    :param pars: NamedTuple containing slurm script name, queue,
                 memory, walltime, jobname information
    :returns: None
    """
    with open(SLURMFILE, 'w') as fp:
        fp.write("#!/bin/bash")
        fp.write(f"\n#SBATCH --partition {QUEUE}")
        fp.write(f"\n#SBATCH --account {QUEUE}")
        fp.write(f"\n#SBATCH --mem {MEMORY}")
        fp.write(f"\n#SBATCH --job-name {JOBNAME}")
        # fp.write("\n#SBATCH --ntasks={}".format(number of cores))
        fp.write("\n#SBATCH --time {}".format(WALLTIME))
        fp.write(f"\n#SBATCH --output {config.LOGDIR}{JOBNAME}.out-%a")
        fp.write(f"\n#SBATCH --error {config.LOGDIR}{JOBNAME}.err-%a")
        fp.write("\n#SBATCH --array 0-{}".format(len(cmds)-1))
   
        fp.write("\n\ncd {}".format(os.getcwd()))

        for ii in range(len(cmds)):
            fp.write("\nCOMMAND[{}]=\"{}\"".format(ii, cmds[ii]))
  
        fp.write("\n\nsrun /usr/bin/nice -n +15 ${COMMAND[$SLURM_ARRAY_TASK_ID]} \n")
  
    os.system(f"cat {SLURMFILE} && echo && echo sbatch {SLURMFILE}")
    return

def main():
    # The function to sum the spectra, mathpha, can only use filenames of limited length
    # --> we need a unique, short filename --> just use ascending integers
    counter = 0
    cmds = []

    for nh in nhs:
        for kt in kts:
            
            # Write the spectrum file. As the source flux is scaled up in
            # the SIMPUT via Src_Flux parameter, the normalization
            # doesn't matter and we can use the same spectrum file for
            # all fluxes and only change it if we change N_H/kT
            parfile = write_bbody_parfile(nh, kt)

            for flux in fluxes:
                cmds.append(f"python3 {config.PILEUPSIM_DIR}run.py --flux={flux} --parfile={parfile} " +
                            f"--shortname={str(counter)} --n_sim={n_sim} --verbose=-1")
                counter += 1

    write_slurm_script(cmds)


if __name__ == "__main__":
    main()
