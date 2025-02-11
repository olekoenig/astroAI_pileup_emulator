# from pileupsim import Pileupsim
from sim_ero_pileup import write_bbody_parfile
from scipy.stats import qmc
import numpy as np
import os

# Run it in parallel with:
# cat batch.sh | xargs -P 30 -I {} tcsh -c "{}"

def generate_latin_hypercube(n_points, flux_min, flux_max, kt_min, kt_max):
    sampler = qmc.LatinHypercube(d = 2)
    sample = sampler.random(n = n_points)
    l_bounds = [np.log(flux_min), kt_min]
    u_bounds = [np.log(flux_max), kt_max]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    sample_scaled[:,0] = np.exp(sample_scaled[:,0])
    return sample_scaled

def write_parfiles(samples):
    """
    Write the spectrum file. As the source flux is scaled up in the SIMPUT
    via Src_Flux parameter, the normalization doesn't matter, and we can
    use the same spectrum file for all fluxes and only change it if we
    change N_H/kT
    """
    used_fluxes = []
    parfiles = []
    for sample in samples:
        flux = sample[0]
        if flux not in used_fluxes:
            used_fluxes.append(flux)
            nh = 1
            kt = sample[1]
            parfile = write_bbody_parfile(nh, kt)
            parfiles.append(parfile)
        else:
            pass
    return parfiles

def main():
    # Define the fluxes you want to simulate, in erg/cm^2/s and in
    # energyband given by EMIN, EMAX in config.py
    flux_min, flux_max = 1e-12, 1e-8  # [cgs]
    kt_min, kt_max = 30, 200  # [eV]
    nh = 1  # [10^22 cm^-2]

    n_points = 10000
    samples = generate_latin_hypercube(n_points, flux_min, flux_max, kt_min, kt_max)
    parfiles = write_parfiles(samples)

    cmds = []
    for idx, (sample, parfile) in enumerate(zip(samples, parfiles), start=1):
        flux = sample[0]
        kt = sample[1]
        print(f"{idx}/{n_points}, flux: {flux} cgs, N_H = {nh} e22 cm-2, kt = {kt} eV, parfiles = {os.path.basename(parfile)}")

        #pileupsim = Pileupsim(flux = flux, parfile = parfile, background = "no",
        #                      verbose = -1, clobber = "yes")
        #pileupsim.create_simput()
        #pileupsim.run_sixtesim()
        #pileupsim.run_makespec()
        #pileupsim.clean_up()

        cmds.append(f"python3 {os.getcwd()}/run.py --flux={flux} --parfile={parfile} --verbose=-1")

    with open("batch.sh", 'w') as fp:
        for cmd in cmds:
            fp.write(cmd + "\n")

if __name__ == "__main__":
    main()
