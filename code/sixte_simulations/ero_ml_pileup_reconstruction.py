from scipy.stats import qmc
import numpy as np
import os

# Run it in parallel with:
# cat batch.sh | xargs -P 30 -I {} tcsh -c "{}"

import config

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

def generate_latin_hypercube(n_points, flux_min, flux_max, kt_min,
                             kt_max, nh_min, nh_max):
    rng = np.random.default_rng(config.DATALOADER_RANDOM_SEED)
    sampler = qmc.LatinHypercube(d = 3, rng=rng)
    sample = sampler.random(n = n_points)
    l_bounds = [np.log(flux_min), kt_min, nh_min]
    u_bounds = [np.log(flux_max), kt_max, nh_max]
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
            kt = sample[1]
            nh = sample[2]
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
    nh_min, nh_max = 0.2, 2  # [10^22 cm^-2]

    n_points = 20000
    samples = generate_latin_hypercube(n_points, flux_min, flux_max,
                                       kt_min, kt_max, nh_min, nh_max)
    parfiles = write_parfiles(samples)

    cmds = []
    for idx, (sample, parfile) in enumerate(zip(samples, parfiles), start=1):
        flux = sample[0]
        kt = sample[1]
        nh = sample[2]
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
