% #!/usr/bin/env isis
% -*- mode: slang; mode: fold -*-

require("isisscripts");

Rmf_OGIP_Compliance=0;

set_fit_statistic("cash");
set_fit_method("subplex");

variable phafile = "/pool/burg1/astroai/pileup/sims/spec_piledup/1.223821Em10cgs_0.981670nH_0.139405kT_piledup_annulus3.fits";
% variable phafile = "/pool/burg1/astroai/pileup/sims/spec_piledup_abs_bbody_grid/0.020043Em10cgs_0.971797nH_0.130109kT_piledup.fits";
variable idx = load_data(phafile);

xnotice_en(idx, 0.2, 8);
list_data;

variable kt, src_flux, nh;
(kt, src_flux, nh) = fits_read_key(phafile, "KT", "SRC_FLUX", "NH");

fit_fun("tbnew_simple(1)*bbody(1)");
set_par("tbnew_simple(1).nH", nh);
set_par("bbody(1).kT", kt);

fit_counts;

emcee_hammer(10000; walkers=100, driver="fork", init="gauss;sigma=0.001", progress="report;n=100",
	     outfile="/pool/burg1/astroai/mcmc/emcee-chain_" + path_basename_sans_extname(phafile) + ".fits",
	     clobber="no");

