require("isisscripts");
require("subs.sl");

variable df = get_dataset();

xlog; ylog;
Rmf_OGIP_Compliance = 0;

variable ii;
_for ii (0, length(df)-1, 1){
  vmessage("%d/%d", ii+1, length(df));
  
  % if (get_flux_from_filename(df[ii]) < 1e-10) continue;

  variable phafile = df[ii];
  variable idx = load_data(phafile);
  
  variable parfile, src_flux, respfile, ancrfile, exposure, telescop, instrume, filters;
  (parfile, src_flux, respfile, ancrfile, exposure, telescop, instrume, filters) =
    fits_read_key(phafile, "PARFILE", "SRC_FLUX", "RESPFILE", "ANCRFILE",
		  "EXPOSURE", "TELESCOP", "INSTRUME", "FILTER");

  % To put onto right grid. Necessary?
  load_par(parfile);
  variable fitfun = get_fit_fun();
  fit_fun("enflux(1," + fitfun + ")");
  freeze("*");
  set_par("enflux(1).E_min", 0.2);
  set_par("enflux(1).E_max", 2.0);
  set_par("enflux(1).enflux", erg2keV(src_flux), 1);
  variable ss;
  eval_counts(&ss);

  variable outname = TARGET_DATADIR + strreplace(path_basename(phafile), "piledup", "nonpiledup");

  variable model = _A(get_model_counts(idx));
  fits_write_pha_file(outname, typecast(model.value, Integer_Type);
		      TELESCOP = telescop,
		      INSTRUME = instrume,
		      FILTER = filters,
		      EXPOSURE = exposure,
		      AREASCAL = 1.0,
		      BACKFILE = "none",
		      RESPFILE = respfile,
		      ANCRFILE = ancrfile);

  delete_data(idx);
}