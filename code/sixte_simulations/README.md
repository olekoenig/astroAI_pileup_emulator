# eROSITA Pile-up scripts

This set of scripts enables running SIXTE simulations for a given spectrum and flux and extracting the spectrum with srctool. In particular it

1. Creates a SIMPUT from the given parameter file and flux
2. Calculates the eROSITA slew GTI from the attitude file
3. Repeats n_sim times:
  3.1 A simulation with erosim for the TMs specified in config.py
  3.2 Runs the PHA2PI correction in ero_calevents and prepares for eSASS processing
  3.3 Runs evtool/srctool to create an image, spectrum, lightcurve
4. Averages all spectra, lightcurve and images

Please adapt the configuration file to your system (in particular the storage areas).


### Example call for parallelizaion

```
./run.py --flux=1.3e-8 --parfile=powerlaw.par --n_sim=10
```
