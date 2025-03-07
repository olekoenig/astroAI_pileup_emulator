#!/usr/bin/env python3
"""Author: Ole Koenig (ole.koenig@fau.de)

Caller for eROSITA pile-up simulations with SIXTE

The setup was tested on Ubuntu 20.04 LTS with Python 3.8.5 under tcsh.

+++++++++++++++++
Required packages
+++++++++++++++++

* pandas-1.2.0
* astropy-4.2
* sixtesoft (see the SIXTE repository sixte/test/e2e/python_wrapper/sixtesoft/)

Example call (currently highly dependent on my setup):

   setenv HEADASNOQUERY ; setenv HEADASPROMPT /dev/null ;
   export HEADASNOQUERY= ; export HEADASPROMPT=/dev/null;
   module unload caldb; module load esass;
   ./run.py --flux=1.e-10 --parfile=powerlaw.par --n_sim=2 --verbose=1

------
To Do:
------

* Write detailed tool description in argparse

"""

import argparse

import config
from pileupsim import Pileupsim


def arguments():
    parser = argparse.ArgumentParser(description = ("Pile-up simulation setup:\nThis script performs pile-up simulations"
                                                    "for the eROSITA telescope. WRITE DETAILLED DESCRIPTION!"),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--flux', type=float, help='Absorbed source flux in units of erg/cm^2/s')
    parser.add_argument('--parfile', type=str, help='Parameter file in ISIS notation')
    parser.add_argument('--n_sim', type=int, help='Number of spectra to be averaged over', default=1)
    parser.add_argument('--shortname', type=str, help='A (unique) file naming for mathpha', default="0")
    parser.add_argument('-v', '--verbose', type=int, help='Increase verbosity', default=0)
    parser.add_argument('--clobber', type=str, choices=["yes", "no"], default="no", help='Clobber existing files')
    return parser.parse_args()


def main():
    args = arguments()
    
    pileupsim = Pileupsim(flux = args.flux, parfile = args.parfile, background = "no",
                          n_sim = args.n_sim, shortname = args.shortname,
                          write_impactlist = False, verbose = args.verbose, clobber = args.clobber)

    pileupsim.create_simput()
    pileupsim.run_sixtesim()
    pileupsim.run_makespec()
    pileupsim.clean_up()


if __name__ == "__main__":
    main()
