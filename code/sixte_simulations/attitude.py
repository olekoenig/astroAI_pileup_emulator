"""
Author: Ole Koenig (ole.koenig@fau.de)
 
A script to write an attitude file for an eROSITA SIXTE simulation.

----------
Changelog:
----------

------
To Do:
------

"""

from astropy.io import fits
import config


def write_attitude():
    """
    Writes an attitude file from the CORRATT extension of the measured eventfile
    (produced by eSASS' evtool). Currently, the file is written from the 
    attitude of TM1.
    """
    in_extname = "CORRATT1"
    with fits.open(config.ERO_EVENTFILE) as hdul:
        ext1 = hdul[in_extname]
        ext1.name = "ATTITUDE"
        ext1.header['HISTORY'] = f"Attitude information copied by Ole Koenig from the evtool-created file {config.ERO_EVENTFILE}[{in_extname}]"
        ext1.data["RA"] -= 180  # SIXTE right ascension defined in -180 to 180 deg
        ext1.writeto(config.ATTITUDE_FILE, overwrite = True)
    return config.ATTITUDE_FILE
    

def main():
    # Testing
    from pileupsim import Pileupsim

    ero_flash_lc = "/userdata/data/koenig/NovaRet2020/flash/020_LightCurve_00001.fits"
    
    pileupsim = Pileupsim(flux=1.0e-10, parfile="/home/koenig/work/erosita/pileup/bbody_fit_wo_pileup.par")    
    pileupsim.create_simput()
    pileupsim.write_attitude_and_gti()

    # Check the exposure of the calculated GTI file
    with fits.open(config.MASTER_GTI) as hdul:
        tstop = hdul[1].data['STOP']
        tstart = hdul[1].data['START']
        
        # If the sky position of the SIMPUT is not covered by the attitude file the GTI file is empty
        if len(hdul[1].data['START']) > 0:
            exposure = tstop - tstart
            print("exposure={} TSTART={}, TSTOP={}".format(exposure, tstart, tstop))
            if len(exposure) > 1:
                print(f"*** More than one GTI in: {config.ATTITUDE_FILE}")
                

    # Calculate the exposure of the observed flash lightcurve to compare it to the GTI (the times should match!)
    with fits.open(ero_flash_lc) as hdul:
        tstop = hdul[1].header['TSTOP']
        tstart = hdul[1].header['TSTART']
        exposure_measured = tstop - tstart
        print("{}: exposure={} TSTART={}, TSTOP={}".format(ero_flash_lc, exposure_measured, tstart, tstop))

        
if __name__ == "__main__":
    main()
