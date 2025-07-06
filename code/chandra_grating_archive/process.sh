#!/bin/tcsh

ciao -o

# cd /pool/burg1/astroai/chandra_grating_archive
cd /pool/burg1/astroai/good_grating_candidates/

set obsid=$1
set obsid_padded = `printf "%05d" $obsid`
set bkgfile=${obsid_padded}_zerobkg.reg
set srcfile=${obsid_padded}_zerosrc.reg

echo "Downloading ${obsid}"
download_chandra_obsid ${obsid}

cd $obsid

# Extract higher order dispersed spectra
echo "Running chandra_repro for ObsID ${obsid}"
punlearn chandra_repro
chandra_repro . repro

## Create zeroth order region files
echo "Extracting regions for ObsID ${obsid}"
dmlist repro/acisf${obsid_padded}_repro_evt2.fits"[REGION]" data,raw | grep circle | awk '{print "circle(" $3 "," $4 "," $5 ")"}' > ${srcfile}
dmlist repro/acisf${obsid_padded}_repro_evt2.fits"[REGION]" data,raw | grep circle | awk '{print "annulus(" $3 "," $4 "," $5*1.5 "," $5*5.5 ")"}' > ${bkgfile}

## Create image with 0th and 1st dispersion indicated for region extraction
echo "Running ds9 for ObsID ${obsid} region file cross-check"
ds9 repro/acisf${obsid_padded}_evt1a.fits -scale log -region load ${srcfile} -region load ${bkgfile} -saveimage ${obsid_padded}_inspect.jpeg -quit

## Extract 0th order undispersed (potentially piled-up) spectrum
echo "Running specextract for ObsID ${obsid}"
punlearn specextract
specextract "repro/acisf${obsid_padded}_repro_evt2.fits[sky=region(${srcfile})][tg_m=0,Null]" ${obsid_padded}_zeroth_order bkgfile="repro/acisf${obsid_padded}_repro_evt2.fits[sky=region(${bkgfile})]"
