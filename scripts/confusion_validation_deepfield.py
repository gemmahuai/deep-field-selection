# Full photometric + redshift validation of spectral confusion injection in the deep field
# 1. Pick coordinates where roughly 100 observations will be taken per fiducial channel (Jean's datafile)
# 2. Pick a subset of sources from the COSMOS 32k refcat (~thousand), in a small area surrounding the coordinates
# 3. Controlled run: isolated photometry
# 4. Validation run: Tractor photometry only including sub-threshold sources
# 5. Run through Photo-z
### Gemma Huai, 01/11/2025
### see Tractor_in_deep_field.ipynb for more analysis

import sys
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, hstack
from astropy.io import fits
from astropy.coordinates import SkyCoord, Distance
from astropy.coordinates import BarycentricMeanEcliptic
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time 
import argparse
import os

import SPHEREx_ObsSimulator as SPobs
from SPHEREx_Simulator_Tools import data_filename
import SPHEREx_InstrumentSimulator as SPinst
from spherex_parameters import load_spherex_parameters
import SPHEREx_SkySimulator as SPsky
from SPHEREx_SkySimulator import QuickCatalog
from SPHEREx_SkySimulator import Catalog_to_Simulate

survey_plan_file = data_filename('spherex_survey_plan_R2.fits')
SPHEREx_Pointings = SPobs.Pointings(input_file=survey_plan_file, 
                                        Gaussian_jitter = 0.0, roll_angle='psi2')
# Load instrument and project parameters as a dictionary
spherex_parameters = load_spherex_parameters()
Scene = SPsky.Scene(SPHEREx_Pointings,
                        zodi_model=SPsky.zodicalc.ModifiedKelsallModelWithHPFT())

Channels = Table.read(data_filename('Channel_Definition_03022021.fits'))
# Instrument (no noise - get rid of noise_model and dark_current_model --> dark current, read noise, photon noise, zodi = 0)
trim = 32
ds = 4
SPHEREx_Instrument = SPinst.Instrument(
    instrument_data=spherex_parameters,
    psf=data_filename("psf/simulated_PSF_database_centered_v3_og.fits"),
    psf_downsample_by_array={1: ds, 2: ds, 3: ds, 4: ds, 5: ds, 6: ds},
    psf_trim_by_array={1: trim, 2: trim, 3: trim, 4: trim, 5: trim, 6: trim},
    # noise_model=SPinst.white_noise,
    # dark_current_model=SPinst.poisson_dark_current,
    lvf_model=SPinst.Tabular_Bandpass()
)

##--------- functions -------------
# function finding the new, corrected fits file
def find_sed_fits_file_corrected(index, tractorID):
    if index < 60000:
        #print("0 - 60000")
        DIR = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/COSMOS_ALL_HIRES_SEDS_010925/0_60000/"
    elif index < 120000:
        #print("60000 - 120000")
        DIR = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/COSMOS_ALL_HIRES_SEDS_010925/60000_120000/"
    else:
        #print("120000 - ")
        DIR = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/COSMOS_ALL_HIRES_SEDS_010925/120000_166041/"
    
    filename = DIR + f"cosmos2020_hiresSED_FarmerID_{tractorID:07d}_corrected.fits"
    return filename


## Load COSMOS reference catalogs
COSMOS_tab = Table.read('/Users/gemmahuai/Desktop/CalTech/SPHEREx/SPHEREx_2023/COSMOS2020_FARMER_R1_v2.1_p3_in_Richard_sim_2023Dec4.fits', format='fits')
# CUT_xmatch = np.loadtxt('../source_selection/final_cut_boolarray.txt', dtype=bool)
# COSMOS_tab = COSMOS_tab[CUT_xmatch]

# # idx_refcat = np.loadtxt("./newrefcat_bool_COSMOSzch1.txt", dtype=bool)
# idx_refcat = np.loadtxt("../source_selection/final_colormag_cut_boolarray_nov2024.txt", dtype=bool)
idx_refcat = np.loadtxt("../source_selection/cosmos166k_posmatch_boolarray.txt", dtype=bool)


## Move COSMOS to the deep field
file = "../source_selection/data/obs_per_ecliptic_lat_R3.fits"
with fits.open(file) as hdul:
    data = hdul[1].data

# find ecliptic latitude where Nobs ~ 100 / channel --> 102*100 obs in total
idx = np.where(abs(data['observations']-(102*100)) == (abs(data['observations']-(102*100))).min())[0][0]
elat = data['ecliptic_lat'][idx]
elon = 0 * u.deg # deg # longitude doesn't matter, assuming symmetric about NEP.
crd = SkyCoord(elon, elat * u.deg, frame=BarycentricMeanEcliptic)  # NEP at (lon=0, lat=90)
ra0 = crd.transform_to('icrs').ra.deg
dec0 = crd.transform_to('icrs').dec.deg
print('RA, DEC for the 100 obs / channel is ', ra0, dec0)

# move COSMOS centered around this calculated coordinate pair.
ra_c = (COSMOS_tab['ALPHA_J2000'].max() - COSMOS_tab['ALPHA_J2000'].min()) / 2 + COSMOS_tab['ALPHA_J2000'].min()
dec_c = (COSMOS_tab['DELTA_J2000'].max() - COSMOS_tab['DELTA_J2000'].min()) / 2 + COSMOS_tab['DELTA_J2000'].min()
if "ra_deep" not in COSMOS_tab.colnames:
    COSMOS_tab.add_column((COSMOS_tab['ALPHA_J2000'].copy() + (ra0 - ra_c)), name="ra_deep")
    COSMOS_tab.add_column((COSMOS_tab['DELTA_J2000'].copy() + (dec0 - dec_c)), name="dec_deep")

## select a subsample surrounding the chosen coordinate pair ~ 0.2 * 0.2 deg^2 area --> 1k refcat sources
ra_min = ra0 - 0.1
ra_max = ra0 + 0.1
dec_min = dec0 - 0.1
dec_max = dec0 + 0.1
## count number of cosmology sources in the patch
flag_sub = (COSMOS_tab['ra_deep']>=ra_min) & \
           (COSMOS_tab['ra_deep']<=ra_max) & \
           (COSMOS_tab['dec_deep']>=dec_min) & \
           (COSMOS_tab['dec_deep']<=dec_max)
print("Number of refcat sources in the selected area = ", len(COSMOS_tab[idx_refcat & flag_sub]))


## Isolated photometry
i = 0
tID_central = COSMOS_tab['Tractor_ID'][idx_refcat & flag_sub][i]
ra = COSMOS_tab['ra_deep'][idx_refcat & flag_sub][i]
dec = COSMOS_tab['dec_deep'][idx_refcat & flag_sub][i]
QC = QuickCatalog(SPHEREx_Pointings, SPHEREx_Instrument, Scene, Use_Tractor=False, spectral_channel_table=Channels,\
                  #do_not_fit=source_name,\
                  subpixel_offset_x=0, subpixel_offset_y=0)
Sources_to_Simulate_close = Catalog_to_Simulate()
# The central source to fit (refcat)
central_sed_fn = find_sed_fits_file_corrected(COSMOS_tab['col1'][idx_refcat & flag_sub][i],
                                              COSMOS_tab['Tractor_ID'][idx_refcat & flag_sub][i])
Sources_to_Simulate_close.load_single(name=f"Central tID {int(tID_central)}", 
                                      ra=ra*u.deg, 
                                      dec=dec*u.deg, 
                                      inputpath=central_sed_fn)
# run QuickCatalog
SPHEREx_Catalog, Truth_Catalog= QC(Sources_to_Simulate_close)
# collate output secondary photometry tables
file_inter = './data/secondary_phot_id{}.parq'.format(k) # intermediate parquet file saving primary photometry
            # save secondary photometry
this = SPHEREx_Catalog['SOURCE_ID']==f"Central tID {int(tID_central)}"
SPsky.save_level3_secondary(SPHEREx_Catalog[this], 
                            Channels, 
                            SPHEREx_Instrument, 
                            file_inter, 
                            pointing_table=SPHEREx_Pointings.pointing_table, 
                            fluxerr_from_weights=True)
secondary_tbl = Table.read(file_inter, format="parquet")









