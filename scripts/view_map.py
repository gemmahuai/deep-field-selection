import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
from astropy import constants as const
import pandas as pd
from astropy.table import Column
from astropy.time import Time
from scipy import interpolate as interp
import time
# from Photoz import photoz_tools as phtz
from tractor import *
from skimage.transform import downscale_local_mean
import scipy.optimize as opt
import matplotlib as mpl
from scipy.spatial.distance import cdist

rom astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import BarycentricMeanEcliptic, BarycentricTrueEcliptic
import healpy

import itertools
import matplotlib.pyplot as plt
from astropy.table import hstack
import os

import SPHEREx_ObsSimulator as SPobs
from SPHEREx_Simulator_Tools import SPHEREx_Logger, data_filename
import SPHEREx_InstrumentSimulator as SPinst
import SPHEREx_SkySimulator as SPsky
from pkg_resources import resource_filename

# survey_plan_file = 'spherex_survey_plan_march_2021.fits'
survey_plan_file = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/spherex_survey_plan_R2.fits"
SPHEREx_Pointings = SPobs.Pointings(input_file = survey_plan_file,
                                   Gaussian_jitter=1.8, 
                                   roll_angle='psi2')

from spherex_parameters import load_spherex_parameters
# Load instrument and project parameters as a dictionary
spherex_parameters = load_spherex_parameters()


ds1 = 4
ds2 = 2
trim = 32
SPHEREx_Instrument = SPinst.Instrument(
    instrument_data=spherex_parameters,
    psf=data_filename("psf/simulated_PSF_database_centered_v3_og.fits"),
    psf_downsample_by_array={1: ds1, 2: ds1, 3: ds1, 4: ds2, 5: ds2, 6: ds2},
    psf_trim_by_array={1: trim, 2: trim, 3: trim, 4: trim, 5: trim, 6: trim},

    noise_model=SPinst.white_noise,
    dark_current_model=SPinst.poisson_dark_current,
    lvf_model=SPinst.Tabular_Bandpass()
)

from SPHEREx_SkySimulator import QuickCatalog
from SPHEREx_SkySimulator import Catalog_to_Simulate
# path='/Users/zhaoyuhuai/SPHEREx-Sky-Simulator/docs/QuickCatalog/'

from pyarrow import parquet
from astropy.table import Table
from astropy.io import fits
Channels = Table.read(data_filename('Channel_Definition_03022021.fits'))
Scene = SPsky.Scene(SPHEREx_Pointings,
                        zodi_model=SPsky.zodicalc.ModifiedKelsallModelWithHPFT())

COSMOS_tab = Table.read('/Users/gemmahuai/Desktop/CalTech/SPHEREx/SPHEREx_2023/COSMOS2020_FARMER_R1_v2.1_p3_in_Richard_sim_2023Dec4.fits', format='fits')
COSMOS_sim_sources = Table.read('/Users/gemmahuai/Desktop/CalTech/SPHEREx/SPHEREx_2023/COSMOS2020_FARMER_R1_v2.1_p3_in_Richard_sim_2023Dec4.fits', format='fits')


fits_file = "./maps/schen6/deep_north_dc.fits" 
with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

# extract wcs info
wcs = WCS(header)

# Step 3: Define the target coordinate frame
# Uncomment one of these based on the desired coordinate system
# target_frame = Galactic()  # For Galactic coordinates
target_frame = BarycentricTrueEcliptic()  # For Ecliptic coordinates

# Step 4: Plot the data in the desired coordinates
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection=wcs)  # Use projection=wcs directly here
ax.imshow(data, origin="lower", cmap="viridis")

# Step 5: Transform WCS to the target frame
ax.coords.grid(True, color="white", ls="dotted")
ax.coords[0].set_axislabel(f"Longitude ({target_frame.__class__.__name__})", fontsize=15)
ax.coords[1].set_axislabel(f"Latitude ({target_frame.__class__.__name__})", fontsize=15)
ax.set_title("SPHEREx Deep Field Region in Ecliptic Coordinates")

# Show the plot
cbar = plt.colorbar(ax.images[0], ax=ax)
cbar.set_label('Surface Brightness [MJy / Sr]', fontsize=15)
plt.show()


