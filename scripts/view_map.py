### Plot a given sky map provided by Shuang-Shuang Chen

# Usage: python view_map.py --h

# Gemma Huai, 12/13/2024


import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS


parser = argparse.ArgumentParser(description="Plot sky maps from Shuang-Shuang")
parser.add_argument('--m', type=int, required=True, help='which map to plot: 0 = galaxies; 1 = zodi; 2 = DC; 3 = read noise.')
parser.add_argument('--z', type=int, required=True, help='zoom in the central 1 sq deg around the NEP or not, 0 = no; 1 = yes')
args = parser.parse_args()

which_map = args.m 
zoom = args.z

## load the map
DIR_MAP = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/maps/schen6/"
match which_map:
    case 0:
        fits_file = DIR_MAP + "deep_north_galaxies.fits"
        label = "SPHEREx Deep Field Map for Galaxies"
    case 1:
        fits_file = DIR_MAP + "deep_north_zodi.fits"
        label = "SPHEREx Deep Field Map for Zodiacal Light"
    case 2:
        fits_file = DIR_MAP + "deep_north_dc.fits"
        label = "SPHEREx Deep Field Map for Dark Current"
    case 3:
        fits_file = DIR_MAP + "deep_north_read_noise.fits"
        label = "SPHEREx Deep Field Map for Read Noise"

with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

# extract wcs info
wcs = WCS(header)

# plot
fig = plt.figure(figsize=(8, 10))

ax1 = fig.add_subplot(111, projection=wcs) 
ax1.imshow(data, aspect='auto', origin="lower", cmap="viridis")
cbar = plt.colorbar(ax1.images[0], ax=ax1, location='bottom', pad=0.1)
cbar.set_label('Surface Brightness [MJy / Sr]', fontsize=15)

## plot a zoomed-in map
if zoom == 1:
    if (header['NAXIS1'] * header['CDELT1'])>1:
        print("plot the 1 deg^2 zoomed-in region...")
        # inset axes....
        # 1 square deg surrounding NEP
        size = 0.5 # deg
        x1 = int(header['CRPIX1'] - size/header['CDELT1'])
        x2 = int(header['CRPIX1'] + size/header['CDELT1'])
        y1 = int(header['CRPIX2'] - size/header['CDELT2'])
        y2 = int(header['CRPIX2'] + size/header['CDELT2'])
        dt = int((x2-x1)/2) # ticking interval

        data_zoom = np.zeros_like(data) + np.nanmedian(data)
        data_zoom[y1:y2, x1:x2] = data[y1:y2, x1:x2].copy()

        ticks = np.arange(x1, x2+dt, dt)
        ticks_label = [f'{i}' for i in ticks]
        axins = ax1.inset_axes(
            [0.7, 0.7, 0.45, 0.45],
            xlim=(x1, x2), ylim=(y1, y2), projection=wcs)
        im = axins.imshow(data_zoom,  origin="lower", cmap="viridis")

        divider = make_axes_locatable(axins)
        cax = divider.new_vertical(size = "5%",
                                pad = 0.2,
                                pack_start = True)

        fig.add_axes(cax, ticks = mpl.ticker.FixedLocator([]))
        cbar1 = fig.colorbar(im, cax = cax, orientation = "horizontal", shrink=0.75)

        axins.set_title(r'1 $\text{deg}^2$ NEP', color='blue')
        axins.coords.grid(True, color='white', ls='dotted')
        ticks = np.arange(x1, x2+dt, dt)
        ticks_label = [f'{i}' for i in ticks]
        axins.coords[0].set_ticks(number=3)
        axins.coords[1].set_ticks(number=3)
        axins.coords[0].set_axislabel(" ")
        axins.coords[1].set_axislabel(" ")

        ax1.indicate_inset_zoom(axins, edgecolor="blue")

    else:
        print("The area of the input map is smaller than 1 deg^2; cannot zoom in further!")
        # raise ValueError


ax1.coords.grid(True, color="white", ls="dotted")
ax1.coords[0].set_axislabel(f"Longitude (RA)", fontsize=15)
ax1.coords[1].set_axislabel(f"Latitude (DEC)", fontsize=15)
ax1.set_title(label, fontsize=17)

plt.show()
