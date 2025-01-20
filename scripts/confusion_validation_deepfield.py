# Full photometric + redshift validation of spectral confusion injection in the deep field
# 1. Pick coordinates where roughly 100 observations will be taken per fiducial channel (Jean's datafile)
# 2. Pick a subset of sources from the COSMOS 32k refcat (~thousand), in a small area surrounding the coordinates
# 3. Controlled run: isolated photometry
# 4. Validation run: Tractor photometry only including sub-threshold sources
# 5. Run through Photo-z
### Gemma Huai, 01/17/2025
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
    noise_model=SPinst.white_noise,
    dark_current_model=SPinst.poisson_dark_current,
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

def write_output(source_id, N_nearsrcs, ra, dec, flux, flux_err, filename, NewFile=False):
    
    """
    In SPHEREx resolution (102 channels)
    flux and flux_err in mJy!!!
    """
    
    # write output .txt file
    spectrum = list(itertools.chain(*zip(flux, flux_err))) # in mJy
    # insert ID, ra, dec
    spectrum.insert(0, N_nearsrcs)
    spectrum.insert(0, dec)
    spectrum.insert(0, ra)
    spectrum.insert(0, int(source_id))
    
    if NewFile == False:
        # open the existing file and append a line of spectrum. 
        with open(filename, "a") as f:
            f.write("\n" + " ".join(map(str, spectrum)))
            
    else:
        # create a new file and write the first line
        with open(filename, "w") as f:
            f.write(" ".join(map(str, spectrum)))
    
    return 


## Load COSMOS reference catalogs
COSMOS_tab = Table.read('/Users/gemmahuai/Desktop/CalTech/SPHEREx/SPHEREx_2023/COSMOS2020_FARMER_R1_v2.1_p3_in_Richard_sim_2023Dec4.fits', format='fits')
# CUT_xmatch = np.loadtxt('../source_selection/final_cut_boolarray.txt', dtype=bool)
# COSMOS_tab = COSMOS_tab[CUT_xmatch]

# # idx_refcat = np.loadtxt("./newrefcat_bool_COSMOSzch1.txt", dtype=bool)
# idx_refcat = np.loadtxt("../source_selection/final_colormag_cut_boolarray_nov2024.txt", dtype=bool)
idx_refcat = np.loadtxt("/Users/gemmahuai/Desktop/CalTech/SPHEREx/source_selection/cosmos166k_posmatch_boolarray.txt", dtype=bool)



# -------------------------------- Photometry ----------------------------------

def photometer_single_src(args):

    k, COSMOS_tab, flag_sub, SPHEREx_Pointings, SPHEREx_Instrument, Scene, output_filename = args


    ## Isolated photometry
    tID_central = COSMOS_tab['Tractor_ID'][idx_refcat & flag_sub][k]
    print(f"\n{k}, tID = ", tID_central)
    ra = COSMOS_tab['ra_deep'][idx_refcat & flag_sub][k]
    dec = COSMOS_tab['dec_deep'][idx_refcat & flag_sub][k]
    # ra = ra_c
    # dec = dec_c

    # hires sed file of the primary source to fit
    central_sed_fn = find_sed_fits_file_corrected(COSMOS_tab['col1'][idx_refcat & flag_sub][k],
                                                  COSMOS_tab['Tractor_ID'][idx_refcat & flag_sub][k])
    print("filename = ", central_sed_fn)


    # start timing
    time_start = time.time()
    QC = QuickCatalog(SPHEREx_Pointings, SPHEREx_Instrument, Scene, Use_Tractor=False, spectral_channel_table=Channels,\
                    #do_not_fit=source_name,\
                    subpixel_offset_x=0, subpixel_offset_y=0)
    Sources_to_Simulate = Catalog_to_Simulate()
    # The central source to fit (refcat)
    Sources_to_Simulate.load_single(name=f"Central tID {int(tID_central)}", 
                                    ra=ra*u.deg, 
                                    dec=dec*u.deg, 
                                    inputpath=central_sed_fn)
    # run QuickCatalog
    SPHEREx_Catalog, Truth_Catalog= QC(Sources_to_Simulate, nmc=50)

    # collate output secondary photometry tables
    file_inter = '/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/secondary_phot_id{}_cntl'.format(k) + '.parq' # intermediate parquet file saving primary photometry
                # save secondary photometry
    this = SPHEREx_Catalog['SOURCE_ID']==f"Central tID {int(tID_central)}"
    SPsky.save_level3_secondary(SPHEREx_Catalog[this], 
                                Channels, 
                                SPHEREx_Instrument, 
                                file_inter, 
                                pointing_table=SPHEREx_Pointings.pointing_table, 
                                fluxerr_from_weights=True)
    secondary_tbl = Table.read(file_inter, format="parquet")
    time_end = time.time()
    print("\nTime elapsed = ", time_end - time_start)

    # ## save secondary photometry into photo-z input format
    # if k==0:
    #     print("create new file", output_filename+"_controlled.csv")
    #     write_output(source_id=tID_central,
    #                 N_nearsrcs=0,
    #                 ra = ra,
    #                 dec = dec,
    #                 flux=secondary_tbl[0]['flux_allsky']/1000, 
    #                 flux_err=secondary_tbl[0]['flux_err_allsky']/1000,
    #                 filename=output_filename+"_controlled.csv", 
    #                 NewFile=True)
    # else:
    #     print("write to an existing file", output_filename+"_controlled.csv")
    #     write_output(source_id=tID_central,
    #                 N_nearsrcs=0,
    #                 ra = ra,
    #                 dec = dec,
    #                 flux=secondary_tbl[0]['flux_allsky']/1000, 
    #                 flux_err=secondary_tbl[0]['flux_err_allsky']/1000,
    #                 filename=output_filename+"_controlled.csv", 
    #                 NewFile=False)



    ## Turn on Tractor and confusion (only from sub-threshold sources)
    # identify all background sub-threshold sources within 5 SPHEREx pixels
    size = 6.2 * 10 / 3600 # 10 pixel extent in deg
    ra_l = ra - size / 2
    ra_h = ra + size / 2
    dec_l = dec - size /2 
    dec_h = dec + size / 2
    want = (COSMOS_tab['ra_deep']<=ra_h) &\
        (COSMOS_tab['ra_deep']>=ra_l) &\
        (COSMOS_tab['dec_deep']<=dec_h) &\
        (COSMOS_tab['dec_deep']>=dec_l)
    # select all background sources within this area, do src_sub
    want_id = np.where(want)[0]

    # initialize arrays to hold true sub-threshold sources
    bg_id = np.array([], dtype=int)
    bg_tractor_id = np.array([], dtype=int)
    for id in want_id:
        if idx_refcat[id] == False:
            bg_id = np.append(bg_id, id) # pre-id (index) among 166k catalog
            bg_tractor_id = np.append(bg_tractor_id, COSMOS_tab['Tractor_ID'][id])
    print("   Number of nearby sub-threshold sources = ", len(bg_tractor_id)) 

    # create a list of background source name
    source_name = []
    for name in bg_tractor_id:
        source_name.append("COSMOS2020_" + "{}".format(name).zfill(7))

    # initiate QC with Tractor
    time_start = time.time()
    QC = QuickCatalog(SPHEREx_Pointings, SPHEREx_Instrument, Scene, Use_Tractor=True, spectral_channel_table=Channels,\
                    do_not_fit=source_name,\
                    subpixel_offset_x=0, subpixel_offset_y=0)
    Sources_to_Simulate_confusion = Catalog_to_Simulate()

    Sources_to_Simulate_confusion.load_single(name=f"Central tID {int(tID_central)}", 
                                            ra=ra*u.deg, 
                                            dec=dec*u.deg, 
                                            inputpath=central_sed_fn)

    ## add in nearby background sources sed
    for n in range(len(bg_tractor_id)):

        ra_sub = COSMOS_tab['ra_deep'][bg_id[n]]
        dec_sub = COSMOS_tab['dec_deep'][bg_id[n]]
        # print("\n")
        # print(bg_tractor_id[n], idx_refcat[bg_id[n]], '  ra=', ra_sub, '  dec=', dec_sub)

        filename = find_sed_fits_file_corrected(index=bg_id[n],
                                                tractorID=bg_tractor_id[n])
        # print(filename)

        Sources_to_Simulate_confusion.load_single(name=source_name[n],
                                                ra=ra_sub*u.deg,
                                                dec=dec_sub*u.deg,
                                                inputpath=filename)

    # run QuickCatalog
    SPHEREx_Catalog, Truth_Catalog= QC(Sources_to_Simulate_confusion, nmc=50)
    time_end = time.time()
    print("\nTime elapsed = ", time_end - time_start)
    # collate output secondary photometry tables
    file_inter = '/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/secondary_phot_id{}.parq'.format(k) # intermediate parquet file saving primary photometry
                # save secondary photometry
    this = SPHEREx_Catalog['SOURCE_ID']==f"Central tID {int(tID_central)}"
    SPsky.save_level3_secondary(SPHEREx_Catalog[this], 
                                Channels, 
                                SPHEREx_Instrument, 
                                file_inter, 
                                pointing_table=SPHEREx_Pointings.pointing_table, 
                                fluxerr_from_weights=True)
    secondary_tbl_confusion = Table.read(file_inter, format="parquet")


    # ## save secondary photometry into photo-z input format
    # if k==0:
    #     print("create new file", output_filename+".csv")
    #     write_output(source_id=tID_central,
    #                 N_nearsrcs=len(bg_id),
    #                 ra = ra,
    #                 dec = dec,
    #                 flux=secondary_tbl_confusion[0]['flux_allsky']/1000, 
    #                 flux_err=secondary_tbl_confusion[0]['flux_err_allsky']/1000,
    #                 filename=output_filename+".csv", 
    #                 NewFile=True)
    # else:
    #     print("write to an existing file", output_filename+".csv")
    #     write_output(source_id=tID_central,
    #                 N_nearsrcs=len(bg_id),
    #                 ra = ra,
    #                 dec = dec,
    #                 flux=secondary_tbl_confusion[0]['flux_allsky']/1000, 
    #                 flux_err=secondary_tbl_confusion[0]['flux_err_allsky']/1000,
    #                 filename=output_filename+".csv", 
    #                 NewFile=False)

    # ## plot to double check photometry
    # hires_sed = Table.read(central_sed_fn, format='fits')
    # fig = plt.figure(figsize=(6,5))
    # plt.plot(hires_sed['lambda'], hires_sed['FLUX'], color='red', lw=3, alpha=0.9, label='Input SED')
    # # plt.errorbar(SPHEREx_Catalog[this]['WAVELENGTH'], SPHEREx_Catalog[this]['FLUX'], SPHEREx_Catalog[this]['FLUX_ERR'], 
    # #              fmt='o', ms=2, alpha=0.1, label='primary')
    # plt.errorbar(secondary_tbl[0]['lambda'], secondary_tbl[0]['flux_allsky']/1000, yerr=secondary_tbl[0]['flux_err_allsky']/1000,
    #             fmt='o', ms=5, color='blue', label='secondary, confusion OFF')
    # plt.errorbar(secondary_tbl_confusion[0]['lambda'], secondary_tbl_confusion[0]['flux_allsky']/1000, 
    #             yerr=secondary_tbl_confusion[0]['flux_err_allsky']/1000,
    #             fmt='o', ms=5, color='green', label='secondary, confusion ON')
    # plt.ylim(0, hires_sed['FLUX'].max()*2)
    # plt.title(f"Tractor ID = {tID_central}", fontsize=15)
    # plt.xlabel("Wavelength (um)", fontsize=15)
    # plt.ylabel("Flux density (mJy)", fontsize=15)
    # plt.legend(fontsize=12, loc='upper right')
    # plt.show()

    print(f"Simulation {k} Done!")


## main parallel function
def parallel_process(func, func_args, max_cpu_percent):
    # For the first N_sources sources
      
    total_cores = mp.cpu_count()
    max_workers = int(total_cores * max_cpu_percent / 100)
    max_workers = max(1, max_workers)  # Ensure at least one worker
    chunksize = 1
    batchsize = chunksize * max_workers # number of tasks to parallel at a time

    
    results_list = []
    
    for start in range(0, N_patches, batchsize):
        end = min(start+batchsize, N_patches)
        batch_source_args = func_args[start:end]
    
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(func, batch_source_args, chunksize=chunksize)
            pool.close()
            pool.join()

        results_list.extend(results)

    print(f"Simulation Done, failed IDs = {results_list}")



# ------------------------------- RUN -----------------------------

if __name__ == '__main__':

    ## parsing command line arguments
    parser = argparse.ArgumentParser(description="Perform confusion validation on the photometric level, running QuickCatalog and Tractor.")
    parser.add_argument('-N', type=int, required=True, help='Number of refcat sources to photometer.')
    parser.add_argument('-c', type=int, required=True, help='CPU percentage to use, out of 14 cores.')
    parser.add_argument('-o', type=str, required=True, help='output files name, will append file extension name')
    args = parser.parse_args()

    N_patches = args.N 
    max_cpu_percent = args.c 
    output_filename = args.o 

    output_filename = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/secondary_combined_to_photoz"

    

    ## Move COSMOS to the deep field
    file = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/source_selection/data/obs_per_ecliptic_lat_R3.fits"
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


    print("\nStart Parallel Processes...")
    ## Run parallel processing
    Time_start = time.time()
    source_args = [(k+200, COSMOS_tab, flag_sub, SPHEREx_Pointings, SPHEREx_Instrument, Scene, output_filename) 
                   for k in range(N_patches)]
    parallel_process(func=photometer_single_src,
                     func_args=source_args, 
                     max_cpu_percent=max_cpu_percent)

    Time_end = time.time()
    Time_elapsed = Time_end - Time_start
    print(f"Total Runtime = {Time_elapsed:.3f} seconds.")








