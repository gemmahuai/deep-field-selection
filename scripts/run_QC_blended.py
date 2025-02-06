### Run QuickCatalog on individual source (GAMA) in parallel job, including nearby refcat sources

# randomly sample ~1k sources above a given cut;
# for each: run QC including nearby neighbors above the cut

# - Gemma Huai 01/27/2025
# local version

import numpy as np
import pandas as pd
import time
import multiprocessing as mp
import itertools
from astropy.table import hstack, Table
from astropy.io import fits
import astropy.units as u
from pyarrow import parquet
from astropy.coordinates import SkyCoord, Distance
import argparse

import os
import sys
import signal
from matplotlib import pyplot as plt

import SPHEREx_ObsSimulator as SPobs
from SPHEREx_Simulator_Tools import SPHEREx_Logger, data_filename
import SPHEREx_InstrumentSimulator as SPinst
import SPHEREx_SkySimulator as SPsky
from spherex_parameters import load_spherex_parameters
from SPHEREx_SkySimulator import QuickCatalog, Catalog_to_Simulate


# survey_plan_file = '/work/09746/gemmah0521/ls6/sims/source_selection/data/spherex_survey_plan_R2.fits' # on TACC
survey_plan_file = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/survey_plan/spherex_survey_plan_R3_trunc3month.fits" # on local machine
SPHEREx_Pointings = SPobs.Pointings(input_file = survey_plan_file,
                                   Gaussian_jitter=0., 
                                   roll_angle='psi2')

# Load instrument and project parameters as a dictionary
spherex_parameters = load_spherex_parameters()

# Scene = SPsky.Scene(SPHEREx_Pointings,
#                          zodi_model=SPsky.zodicalc.ModifiedKelsallModelWithHPFT())
#                    dgl_model =SPsky.DGLCalculator(Logger=Logger), Logger=Logger)
Scene = SPsky.Scene(SPHEREx_Pointings, 
                    zodi_model=SPsky.zodicalc.ModifiedKelsallModelWithHPFT())
                        # zodi_model=SPsky.zodicalc.SkyAveragedZodi(filepath='/Users/gemmahuai/Desktop/CalTech/SPHEREx/SPHEREx_2023/Codes/intensity_mapper_v30.csv'))

SPHEREx_Instrument = SPinst.Instrument(
    instrument_data=spherex_parameters,
    psf=data_filename("psf/simulated_PSF_database_centered_v3_og.fits"),
    psf_downsample_by_array={1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4},
    psf_trim_by_array={1: 32, 2: 32, 3: 32, 4: 32, 5: 32, 6: 32},
    noise_model=SPinst.white_noise,
    dark_current_model=SPinst.poisson_dark_current,
    lvf_model=SPinst.Tabular_Bandpass()
)

Channels = Table.read(data_filename('Channel_Definition_03022021.fits'))


### Set up some directories
primary_dir = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/blended_QC/"
file_inter = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/blended_QC/"

## try parsing arguments here?
parser = argparse.ArgumentParser(description="blending photometry with QC + Tractor")
parser.add_argument('-N', '-N_sources',  type=int, required=True, help="Number of sources to perform photometry")
parser.add_argument('-C', '-CPU', type=int, required=True, help="Maximum percentage of CPU to use")
parser.add_argument('-c', '-contour', type=float, required=True, choices=[0.2, 0.4, 0.6, 0.8], help="contour level used for the refcat selection")
parser.add_argument('-p', '-prim_combine', type=int, required=True, choices=[0, 1], help="combine primary photometry? 0 = do not combine; 1 = combine")
parser.add_argument('-o', '-output', type=str, required=True, help='full path to the output combined secondary photometry file with txt extension')
args = parser.parse_args()

N_sources = args.N
max_cpu_percent = args.C
contour = args.c
combine_prim_phot = args.p
output_filename = args.o



## input data files as global variables
COSMOS_tab = Table.read("/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/refcat_cuts/COSMOS2020_SPHEXrefcat_v0.6_166k_matched_Jean8k.csv")
idx_refcat = np.loadtxt(f"/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/refcat_cuts/boolean_cut_{contour}.txt", 
                        dtype=bool, skiprows=2)



# ------------------------------------------------------------------------------------
### All functions defined here

# write a single input spectrum to an output file that can directly go to photo-z
def write_output_to_photoz(flux, flux_err, source_id, filename, NewFile=False):
    
    """
    For a given source,
    flux, flux_err = 1D arrays of floats, ideally in spherex 102 fiducial bands, match Photoz input filters. [mJy]
    source_id = int, source ID as the first column
    filename = output filename
    NewFile = True/False: if True, create a new file; if False, append a line to the given file.
    ---------
    Returns: nothing
    """
    
    spectrum = list(itertools.chain(*zip(flux, flux_err))) # in mJy
    # insert ID, ra, dec
    spectrum.insert(0, 0.0) # fake RA, placeholder
    spectrum.insert(0, 0.0) # fake DEC, placeholder
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

def handler(signum, frame):
    ### register a handler for a timeout
    print("Terminate")
    raise Exception("End of time")

### function finding the fits file
def find_sed_fits_file(index, tractorID):

    """
    Given index of a source in the COSMOS catalog and its tractor ID, return the corresponding fits file containing the hires sed.
    """
    
    DIR_SED = "/work/09746/gemmah0521/ls6/sims/source_selection/data/COSMOS_ALL_HIRES_SEDS_LINES/" # on TACC
    # DIR_SED = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/COSMOS_ALL_HIRES_SEDS/" # on local machine
    if index < 60000:
        #print("0 - 60000")
        DIR = DIR_SED + "0_60000/"
    elif index < 120000:
        #print("60000 - 120000")
        DIR = DIR_SED + "60000_120000/"
    else:
        #print("120000 - ")
        DIR = DIR_SED + "120000_166041/"
    
    filename = DIR + f"cosmos2020_hiresSED_FarmerID_{tractorID:07d}.fits"
    return filename
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

## GAMA sources
def find_sed_fits_file_GAMA(uberID):
    DIR_SED = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/GAMA_ALL_HIRES_SEDS/"
    filename = DIR_SED + f"gama_hires_sed_uberID_{int(uberID)}.fits"
    return filename


def calc_source_separation(RA, DEC):
    """
    RA, DEC = numpy arrays of ra, dec
    
    calculates the distance to nearest neighbors for given sources
    
    """
    nearest_distances = []
    coords = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg, frame='icrs')

    # calculate the angular separation to find nearest neighbors
    for ii, coord in enumerate(coords):
        separations = coord.separation(coords)
        separations[ii] = np.inf

        # find the minimum separation, which is the nearest neighbor
        nearest_distance = np.min(separations).to("arcsec").value
        nearest_distances.append(nearest_distance)

    nearest_distances = np.array(nearest_distances)
    
    return nearest_distances



# --------------------------------------------------------------------------------
### Main parallel jobs 


def process_source(source_data):

    (i, ra_colname, dec_colname, 
     SPHEREx_Pointings, SPHEREx_Instrument, Scene, output_filename) = source_data


    ## select sources from Jean's 8k sources (match) that meet the given selection cut (idx_refcat)
    print(f"\n{i}")
    match = COSMOS_tab['match'] == 'True'
    cosmology = COSMOS_tab['COSMOLOGY'] == 1
    xmatch = COSMOS_tab['xmatched_LS_110k']==1

    ra = COSMOS_tab[ra_colname][idx_refcat & (~cosmology) & match][i]
    dec = COSMOS_tab[dec_colname][idx_refcat & (~cosmology) & match][i]

    tractorID = COSMOS_tab['Tractor_ID'][idx_refcat & (~cosmology) & match][i]
    source_ID = np.where(COSMOS_tab["Tractor_ID"] == tractorID)[0][0] # index among 166k

    # find input hires sed
    sed_path = find_sed_fits_file_corrected(source_ID, tractorID)

    print(f"    tractor ID for {i} = ", tractorID, " sed file ", sed_path)


    ## set up timer for timeout session
    signal.signal(signal.SIGALRM, handler)
    timeout_duration = 2000 # seconds
    signal.alarm(timeout_duration)


    try:
        # initialize QuickCatalog, only forced photometry, no Tractor
        QC = QuickCatalog(SPHEREx_Pointings, SPHEREx_Instrument, Scene, spectral_channel_table=Channels, \
                          Use_Tractor=True, \
                          subpixel_offset_x=0, subpixel_offset_y=0)
    
        Sources_to_Simulate = Catalog_to_Simulate()

        Sources_to_Simulate.load_single(name='COSMOS_{}'.format(tractorID),
                                              ra=ra*u.deg, 
                                              dec=dec*u.deg,
                                              inputpath=sed_path)
        
        ## add nearby photometered sources, from xmatched 110k catalog (< 4 SPHEREx pixels)
        size = 4 * 6.2 / 3600 # deg 
        cosmology = (COSMOS_tab['COSMOLOGY'] == 1)
        idx_close = np.where((COSMOS_tab[ra_colname][idx_refcat & xmatch] <= (ra+size)) &
                             (COSMOS_tab[ra_colname][idx_refcat & xmatch] >= (ra-size)) &
                             (COSMOS_tab[dec_colname][idx_refcat & xmatch] <= (dec+size)) &
                             (COSMOS_tab[dec_colname][idx_refcat & xmatch] >= (dec-size)))[0]
        print("   Number of nearby photometered sources  = ", len(idx_close))

        ## add into QC to photometer
        for idx in idx_close:

            if COSMOS_tab['Tractor_ID'][idx_refcat & xmatch][idx] == tractorID:
                # skip the primary central source itself
                continue

            ra_this = COSMOS_tab[ra_colname][idx_refcat & xmatch][idx]
            dec_this = COSMOS_tab[dec_colname][idx_refcat & xmatch][idx]
            tractorID_this = COSMOS_tab['Tractor_ID'][idx_refcat & xmatch][idx]
            source_ID_this = np.where(COSMOS_tab["Tractor_ID"] == tractorID_this)[0][0] # index among 166k

            sed_this = find_sed_fits_file_corrected(source_ID_this, tractorID_this)
            print("   tractor ID ", tractorID_this, "sed file ", sed_this)
            Sources_to_Simulate.load_single(name='COSMOS_{}'.format(tractorID_this),
                                            ra=ra_this*u.deg, 
                                            dec=dec_this*u.deg,
                                            inputpath=sed_this)
        # calculate distance to nearest neighbors
        sep = calc_source_separation(COSMOS_tab['ra'][idx_refcat & xmatch][idx_close], 
                                     COSMOS_tab['dec'][idx_refcat & xmatch][idx_close])
        print('smallest separation = ', np.nanmin(sep), ' arcsec')
        print('   ', sep)

        print("photometry with QC...")
        # photometry
        SPHEREx_Catalog, Truth_Catalog = QC(Sources_to_Simulate) 
        print("Done QC")

        id_primary = SPHEREx_Catalog['SOURCE_ID'] == f'COSMOS_{tractorID}'

        # save primary photometry
        SPsky.save_level3_primary(SPHEREx_Catalog[id_primary], primary_dir + f'primary_phot_id{tractorID}.parq')

        ## intermediate files, need to be removed
        # file_inter = "./" # on local machine
        SPsky.save_level3_secondary(SPHEREx_Catalog[id_primary], 
                                    Channels, 
                                    SPHEREx_Instrument, 
                                    file_inter+'secondary_phot_id{}.parq'.format(tractorID), 
                                    pointing_table=SPHEREx_Pointings.pointing_table, 
                                    fluxerr_from_weights=True)
        secondary_tbl = Table.read(file_inter+'secondary_phot_id{}.parq'.format(tractorID), format="parquet")
    
        f = secondary_tbl['flux_allsky'][0] / 1000 # mJy
        fe = secondary_tbl['flux_err_allsky'][0] / 1000 # mJy

        # delete the intermediate file
        try:
            os.remove(file_inter + 'secondary_phot_id{}.parq'.format(tractorID))
        except FileNotFoundError:
            print(f"File '{file_inter}' not found.")

    
        # write / append to the output txt file as photoz input
        if i==0:
            write_output_to_photoz(flux=f, 
                                   flux_err=fe, 
                                   source_id=tractorID, 
                                   filename=output_filename, 
                                   NewFile=True)
        else:
            write_output_to_photoz(flux=f, 
                                   flux_err=fe, 
                                   source_id=tractorID, 
                                   filename=output_filename, 
                                   NewFile=False)
        print(f"    Done {i}")
    
    except Exception as exc:
        # if timeout, return ID so that we could revisit the source later.
        print(exc)
        return tractorID

    signal.alarm(0) # disable alarm


    return None
        

    
def parallel_process(func, func_args, N_runs, max_cpu_percent):

    """
    Inputs:

    func: function,
        function to be parallelized.
    
    func_args: list,
        Arguments passed to the function.
    
    N_runs: int,
        Total number of runs of the given function.

    max_cpu_percent: int,
        Percentage of total CPU cores to be used.
    
    """
    
    total_cores = mp.cpu_count()
    max_workers = int(total_cores * max_cpu_percent / 100)
    max_workers = max(1, max_workers)  # Ensure at least one worker
    chunksize = 1
    batchsize = chunksize * max_workers # number of tasks to parallel at a time

    
    results_list = []
    
    for start in range(0, N_runs, batchsize):
        end = min(start+batchsize, N_runs)
        batch_source_args = func_args[start:end]
    
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(func, batch_source_args, chunksize=chunksize)
            pool.close()
            pool.join()

        results_list.extend(results)

    print(f"Simulation Done, failed IDs = {results_list}")


if __name__ == '__main__':
    Time_start = time.time()

    # # number of sources to be included from the 30k reference catalog
    # N_sources = int(sys.argv[1])
    # # Set maximum CPU usage
    # max_cpu_percent = int(sys.argv[2])
    # output_filename = str(sys.argv[3]) # txt extension
    # # starting index in the COSMOS 166k catalog
    # start_index = int(sys.argv[4])
    # combine_prim_phot = int(sys.argv[5]) # 0: not combine; 1: combine primary photometry

    # ----------------------- Inputs -------------------------------

    ra_colname = "ra"
    dec_colname = "dec"


    # ------------------- Start Simulation -------------------------

    # Down-select from Jean's 8k catalog + refcat randomly.
    match = COSMOS_tab['match'] == 'True'
    cosmology = (COSMOS_tab['COSMOLOGY'] == 1) # full sky cut
    # rand_ids = np.random.randint(low=0,
    #                              high=len(COSMOS_tab[match & (~cosmology) & idx_refcat]),
    #                              size=N_sources)
    ## follow the order, instead of random draws
    id_start = 1100
    rand_ids = np.arange(id_start, N_sources+id_start)

    print(len(idx_refcat), len(match))

    # ## plot and check refcat selection
    # def get_mag_from_flux(flux_Jy):
    #     mag = -2.5 * np.log10(flux_Jy / 3631)
    #     return mag

    # mag_w1 = get_mag_from_flux(COSMOS_tab['LS_W1'] / 1e6)
    # mag_z  = get_mag_from_flux(COSMOS_tab['LS_z']  / 1e6)

    # fig = plt.figure(figsize=(6,5))
    # plt.scatter(mag_z, mag_z - mag_w1, s=1, alpha=0.1)
    # plt.scatter(mag_z[match], mag_z[match] - mag_w1[match], s=1, color='black')
    # plt.scatter(mag_z[match & idx_refcat], mag_z[match & idx_refcat] - mag_w1[match & idx_refcat], s=1, color='red')
    # plt.scatter(mag_z[match & idx_refcat][rand_ids], mag_z[match & idx_refcat][rand_ids] - mag_w1[match & idx_refcat][rand_ids], s=10, color='green')
    # plt.show()

    # constrcut argument list passed to process per source function
    source_args = [(k, ra_colname, dec_colname,
                    SPHEREx_Pointings, SPHEREx_Instrument, Scene,
                    output_filename)
                    for k in rand_ids]

    # Run parallel processing
    results = parallel_process(func=process_source,
                               func_args=source_args,
                               N_runs=N_sources,
                               max_cpu_percent=max_cpu_percent)
    

    Time_end = time.time()
    Time_elapsed = Time_end - Time_start
    print(f"Total Runtime = {Time_elapsed:.3f} seconds.")

    if combine_prim_phot == 1:
        # combine all primary photometry parquet files into one
        directory = primary_dir
        name = 'primary_' # files starting with 'primary': primary_phot_id{source_tID}.parq

        files = os.listdir(directory)
        filenames = [filename for filename in files if filename.startswith(name)]

        files_parq = [file for file in filenames if file.endswith(".parq")]

        # sort files based on tractor ID
        files_parq_sorted = sorted(files_parq, key=lambda x: int(x.split('id')[-1].split('.parq')[0]))

        # combine the table
        output_combined_parq_name = output_filename.split(".txt")[0] + "_primary_combined.parq"
        remove = False # if remove is True, remove individual primary photometry parquet files
        tab_C = None # initialize the combined table
        for i, file in enumerate(files_parq_sorted):

            # read the primary photometry parquet table
            data = Table.read(directory+file)
            print(data['source_id'][0])
            data['source_id'][0] = int(file.split('id')[-1].split('.parq')[0])
            print(data['source_id'][0])

            if i==0:
                tab_C = data.copy()
            else:
                # if not the first source, append photometry to the first table.
                tab_C.add_row(data[0])

            # if remove is True, remove individual primary parquet files
            if remove is True:
                os.remove(directory + file)

        print("\nLength of the combined parq table = ", len(tab_C))

        # output
        tab_C.write(output_combined_parq_name, format='parquet', overwrite=True)
        print("\nDONE writing to a combined parquet file.")


