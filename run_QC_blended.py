### Run QuickCatalog on individual source in parallel job, including nearby refcat sources

# randomly sample ~1k sources above a given cut;
# for each: run QC including nearby neighbors above the cut

# - Gemma Huai 01/29/2025
# TACC version

import numpy as np
import pandas as pd
import time
import multiprocessing as mp
import itertools
from astropy.table import hstack, Table, vstack
from astropy.io import fits
import astropy.units as u
from pyarrow import parquet
import argparse
import pyarrow as pa
import h5py

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

# survey_plan_file = data_filename('spherex_survey_plan_R2.fits')
survey_plan_file = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/survey_plan/spherex_survey_plan_R3_trunc3month.fits"
SPHEREx_Pointings = SPobs.Pointings(input_file = survey_plan_file,
                                   Gaussian_jitter=0., 
                                   roll_angle='psi2')

# Load instrument and project parameters as a dictionary
spherex_parameters = load_spherex_parameters()

Scene = SPsky.Scene(SPHEREx_Pointings, 
                    zodi_model=SPsky.zodicalc.ModifiedKelsallModelWithHPFT())

SPHEREx_Instrument = SPinst.Instrument(
    instrument_data=spherex_parameters,
    psf=data_filename("psf/simulated_PSF_database_centered_v3.fits"),
    psf_downsample_by_array={1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4},
    psf_trim_by_array={1: 32, 2: 32, 3: 32, 4: 32, 5: 32, 6: 32},
    noise_model=SPinst.white_noise,
    dark_current_model=SPinst.poisson_dark_current,
    lvf_model=SPinst.Tabular_Bandpass()
)

Channels = Table.read(data_filename('Channel_Definition_03022021.fits'))


### Set up some directories for outputs
# primary_dir = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/blended_QC/"
# file_inter = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/blended_QC/"
primary_dir = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/QCoutput/"
file_inter = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/QCoutput/"

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


# function finding the new, corrected fits file
def find_sed_fits_file_corrected(index, tractorID):
    if index < 60000:
        #print("0 - 60000")
        DIR = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/COSMOS_ALL_HIRES_SEDS_010925/0_60000/"
    elif index < 120000:
        #print("60000 - 120000")
        DIR = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/COSMOS_ALL_HIRES_SEDS_010925/60000_120000/"
    else:
        #print("120000 - ")
        DIR = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/COSMOS_ALL_HIRES_SEDS_010925/120000_166041/"
    filename = DIR + f"cosmos2020_hiresSED_FarmerID_{tractorID:07d}_corrected.fits"
    return filename

# --------------------------------------------------------------------------------
### Main parallel jobs 


def process_source(source_data):

    (i, COSMOS_tab, idx_refcat, ra_colname, dec_colname, 
     SPHEREx_Pointings, SPHEREx_Instrument, Scene, output_filename) = source_data


    ## select sources from Jean's 8k sources (match) that:
    # meet the given deep field selection cut (idx_refcat) &
    # below the cosmology full-sky cut

    print(f"\n{i}")
    match = COSMOS_tab['match'] == 'True'
    cosmology = COSMOS_tab['COSMOLOGY'] == 1
    xmatch = COSMOS_tab['xmatched_LS_110k'] == 1

    ra = COSMOS_tab[ra_colname][idx_refcat & (~cosmology) & match][i]
    dec = COSMOS_tab[dec_colname][idx_refcat & (~cosmology) & match][i]

    tractorID = COSMOS_tab['Tractor_ID'][idx_refcat & (~cosmology) & match][i]
    source_ID = np.where(COSMOS_tab["Tractor_ID"] == tractorID)[0][0] # index among 166k

    # find input hires sed
    sed_path = find_sed_fits_file_corrected(source_ID, tractorID)

    print(f"    tractor ID for {i} = ", tractorID, " sed file ", sed_path)


    ## set up timer for timeout session
    signal.signal(signal.SIGALRM, handler)
    timeout_duration = 3550 # seconds timeout
    signal.alarm(timeout_duration)


    try:
        
        # timing for this source
        time_start = time.time()
        # print("Start QC")
        # initialize QuickCatalog, only forced photometry, no Tractor
        QC = QuickCatalog(SPHEREx_Pointings, SPHEREx_Instrument, Scene, spectral_channel_table=Channels, \
                          Use_Tractor=True, \
                          subpixel_offset_x=0, subpixel_offset_y=0)
    
        Sources_to_Simulate = Catalog_to_Simulate()
        # print("Start loading single source")
        Sources_to_Simulate.load_single(name='COSMOS_{}'.format(tractorID),
                                        ra=ra*u.deg, 
                                        dec=dec*u.deg,
                                        inputpath=sed_path, 
                                        no_phot='yes')
        
        ## add nearby photometered sources (< 4 SPHEREx pixels) that:
        # are from the cross-matched 110k catalog (xmatch)
        # are above the given deep field cut (idx_refcat)
        size = 4 * 6.2 / 3600 # deg 
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
            # print(f"   for {tractorID}, neighbor tractor ID ", tractorID_this, "sed file ", sed_this)
            Sources_to_Simulate.load_single(name='COSMOS_{}'.format(tractorID_this),
                                            ra=ra_this*u.deg, 
                                            dec=dec_this*u.deg,
                                            inputpath=sed_this)
            
        ## photometry
        SPHEREx_Catalog, Truth_Catalog = QC(Sources_to_Simulate) 

        id_primary = SPHEREx_Catalog['SOURCE_ID'] == f'COSMOS_{tractorID}'

        # save primary photometry with the truth table
        SPHEREx_Catalog.add_column(Truth_Catalog['Zodi'], name='Zodi')
        SPHEREx_Catalog.add_column(Truth_Catalog['Flux'], name='Flux')

      
        ## Directly save SPHEREx Catalog table with Truth Catalog... for later secondary photometry calc
        SPHEREx_Catalog[id_primary].write(primary_dir + f'primary_phot_id{tractorID}.fits', format='fits', overwrite=True)
        # print("Done writing to fits file")

        # end timing
        time_end = time.time()
        time_elapsed = time_end - time_start
        print(f"    Done {i}, time elapsed = {time_elapsed:.3f}")

    
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
    
    total_cores = int(os.environ.get("SLURM_NTASKS", mp.cpu_count())) 
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

    mp.set_start_method("forkserver")

    ## set up parser
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


    # ----------------------- Inputs -------------------------------

    COSMOS_tab = Table.read("/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/refcat_cuts/COSMOS2020_SPHEXrefcat_v0.6_166k_matched_Jean8k.csv")
    ra_colname = "ra_deep"
    dec_colname = "dec_deep"

    idx_refcat = np.loadtxt(f"/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/refcat_cuts/boolean_cut_{contour}.txt", dtype=bool, skiprows=2)


    # ------------------- Start Simulation -------------------------

    # Down-select from Jean's 8k catalog + refcat randomly, below the full-sky cut, above the chosen deep field contour cut
    match = COSMOS_tab['match'] == 'True'
    cosmology = COSMOS_tab['COSMOLOGY'] == 1
    indices = np.arange(len(COSMOS_tab[match & (~cosmology) & idx_refcat]))
    # randomly select N_sources from the pool
    # rand_ids = np.random.choice(indices, size=N_sources, replace=False)
    # given start index, select N sources in order, from match & (~cosmology) & idx_refcat
    id_start = 0
    rand_ids = np.arange(id_start, N_sources+id_start)

    print(len(idx_refcat), len(match))


    # constrcut argument list passed to process per source function
    source_args = [(k, COSMOS_tab, idx_refcat, ra_colname, dec_colname,
                    SPHEREx_Pointings, SPHEREx_Instrument, Scene,
                    output_filename)
                    for k in rand_ids]

    # start timing
    Time_start = time.time()

    # Run parallel processing
    results = parallel_process(func=process_source,
                               func_args=source_args,
                               N_runs=N_sources,
                               max_cpu_percent=max_cpu_percent)
    

    Time_end = time.time()
    Time_elapsed = Time_end - Time_start
    print(f"Total Runtime = {Time_elapsed:.3f} seconds.")

