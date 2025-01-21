"""

generate confusion library from sub-threshold sources, from tools in confusion_lib_build.py

@author: gemmahuai
01/21/2025

"""

import numpy as np
from astropy.table import Table
import multiprocessing as mp
import time 
import argparse
from confusion_lib_build import ConfusionLibrary

import SPHEREx_ObsSimulator as SPobs
from SPHEREx_Simulator_Tools import data_filename
import SPHEREx_InstrumentSimulator as SPinst
from spherex_parameters import load_spherex_parameters
import SPHEREx_SkySimulator as SPsky


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

# ---------------------------------------------------------------------
### input catalogs and refcat cut
COSMOS_tab = Table.read('/Users/gemmahuai/Desktop/CalTech/SPHEREx/SPHEREx_2023/COSMOS2020_FARMER_R1_v2.1_p3_in_Richard_sim_2023Dec4.fits', format='fits')
idx_refcat = np.loadtxt("/Users/gemmahuai/Desktop/CalTech/SPHEREx/source_selection/cosmos166k_posmatch_boolarray.txt", dtype=bool)

# define channel bins
wl = Channels['lambda_min'] + (Channels['lambda_max']-Channels['lambda_max'])/2

# output data dir
DIR = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/deep-field-phot-on-maps/data/"


# # ---------------------------------------------------------------------
# ### test confusion_lib_build.py:

# # instantiate ConfusionLibrary
# confusionlib = ConfusionLibrary(Pointings=SPHEREx_Pointings,
#                                 Instrument=SPHEREx_Instrument,
#                                 Scene=Scene,
#                                 Channels=Channels,
#                                 catalog=COSMOS_tab,
#                                 ra_colname="ALPHA_J2000",
#                                 dec_colname="DELTA_J2000",
#                                 id_colname="Tractor_ID",
#                                 idx_refcat=idx_refcat)

# # generate a zero SED
# confusionlib()

# # do photometry
# index = 0
# confusionlib.photometer_one_spot(index=index,
#                                  Npix=10,
#                                  Use_Tractor=True,
#                                  )

# # save primary photometry
# confusionlib.collate_QuickCatalog_primary(output_filename="../data/test_confusion_lib_primary.parq")

# # save secondary photometry
# confusionlib.collate_QuickCatalog_secondary(output_filename="../data/test_confusion_lib_secondary_combined.csv",
#                                             file_intermediate=f"../data/test_confusion_lib_secondary_{index}.parq",
#                                             NewFile=True)

# # plot confusion library stats, using an old library
# lib_path = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/data/QCdata_N2000_Pix10_updatedJan2025_new32krefcat.txt"
# (std, mean, hpdi) = confusionlib.calc_lib_variation(path_lib=lib_path,
#                                                     hpdi_bins=50)
# # fig = plt.figure(figsize=(5,4))
# # plt.plot(wl, std)
# # plt.plot(wl, mean)
# # plt.plot(wl, hpdi)
# # plt.show()

# # testing plot method in the class.
# confusionlib.plot_lib(hpdi=hpdi,
#                       mean=mean,
#                       std=std,
#                       path_lib=lib_path,
#                       output_pdf="../data/test_plot.pdf")


# # print all attribute names
# print(list(vars(confusionlib).keys()))



## ---------------------------------------------------------------------
### Set up parallel process

def run_single_job(args):

    (index, confusionlib, output_filename) = args

    print(f"\nIndex = {index}...")

    try:
        confusionlib.photometer_one_spot(index=index,
                                         Npix=10,
                                         Use_Tractor=True,
                                         )
        
        # save secondary photometry into a combined file, 
        # discarding all individual intermediate secondary photometry file
        file_intermediate = DIR + f"secondary_{index}_del.parq"
        confusionlib.collate_QuickCatalog_secondary(output_filename=DIR+output_filename,
                                                    file_intermediate=file_intermediate,
                                                    NewFile=(index==0))
        
        print(f"     [{index}] Done!")
        return None
    
    except:
        print(f"     Photometry failed, index = {index}.")
        return index

    

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

    ## set up parser
    parser = argparse.ArgumentParser(description="Generating confusion library from sub-threshold sources, given catalog and seleciton cut.")
    parser.add_argument('-N', type=int, required=True, help='Number of confusion spectra to generate.')
    parser.add_argument('-c', type=int, required=True, help='CPU percentage to use, out of 14 cores.')
    parser.add_argument('-o', type=str, required=True, help='output files name, will append file extension name')
    args = parser.parse_args()

    N_srcs = args.N 
    max_cpu_percent = args.c 
    output_filename = args.o 


    # instantiate ConfusionLibrary
    confusionlib = ConfusionLibrary(Pointings=SPHEREx_Pointings,
                                    Instrument=SPHEREx_Instrument,
                                    Scene=Scene,
                                    Channels=Channels,
                                    catalog=COSMOS_tab,
                                    ra_colname="ALPHA_J2000",
                                    dec_colname="DELTA_J2000",
                                    id_colname="Tractor_ID",
                                    idx_refcat=idx_refcat)
    
    # generate a zero SED
    confusionlib()

    # set up parallel job arguments
    # TODO: fill in more arguments...
    source_args = [(index, confusionlib, output_filename)
                   for index in range(N_srcs)]

    parallel_process(func=run_single_job,
                     func_args=source_args,
                     N_runs=N_srcs,
                     max_cpu_percent=max_cpu_percent)








