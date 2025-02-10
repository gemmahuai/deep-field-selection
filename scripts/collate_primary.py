# collect all primary photometry fits table and scale from 3 month to 2 year survey;
# write combined secondary photometry into a text file that can be passed to Photo-z. 
# gemmahuai 02/03/25

import numpy as np
import os
import itertools
from astropy.table import Table

import SPHEREx_ObsSimulator as SPobs
from SPHEREx_Simulator_Tools import SPHEREx_Logger, data_filename
import SPHEREx_InstrumentSimulator as SPinst
import SPHEREx_SkySimulator as SPsky
from spherex_parameters import load_spherex_parameters
from SPHEREx_SkySimulator import QuickCatalog, Catalog_to_Simulate

survey_plan_file = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data//survey_plan/spherex_survey_plan_R3_trunc3month.fits"
SPHEREx_Pointings = SPobs.Pointings(input_file = survey_plan_file,
                                   Gaussian_jitter=0., 
                                   roll_angle='psi2')
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




def write_output_to_photoz(flux, flux_err, source_id, ra, dec, filename, NewFile=False):
    
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




### find all primary photometry fits tables
# on TACC
directory = '/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/QCoutput/'
name = 'primary_phot' # files starting with 'primary' 
 
files = os.listdir(directory)
 
# sort files based on their id
filenames = [filename for filename in files if filename.startswith(name)]
files_sorted = sorted(filenames, key=lambda x: int(x.split('id')[-1].split('.fits')[0]))

### for each, calculate original secondary photometry, save flux errors

## output directory
secondary_dir = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/intermediate_files/"
ra_colname = 'ra_deep'
dec_colname = 'dec_deep'
output_filename = "/work2/09746/gemmah0521/frontera/sims/deepfield_sim/data/QCoutput/secondary_phot_combined_noAri.txt"



## Z-score calculation later
F_2yr = []
F_truth = []
Fe_scaled = []

count = 0
for f in range(len(files_sorted)):
    
    # QC primary photometry catalog,
    prim_cat = Table.read(directory + files_sorted[f])

    # compute secondary photometry, using internal filter sets
    SPsky.save_level3_secondary(prim_cat, 
                                Channels, 
                                SPHEREx_Instrument, 
                                secondary_dir+"secondary_inter.parq", 
                                pointing_table=SPHEREx_Pointings.pointing_table, 
                                fluxerr_from_weights=True)


    ## save truth secondary catalog
    prim_cat['FLUX'] = prim_cat['Flux'] # replace the measured primary photometry with true flux
    prim_cat['FLUX_ERR'] = np.ones_like(prim_cat['FLUX_ERR']) # constant flux err, doesn't matter here

    SPsky.save_level3_secondary(prim_cat, 
                                Channels, 
                                SPHEREx_Instrument, 
                                secondary_dir+"secondary_inter_truth.parq", 
                                pointing_table=SPHEREx_Pointings.pointing_table, 
                                fluxerr_from_weights=True)


    ## read in the secondary photometry
    sec_tbl = Table.read(secondary_dir+'secondary_inter.parq', format="parquet")

    ## read in the secondary truth photometry
    sec_truth = Table.read(secondary_dir+'secondary_inter_truth.parq', format="parquet")

    # save secondary (full sky / deep field)
    if ra_colname.endswith('_deep'):
        flux_colname = 'flux_deepfield'
        flux_err_colname = 'flux_err_deepfield'
    else:
        flux_colname = 'flux_allsky'
        flux_err_colname = 'flux_err_allsky'

    # original secondary error bars
    fe_og = sec_tbl[flux_err_colname][0] / 1000 # mJy
    # truth secondary photometry
    f_truth = sec_truth[flux_colname][0] / 1000 # mJy

    # well if there's singular matrix non-invertible, put nan everywhere...
    if False in np.isfinite(fe_og):
        ## skip this source!!
        fe_og = np.nan + np.zeros_like(fe_og)
        continue


    # delete the intermediate file
    try:
        os.remove(secondary_dir + 'secondary_inter.parq')
        os.remove(secondary_dir + 'secondary_inter_truth.parq')
    except FileNotFoundError:
        print("File " + secondary_dir + 'secondary_inter.parq ' +  "not found.")

    ## Modify error bars
    scaling = np.sqrt(24 / 3) # for the 3 month survey plan
    fe_scaled = fe_og / scaling

    ## Inject Gaussian noise into truth photometry
    noise = np.random.normal(0, 1, size=f_truth.shape) * fe_scaled
    f_2yr = f_truth + noise
    
    ## append to the z-score calculation
    F_2yr.append(f_2yr)
    F_truth.append(f_truth)
    Fe_scaled.append(fe_scaled)
    
    # ## plot scaled error bars
    # if plot is True:
    #     plt.figure(figsize=(5,4))
    #     plt.plot(wl, fe_og)
    #     plt.plot(wl, fe_scaled)
    #     plt.show()
    
    
    ## append this spectrum to an output file for photo-z run
    # need write_output_to_photoz(flux, flux_err, source_id, filename, NewFile=False)
    source_id = int(files_sorted[f].split("id")[-1].split(".fits")[0])
    ra = prim_cat['RA'][0]
    dec = prim_cat['DEC'][0]
    write_output_to_photoz(flux=f_2yr,
                           flux_err=fe_scaled,
                           source_id=source_id,
                           ra=ra,
                           dec=dec,
                           filename=output_filename,
                           NewFile=(count==0))


    count += 1
    

# F_2yr = np.array(F_2yr)
# F_truth = np.array(F_truth)
# Fe_scaled = np.array(Fe_scaled)


