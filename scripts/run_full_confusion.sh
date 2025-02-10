#!/bin/bash

# combines run_QC_blended.py, confusion_lib_run.py, and confusion_injection.py 
# @gemmahuai 01/28/25

### Parse arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <n_conf> <cpu_percent> <cut_value> <conf_lib_output> <injected_phot_output>"
    exit 1
fi
# number of confusion spectra to generate
n_conf=$1
# percentage of cpu to use
cpu=$2
# which color magnitude contour cut {0.2, 0.4, 0.6, 0.8}
cut=$3
# filename of the output confusion library
conf_lib_fn=$4
# filename of the output injected photometry file
output_phot=$5
# # Output secondary photometry file with confusion injection, will be passed to Photo-z
# output_phot=$DIR+"secondary_combined_inject.txt"


### Inputs
# Pull the combined secondary photometry file from TACC, with scaled error bars, but no injection yet
DIR="/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/deep-field-phot-on-maps/data/secondary/"
input_phot="${DIR}/secondary_phot_combined_contour${cut}_noAri.txt"
# Noiseless 102 band SED
Catgrid="/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/makegrid_photometry/makegrid_output/cosmos2020_166k_catgrid_102spherex_corrected.out"
# Color-magnitude cuts
Cut="/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/refcat_cuts/boolean_cut_${cut}.txt"

## Initialize the environment
echo "Activating conda environment 'spherexsim_tractor'..."
conda activate spherexsim_tractor

## change to the relevant directory
cd /Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/deep-field-phot-on-maps/scripts/



# ## Generate confusion library
# echo "Generate confusion library ..."
# python confusion_lib_run.py -N "$n_conf" -c "$cpu" -i "$Cut" -o "$conf_lib_fn"


## Inject confusion lib into QC output secondary photometry
cd /Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/
echo "Confusion injection..."
python confusion_injection.py -i "$input_phot" -l "$conf_lib_fn" -s "$Catgrid" -o "$output_phot"




