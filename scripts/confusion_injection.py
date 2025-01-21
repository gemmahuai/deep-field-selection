import numpy as np
from deep_field_utils import write_output_to_photoz, Gaussian
from matplotlib import pyplot as plt
from astropy.table import Table

## load in SEDs and confusion library
DIR = "./sphx_newrefcat_cosmos_32k/"


# sed_noiseless = np.loadtxt(DIR+"Catgrid_phot_cosmos_withlines_newzw1refcat_32k.txt")
# sed_noisy = np.loadtxt(DIR+"NoisySphx_shallow_withlines_newrefcat_32k_catgrid.txt")

sed_noiseless = np.loadtxt("/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/makegrid_photometry/makegrid_output/cosmos2020_166k_catgrid_102spherex.out")
sed_noisy = np.loadtxt("/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/secondary_combined_redshift_valid1k_deepfield_cntl.txt")
lib = np.loadtxt("./data/QCdata_N2000_Pix10_updatedDec2024.txt")

# cosmos catalog
COSMOS_tab = Table.read('/Users/gemmahuai/Desktop/CalTech/SPHEREx/SPHEREx_2023/COSMOS2020_FARMER_R1_v2.1_p3_in_Richard_sim_2023Dec4.fits', format='fits')

## output filename
# filename = DIR+"Confusion_shallow/Confusion_noisy_cosmos32krefcat_randomlib_catgrid.txt"
filename = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/deep_field/data/secondary_combined_redshift_valid1k_deepfield_confinj.txt"


def add_one_confusion_to_phot_comb_noise(galaxy_spec, confusion_lib, confusion_mean, filename, Newfile, confusion_std=None, combined_noise=False):
    """
    Inputs:
    galaxy_spec = 1d array: source_id, ra, dec, flux1, fluxerr1, flux2, fluxerr2, ..., flux102, fluxerr102 (mJy)
    confusion_lib = all confusion spectra (shape = (N_spec, 208)) (mJy)
    confusion_mean = average confusion spectrum, shape = (1, 102) (mJy)
    confusion_std = std of confusion library, shape = (1, 102) (mJy)
    filename = output filename
    Newfile = if or not create a new file
    combined_noise = False: if True, combining photometry error, individual confusion noise and averaged confusion sigma
    ----------
    If confusion_std is not given, combine noise from each individual confusion realization
    If confusion_std is given, use that as confusion uncertainty.
    
    """
    
    flux = galaxy_spec[3:][::2] # mJy
    fluxerr = galaxy_spec[4:][::2] # mJy
    source_id = galaxy_spec[0]
    
    # draw a random confusion spectrum from the library 
    idx = np.random.randint(low=0, high=len(confusion_lib))
    flux_c = confusion_lib[idx][4:][::2]
    flux_c_err = confusion_lib[idx][5:][::2]
    
    total_flux = flux + flux_c - confusion_mean
    
    if confusion_std is not None:
        
        if combined_noise is False:
            # print("library std")
            # combine error bars using confusion_std
            fluxerr = np.sqrt(fluxerr**2 + confusion_std**2)
        elif combined_noise is True:
            ## combining all sigma
            fluxerr = np.sqrt(fluxerr**2 + confusion_std**2 + flux_c_err**2)
    else:
        # print("individual noise")
        # combine error bars from each individual confusion realization
        fluxerr = np.sqrt(fluxerr**2 + flux_c_err**2)

    # write output to txt
    if Newfile is True:
        # create a new file
        write_output_to_photoz(total_flux, fluxerr, source_id, filename, NewFile=True)

    else:
        # append to an existing file
        write_output_to_photoz(total_flux, fluxerr, source_id, filename, NewFile=False)

    return idx, total_flux, fluxerr


def compute_hdpi_new(zs, z_likelihood, frac=0.68):

    ''' 
    This script computes the 1d highest posterior density interval. 
    The HDPI importantly assumes that the true posterior is unimodal,
    so care should be taken to identify multi-modal PDFs before using it on arbitrary PDFs.
    '''

    idxs_hdpi = []
    idxmax = np.argmax(z_likelihood)
    idxs_hdpi.append(idxmax)

    psum = z_likelihood[idxmax]

    idx0 = np.argmax(z_likelihood)+1
    idx1 = np.argmax(z_likelihood)-1

    if idx0>=len(z_likelihood):
        print('idx0>=len(z_likelihood)')
        return None, None

    while True:

        if idx0==len(z_likelihood) or np.abs(idx1)==len(z_likelihood):
            print('idx0 = ', idx0)
            print('hit a limit, need to renormalize but passing None for now')
            return None, None

        if z_likelihood[idx0] > z_likelihood[idx1]:
            psum += z_likelihood[idx0]
            idxs_hdpi.append(idx0)
            idx0 += 1
        else:
            psum += z_likelihood[idx1]
            idxs_hdpi.append(idx1)
            idx1 -= 1

        if psum >= frac:
            break

    zs_credible = np.sort(zs[np.array(idxs_hdpi)])

    return zs_credible, np.array(idxs_hdpi)




## compute confusion lib stats
confusion_mean = np.mean(lib[:, 4::2], axis=0)
confusion_std = np.std(lib[:, 4::2], axis=0)

# hpdi of the confusion lib
fluxes = lib[:,4::2]
SIGMA = []
for c in range(fluxes.shape[1]):
    
    flux_c = fluxes[:, c]
    bins = np.linspace(0, flux_c.max(), 40)
    hist, bins_edge = np.histogram(flux_c, bins=bins)
    bins_center = bins_edge[:-1] + np.diff(bins_edge)[0]/2
    
    (flux_cred, hpdi_idxs) = compute_hdpi_new(bins_center, hist / np.sum(hist)) 
    hpdi = (flux_cred.max() - flux_cred.min())
    SIGMA.append(hpdi)
SIGMA = np.array(SIGMA)

for i, glxy_spec in enumerate(sed_noisy):

    ## only original errorbar & lib sigma errorbar
    (idx, F, fe) = add_one_confusion_to_phot_comb_noise(galaxy_spec=glxy_spec, 
                                                        confusion_lib=lib, 
                                                        confusion_mean=confusion_mean, 
                                                        filename=filename, 
                                                        Newfile=(i==0), 
                                                        confusion_std=SIGMA,
                                                        combined_noise=False)


## plot and double check the results
sed_conf = np.loadtxt(filename)
# photometric z-score
zscores = np.array([])
zscores_og = np.array([])
for s in range(len(sed_conf)):

    ## match ID    
    idx_match = np.where(COSMOS_tab['Tractor_ID']==sed_conf[s][0])[0][0]
    print(idx_match)
    
    zscore = (sed_conf[s][3::2] - sed_noiseless[idx_match][3:]) / sed_conf[s][4::2]
    zscore_og = (sed_noisy[s][3::2] - sed_noiseless[idx_match][3:]) / sed_noisy[s][4::2]
    zscores = np.append(zscores, zscore)
    zscores_og = np.append(zscores_og, zscore_og)
    
plt.figure(figsize=(6,5))
bins = np.linspace(-10,10,100)
hist = plt.hist(zscores, bins=bins, alpha=0.4, density=True)[0]
hist_og = plt.hist(zscores_og, bins=bins, alpha=0.4, density=True)[0]
plt.xlabel("Photometric z-score", fontsize=15)

# plot Gaussiain
xx = np.linspace(-10, 10, 300)
plt.plot(xx, Gaussian(xx, hist.max(), 1, 0), color='C0', label='Confusion ON, Gaussian sig=1')
plt.plot(xx, Gaussian(xx, hist_og.max(), 1, 0), color='C1', label='Confusion OFF, Gaussian sig=1')
plt.legend(fontsize=11)
plt.ylim(0, hist_og.max()*1.2)
plt.title("Clustering Library", fontsize=15)
plt.show()