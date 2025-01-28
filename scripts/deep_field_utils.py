
import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
from photoz_tools import *
import pandas as pd

import itertools
import matplotlib.pyplot as plt
import os
import scipy.interpolate as interp

import SPHEREx_ObsSimulator as SPobs
import SPHEREx_InstrumentSimulator as SPinst
import SPHEREx_SkySimulator as SPsky
from astropy.io import fits
from tractor import *
from tractor_utils import *

from scipy.interpolate import interpn
from astropy.coordinates import SkyCoord
import astropy.units as u


survey_plan_file = 'spherex_survey_plan_march_2021.fits'
SPHEREx_Pointings = SPobs.Pointings(input_file = survey_plan_file,
                                   Gaussian_jitter=1.8, 
                                   roll_angle='psi2')

from spherex_parameters import load_spherex_parameters
# Load instrument and project parameters as a dictionary
spherex_parameters = load_spherex_parameters()


### Functions 

def Gaussian(x, A, sig, x0):
    """
    A = amplitude
    sig = sigma
    x0 = Gaussian center
    """
    g = A * np.exp(-(x-x0)**2 / (2*sig**2))
    return(g)

def Gaussian2d(x, y, x0, y0, sigma):
    g = np.exp( - ((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    g /= np.sum(g) # normalize
    return(g)


### noise calc from Richard
def quick_sphx_Mjysr_to_e_per_s(R=None, eta_total=None, verbose=False):
    
    A = np.pi * (0.1) * (0.1)
    Omega = 9.04e-10

    if R is None:
        R = np.array([41, 41, 41, 35, 110, 130])
    if eta_total is None:
        eta_total = np.array([0.64, 0.67, 0.62, 0.60, 0.55, 0.51])

    Mjysr_to_e_per_s = (A * Omega * eta_total / R / 6.62607e-34 * 1e-20)
    if verbose:
        print('Mjysr_to_e_per_s is ', Mjysr_to_e_per_s)

    return Mjysr_to_e_per_s


def surf_brightness_to_photocurrent_cal(array_idx, n_samp=100, t_samp=1.5):

    Mjysr_to_e_per_s = quick_sphx_Mjysr_to_e_per_s()
    t_int = n_samp*t_samp
    cal = Mjysr_to_e_per_s[array_idx]*t_int

    return cal

def sort_wavelengths_to_detectors(central_wavelengths, band_lams = [0.75, 1.1, 1.64, 2.42, 3.82, 4.42, 5.0]):

    # band_lams is the wavelength edges of the detectors
    # assumes central_wavelengths is in micron
    # already zero indexed so don't need to with surf_brightness_to_photocurrent
    which_detectors = np.digitize(central_wavelengths, band_lams) - 1

    return which_detectors

def rough_photon_noise_from_flux_density(flux_density_uJy, sphx_sb_to_photocurrent_cal, sphx_pixel_side=6.2, n_samp=100):

    Apix = sphx_pixel_side**2/(4.25e10)

    ptsrc_sb = flux_density_uJy / Apix # uJy/sr
    ptsrc_sb_MJysr = ptsrc_sb * 1e-12 # 12 orders of magnitude between microJansky and megaJansky
    sigphot = np.sqrt(1.2*(ptsrc_sb_MJysr*sphx_sb_to_photocurrent_cal)*(n_samp**2+1)/(n_samp**2-1))

    return sigphot, Apix

def rough_photon_noise_from_sb(sb_Mjysr, sphx_sb_to_photocurrent_cal, n_samp=100):

    sigphot = np.sqrt(1.2*(sb_Mjysr*sphx_sb_to_photocurrent_cal)*(n_samp**2+1)/(n_samp**2-1))

    return sigphot

# add noise to noiseless photometry
def noisy_phot(flux_noiseless, wl, Nobs, shallow_dfnu=None, deep_dfnu=None, include_poisson_noise=True):
    """
    flux_noiseless = not necessarily SPHEREx binned, in uJy
    deep_field_sens = deep field point source sensitivity in SPHEREx channels, in uJy!
    wl = wavelength of SPHEREx channels
    Nobs_deep = number of observation (for deep field)
    """
    
    which_detectors = sort_wavelengths_to_detectors(wl)
    sb_to_photocurrent_cal = surf_brightness_to_photocurrent_cal(which_detectors)

    # convert to surface brightness (assume all falls in one SPHEREx pixel), 
    # convert to electrons, use Garnett and forest, go back to uJy
    sigphot, Apix = rough_photon_noise_from_flux_density(flux_noiseless, sb_to_photocurrent_cal)

    phot_noise = np.random.normal(0, 1, len(flux_noiseless))*sigphot
    phot_noise_MJy = phot_noise*Apix/sb_to_photocurrent_cal # back to flux densities
    phot_noise_mJy = phot_noise_MJy * 1e12
    poisson_sigma = sigphot*Apix*1e12/sb_to_photocurrent_cal
    
    if shallow_dfnu is not None: # uncertainties are in uJy
        noise_realiz_shallow = np.random.normal(0, 1, len(shallow_dfnu))*shallow_dfnu
        noisy_shallow_flux = flux_noiseless + noise_realiz_shallow

        if include_poisson_noise:
            noisy_shallow_flux += phot_noise_mJy/np.sqrt(Nobs)
            poisson_sigma_shallow = poisson_sigma/np.sqrt(Nobs)
        
        return noisy_shallow_flux


    if deep_dfnu is not None: # uncertainties are in uJy
        noise_realiz_deep = np.random.normal(0, 1, len(deep_dfnu))*deep_dfnu
        noisy_deep_flux = flux_noiseless + noise_realiz_deep

        if include_poisson_noise:
            noisy_deep_flux += phot_noise_mJy/np.sqrt(Nobs)
            poisson_sigma_deep = poisson_sigma/np.sqrt(Nobs)
            
        return noisy_deep_flux

# add noise to noiseless image (per spherex channel)
def noisy_phot_per_chan(image_noiseless, wl, Nobs, shallow_dfnu=None, deep_dfnu=None, include_poisson_noise=True):
    """
    One SPHEREx channel
    image_noiseless = 2D image of flux in uJy
    deep_field_sens = deep field point source sensitivity in one SPHEREx channel, in uJy!
    wl = central wavelength of the given channel (float)
    Nobs_deep = number of observation (for deep field)
    """
    
    which_detector = sort_wavelengths_to_detectors(wl)
    sb_to_photocurrent_cal = surf_brightness_to_photocurrent_cal(which_detector)

    # convert to surface brightness (assume all falls in one SPHEREx pixel), 
    # convert to electrons, use Garnett and forest, go back to uJy
    sigphot, Apix = rough_photon_noise_from_flux_density(image_noiseless, sb_to_photocurrent_cal)

    phot_noise = np.random.normal(0, 1, size=image_noiseless.shape)*sigphot
    phot_noise_MJy = phot_noise*Apix/sb_to_photocurrent_cal # back to flux densities
    phot_noise_uJy = phot_noise_MJy * 1e12
    poisson_sigma = sigphot*Apix*1e12/sb_to_photocurrent_cal
    
    if shallow_dfnu is not None: # uncertainties are in uJy
        noise_realiz_shallow = np.random.normal(0, 1, size=image_noiseless.shape)*shallow_dfnu
        noisy_shallow_flux = image_noiseless + noise_realiz_shallow

        if include_poisson_noise:
            noisy_shallow_flux += phot_noise_uJy/np.sqrt(Nobs)
            poisson_sigma_shallow = poisson_sigma/np.sqrt(Nobs)
        
        return noisy_shallow_flux


    if deep_dfnu is not None: # uncertainties are in uJy
        noise_realiz_deep = np.random.normal(0, 1, size=image_noiseless.shape)*deep_dfnu
        noisy_deep_flux = image_noiseless + noise_realiz_deep

        if include_poisson_noise:
            noisy_deep_flux += phot_noise_uJy/np.sqrt(Nobs)
            poisson_sigma_deep = poisson_sigma/np.sqrt(Nobs)
            
        return noisy_deep_flux

# include zodi photon noise to noiseless image (per spherex channel)
def noisy_phot_per_chan_w_zodi(image_noiseless, wl, Nobs, n_samp=100, Zodi=None, pixel_size=6.2, include_poisson_noise=True):
    """
    One SPHEREx channel
    image_noiseless = 2D image of flux in uJy / pixel
    wl = wavelength (float) in um
    Nobs = number of observations (deep field ~ 50, full sky ~4)
    n_samp = number of samples, 100 by default for a single exposure (~120s)
    Zodi = zodi_flux_uJy, include photon noise due to zodi; default is None
    pixel_size = 6.2 arcsec by default
    include_poisson_noise = True or False, add poisson noise onto the image or not.
    """
    
    which_detector = sort_wavelengths_to_detectors(wl)
    sb_to_photocurrent_cal = surf_brightness_to_photocurrent_cal(which_detector, n_samp=n_samp)

    # convert to surface brightness (assume all falls in one SPHEREx pixel), 
    # convert to electrons, use Garnett and forest, go back to uJy
    sigphot_img, Apix = rough_photon_noise_from_flux_density(image_noiseless, 
                                                             sb_to_photocurrent_cal, 
                                                             sphx_pixel_side=pixel_size,
                                                             n_samp=n_samp)
    
    # add zodi photon noise
    sigphot_zodi = rough_photon_noise_from_flux_density(Zodi, 
                                                        sb_to_photocurrent_cal, 
                                                        sphx_pixel_side=pixel_size,
                                                        n_samp=n_samp)[0] # number of e- / pixel

    
    # total photon noise
    if Zodi is not None:
        sigphot = np.sqrt(sigphot_img**2 + sigphot_zodi**2)
    else:
        sigphot = sigphot_img
    
    phot_noise = np.random.normal(0, 1, size=image_noiseless.shape) * sigphot/np.sqrt(Nobs) # number of e- / pixel
    phot_noise_MJy = phot_noise*Apix/sb_to_photocurrent_cal # back to flux densities
    phot_noise_uJy = phot_noise_MJy * 1e12 # uJy / pixel
    poisson_sigma = sigphot/np.sqrt(Nobs) * Apix*1e12/sb_to_photocurrent_cal # uJy / pixel
    

    if include_poisson_noise:
        image_noisy_flux_uJy = image_noiseless.copy()

        if Zodi is not None:
            image_noisy_flux_uJy += (phot_noise_uJy + Zodi)

        else:
            image_noisy_flux_uJy += phot_noise_uJy

        poisson_sigma_deep = poisson_sigma 
    
        # deal with nan values
        image_noisy_flux_uJy[np.isnan(image_noisy_flux_uJy)] = np.nanmedian(image_noisy_flux_uJy)

    return image_noisy_flux_uJy, poisson_sigma


def calc_zodi(zodi_coord, zodi_time, wavelength, SPsky, Apix):
    """Given SkyCoord, Astropy.time.Time, and SPHEREx Sky Simulator, calculate the zodi level"""
    zodi = SPsky.zodicalc.ModifiedKelsallModel()
    zodi_sb_Mjysr = zodi(zodi_time, zodi_coord, wavelength*u.um) # MJy / Sr
    zodi_flux_uJy = zodi_sb_Mjysr * Apix * 1e12 # uJy / pixel
    return zodi_sb_Mjysr, zodi_flux_uJy


# write a single input spectrum to an output file that can directly go to photo-z
def write_output_to_photoz(flux, flux_err, source_id, filename, NewFile=False):
    
    """
    In SPHEREx resolution (102 channels)
    flux and flux_err in mJy!!!
    """
    
    # write output .txt file
    spectrum = list(itertools.chain(*zip(flux, flux_err))) # in mJy
    # insert ID, ra, dec
    spectrum.insert(0, 0.0)
    spectrum.insert(0, 0.0)
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


# updated functions
# function to convolve input spectra with SPHEREx filters
def data_to_sphx_band(input_flux, input_wl):
    
    # work_dir = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/code_for_richard/spherex_filters102/clip_filter/"
    work_dir = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/code_for_richard/"
    path_filter = work_dir + "W1W2_grz_sphx_102filters.list"
    # path_filter = work_dir + "clip_filter_filename.txt"
    fn_filter = np.loadtxt(path_filter, dtype='str')
    
    sphx_flux, sphx_wl = np.zeros(102), np.zeros(102)
    
    for i in range(len(fn_filter)):
        
        # read in each filter
        (ww, T) = np.loadtxt(work_dir + fn_filter[i], unpack=True)
        ww = ww / 1e4 # um
        
        # calc spherex channel center
        fwave = np.where(T >= 0.1*np.max(T))[0]
        w_min = np.min(ww[fwave])
        w_max = np.max(ww[fwave])
        w_med = np.median(input_wl[(input_wl<w_max) & (input_wl>w_min)]) # median wavelength of a channel
        sphx_wl[i] = w_med
        
        # calc flux measured in each spherex band
        # interpolate input data
        flux_interp = interp.InterpolatedUnivariateSpline(input_wl, input_flux)
        # convolve with filter to obtain photometry in the channel; filter already normalized
        sphx_flux[i] = np.sum(T * flux_interp(ww))
    
    return(sphx_wl, sphx_flux)


### Updated version, borrowed from Richard's script. 

def read_one_filter(fn_filter):
    
    """
    Load one filter
    """
    
    work_dir = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/code_for_richard/"
    
    # read in each filter - wavelength & transmission value
    (wl, Tr) = np.loadtxt(work_dir + fn_filter, unpack=True)
    non0 = (Tr != 0)
    bandpass_wl = wl[non0] * 1e-4 # convert from angstrom to um
    bandpass_Tr = Tr[non0]
    bandpass_Tr /= np.sum(bandpass_Tr) # normalize

    # calc channel central wavelength
    bandpass_cen_wl = np.dot(bandpass_wl, bandpass_Tr) 

    # return : 1darray, 1darray, float
    return bandpass_wl, bandpass_Tr, bandpass_cen_wl

def get_all_filters(fn_filters):
    """
    fn_filters = a list of names of all sphx filters
    """
    
    all_bandpass_wl = []
    all_bandpass_Tr = []
    all_bandpass_cen_wl = []
    
    for i, fn_filter in enumerate(fn_filters):
        (bandpass_wl, bandpass_Tr, bandpass_cen_wl) = read_one_filter(fn_filter)
        
        all_bandpass_wl.append(bandpass_wl)
        all_bandpass_cen_wl.append(bandpass_cen_wl)
        all_bandpass_Tr.append(bandpass_Tr)
    
    return all_bandpass_wl, all_bandpass_Tr, all_bandpass_cen_wl
    
    
def sed_to_sphx_res(input_wl, input_flux, all_bandpass_wl, all_bandpass_Tr, all_bandpass_cen_wl, plot=False):
    
    """
    for one source's sed, input all filters (generated using get_all_filters)
    """
    
    # interpolate the input flux
    flux_interp = interp.InterpolatedUnivariateSpline(input_wl, input_flux)
    
    sphx_flux = np.zeros(len(all_bandpass_Tr))
    
    i = 0
    for (bandpass_wl, bandpass_Tr) in zip(all_bandpass_wl, all_bandpass_Tr):
        
        # calc the flux interpolated at the bandpass resolution
        band_fluxes = flux_interp(bandpass_wl)
        # convolve with the bandpass
        flux_thisband = np.dot(band_fluxes, bandpass_Tr) # assume already normalized
        sphx_flux[i] = flux_thisband
        
        i = i+1
        
    if plot:
        plt.plot(all_bandpass_cen_wl, sphx_flux, 'o', ms=5, color='cornflowerblue', label='SPHX obs')
        plt.plot(input_wl, input_flux, color='black', label='high res sed')
        plt.xlabel(r"Wavelength $\mu$m")
        plt.ylabel(r"Flux mJy")
        plt.yscale("log")
        plt.legend()
    
    return sphx_flux
    

def sed_to_sphx_res_filtering(input_wl, input_flux, fn_filters, plot=False):
    """
    Calculate flux observed by spherex for one given source sed
    """
    
    # interpolate the input flux
    flux_interp = interp.InterpolatedUnivariateSpline(input_wl, input_flux)
    
    sphx_flux = np.zeros(len(fn_filters))
    sphx_cen_wl = np.zeros(len(fn_filters))
    
    for i, fn_filter in enumerate(fn_filters):
        
        # calc the bandpass
        (bandpass_wl, bandpass_Tr, bandpass_cen_wl) = read_one_filter(fn_filter)
        # calc the flux interpolated at the bandpass resolution
        band_fluxes = flux_interp(bandpass_wl)
        # convolve with the bandpass
        flux_thisband = np.dot(band_fluxes, bandpass_Tr) # assume already normalized
        sphx_flux[i] = flux_thisband
        sphx_cen_wl[i] = bandpass_cen_wl
        
    if plot:
        plt.plot(sphx_cen_wl, sphx_flux, 'o', ms=5, color='cornflowerblue', label='SPHX obs')
        plt.plot(input_wl, input_flux, color='black', label='high res sed')
        plt.xlabel(r"Wavelength $\mu$m")
        plt.ylabel(r"Flux mJy")
        plt.yscale("log")
        plt.legend();
        
    return sphx_flux, sphx_cen_wl
        
# New function generating noisy spectra by directly loading a noiseless photometry file at spherex resolution!!!
def make_noisy_phot_to_photoz(sphx_noiseless_phot, N_sources, N_obs, output_txt, deep, 
                              all_bandpass_wl, all_bandpass_Tr, all_bandpass_cen_wl, new_file):
    
    # read the SPHEREx shallow and deep field sensitivities
    spec = pd.read_csv('/Users/gemmahuai//Desktop/CalTech/SPHEREx/SPHEREx_2023/Codes/intensity_mapper_v30.csv')
    band = np.array([1,2,3,4,5,6])
    wl = np.array([])
    sensitivity = np.array([])
    
    if deep == True:
        # deep field
        idx_d = (band-1)*5 + 3
    
        for i in range(len(band)):
            name = "Band {}.1".format(band[i])
            wl = np.append(wl, spec[name][3:66].to_numpy().astype(float))
            name = "deep.{}".format(idx_d[i])
            sensitivity = np.append(sensitivity, spec[name][3:66].to_numpy().astype(float) / 1) # uJy

        # convert to spherex resolution 
        
        sensitivity_interp = interp.InterpolatedUnivariateSpline(wl, sensitivity)
        sensitivity = sensitivity_interp(all_bandpass_cen_wl)
        
#         sensitivity = sed_to_sphx_res(wl, 
#                                       sensitivity, 
#                                       all_bandpass_wl, all_bandpass_Tr, all_bandpass_cen_wl, # filters
#                                       plot=False) # m Jy
        sensitivity = sensitivity * np.sqrt(50 / N_obs) # scaled by the number of observations
        
    else:
        # shallow field
        idx_s = (band-1)*3 + 1
        
        for i in range(len(band)):
            name = "Band {}.1".format(band[i])
            wl = np.append(wl, spec[name][3:66].to_numpy().astype(float))
            name = "shallow.{}".format(idx_s[i])
            sensitivity = np.append(sensitivity, spec[name][3:66].to_numpy().astype(float) / 1) # uJy

        sensitivity_interp = interp.InterpolatedUnivariateSpline(wl, sensitivity)
        sensitivity = sensitivity_interp(all_bandpass_cen_wl)
        
#         sensitivity = sed_to_sphx_res(wl, 
#                                       sensitivity, 
#                                       all_bandpass_wl, all_bandpass_Tr, all_bandpass_cen_wl, # filters
#                                       plot=False) # m Jy
    
    wl = all_bandpass_cen_wl
    
    # noiseless photometry
    sed = np.loadtxt(sphx_noiseless_phot) # ID, ra, dec, flux...
        
    # Add noise
    if deep == True: 
        
        # deep field
        for ii in range(N_sources): # len(data)
            
            flux_noiseless_sphx = sed[ii][3:]
            sed_ID = sed[ii][0]

            # add noise (deep field &/ poisson) 
            # flux, err in uJy
            (noisy_flux) = noisy_phot(flux_noiseless=flux_noiseless_sphx, 
                                      wl=wl, 
                                      Nobs=N_obs, 
                                      shallow_dfnu=None, 
                                      deep_dfnu=sensitivity, 
                                      include_poisson_noise=True)
            
            if (ii == 0) & (new_file == True):
                # create a new txt file (ascii)
                # convert flux, err to mJy
                write_output_to_photoz(flux=noisy_flux/1000, 
                                       flux_err=sensitivity/1000, 
                                       source_id=int(sed_ID), 
                                       filename=output_txt,
                                       NewFile=True)
            else: 
                # append to the existing file
                write_output_to_photoz(flux=noisy_flux/1000, 
                                       flux_err=sensitivity/1000, 
                                       source_id=int(sed_ID), 
                                       filename=output_txt,
                                       NewFile=False)
                
    else: 
        
        # shallow field
        for ii in range(N_sources): # len(data)
            
            flux_noiseless_sphx = sed[ii][3:]
            sed_ID = sed[ii][0]

            # add noise (deep field &/ poisson) 
            # flux, err in uJy
            (noisy_flux) = noisy_phot(flux_noiseless=flux_noiseless_sphx, 
                                      wl=wl, 
                                      Nobs=N_obs, 
                                      shallow_dfnu=sensitivity, 
                                      deep_dfnu=None, 
                                      include_poisson_noise=True)
            
            if (ii == 0) & (new_file == True):
                # create a new txt file (ascii)
                # convert flux, err to mJy
                write_output_to_photoz(flux=noisy_flux/1000, 
                                       flux_err=sensitivity/1000, 
                                       source_id=int(sed_ID), 
                                       filename=output_txt,
                                       NewFile=True)
            else: 
                # append to the existing file
                write_output_to_photoz(flux=noisy_flux/1000, 
                                       flux_err=sensitivity/1000, 
                                       source_id=int(sed_ID), 
                                       filename=output_txt,
                                       NewFile=False)
    return


### Generate a noiseless image in one channel
def gen_noiseless_img_per_chnl(main_source_Coord, flux_i_mJy, chnl, sources_idx, sources_tID, COSMOS_table, cosmos_full_phot, interp_PSF, Instrument, cutout_size_sphxpix=40, plot=False, verbose=False):
    """
    Takes ra, dec, flux (mJy), channel number of the central source, a list of nearby sources' ID, COSMOS input table, interpolated PSF of this channel;
    main_source_Coord = SkyCoord of the main source 
    (sources_idx, sources_tID) = direct outputs from find_nearby_sources() function
                                 already get rid of the central main source from the nearby source list!
    Instrument = I
    cosmos_full_phot = np.loadtxt("/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/Noiseless_phot_cosmos_nolines_fullCOSMOS.txt")
    Returns a cutout including all nearby sources in the oversampled pixel space.
    """
    
    # in one channel, create a cutout in oversampled space
    oversample = 5
    # cutout_size_sphxpix = 40 # input
    pixel_size = Instrument.pixel_size

    cutout_size_arcsec = cutout_size_sphxpix * pixel_size 
    cutout_size_oversample = cutout_size_sphxpix * oversample
    cutout_center = cutout_size_oversample // 2 + 1
    cutout_full = np.zeros((cutout_size_oversample, cutout_size_oversample))

    ## add the central main source
    kernel = interp_PSF.T 
    kernel_off_x = int(kernel.shape[0] / 2.0)
    kernel_off_y = int(kernel.shape[1] / 2.0)
    
    xlow = cutout_center - kernel_off_x
    xhigh = cutout_center + kernel_off_x
    ylow = cutout_center - kernel_off_y
    yhigh = cutout_center + kernel_off_y

    cutout_full[ylow:yhigh, xlow:xhigh] += kernel * flux_i_mJy 
    
    ## Other nearby sources
    # read in the full COSMOS noiseless photometry at SPHEREx resolution (NO lines!)

    # for each nearby source, need to know flux at spherex resolution
    ii = 0

    for source_idx, source_tID in zip(sources_idx, sources_tID):

#         if source_tID == sed_tID:
#             ii+=1
#             continue

        if verbose==True:
            print(f"i = {ii+1} / {len(sources_idx)}")
            
        source_Coord = SkyCoord(COSMOS_table[source_idx]['ALPHA_J2000'], 
                        COSMOS_table[source_idx]['DELTA_J2000'], 
                        unit='deg')
        dra, ddec = main_source_Coord.spherical_offsets_to(source_Coord) # check direction positive / negative

        # to pixel separation
        dx = dra.to_value(u.arcsec) / pixel_size
        dy = ddec.to_value(u.arcsec) / pixel_size
        if verbose == True:
            print(f'dx = {dx}, dy = {dy}')

        # check if this is a bright source in the ref cat
        if COSMOS_table[source_idx]['InSPHERxRefCat']==1:
            # convolve with the filter
            xoff = int(dx * oversample)
            yoff = int(dy * oversample)

            xlow = cutout_center + xoff - kernel_off_x
            xhigh = cutout_center + xoff + kernel_off_x
            ylow = cutout_center + yoff - kernel_off_y
            yhigh = cutout_center + yoff + kernel_off_y

            cutout_full[ylow:yhigh, xlow:xhigh] += (
                kernel * cosmos_full_phot[source_idx][3 + chnl]/1000
            ) # mJy
        else:
            # if it's a faint source, do not do convolution, directly add the flux to the oversampled pixel
            xoff = int(dx * oversample)
            yoff = int(dy * oversample)
            cutout_full[cutout_center+yoff, cutout_center+xoff] += cosmos_full_phot[source_idx][3 + chnl]/1000 # mJy

        ii+=1
        
    if plot:
        plt.figure(figsize=(14,4.5))
        plt.subplot(1,2,1)
        plt.imshow(cutout_full, extent=(-cutout_size_oversample//2, cutout_size_oversample//2, cutout_size_oversample//2, -cutout_size_oversample//2))
        cbar = plt.colorbar()
        cbar.set_label('mJy')
        plt.xlabel("X (oversampled pixel)")
        plt.ylabel("Y (oversampled pixel)")
        plt.subplot(1,2,2)
        plt.title("Central pixels")
        plt.imshow(cutout_full, extent=(-cutout_size_oversample//2, cutout_size_oversample//2, cutout_size_oversample//2, -cutout_size_oversample//2))
        plt.xlabel("X (oversampled pixel)")
        plt.ylabel("Y (oversampled pixel)")
        plt.xlim(-cutout_size_oversample//10, cutout_size_oversample//10)
        plt.ylim(-cutout_size_oversample//10, cutout_size_oversample//10)
        cbar = plt.colorbar()
        cbar.set_label('mJy')

        
    return cutout_full


# interpolate the SPHEREx PSFs
def interp_images_1d(imgs , Z1 , Z1_interp):
    ''' 
    Interpolates an array of images in 1 dimension
    
    INPUT:
    - imgs: an array of NxN images of shape (z , N , N), where z is the 3rd dimension (= number of images).
    - Z1: the axis of 3rd dimension (the simplest is np.arange(number_of_images))
    - Z1_interp: where to interpolate in 3rd dimension (relative to Z1). 
    
    OUTPUT:
    The interpolated NxN image.
    '''
    
    # get points
    n = imgs.shape[1]
    points = (Z1, np.arange(n), np.arange(n))
    
    # get interpolated points
    xi = np.rollaxis(np.mgrid[:n, :n], 0, 3).reshape((n**2, 2))
    xi = np.c_[np.repeat(Z1_interp,n**2), xi]
    #print(xi.shape)
    
    # interpolate
    img_interp = interpn(points, imgs, xi, method='linear', bounds_error=False, fill_value=None)
    img_interp = img_interp.reshape((n, n))
    #print(np.sum(img_interp))
    
    return(img_interp)



def find_nearby_sources(ra, dec, COSMOS_table, size=50/3600):
    """
    Given ra and dec, find all nearby sources from the COSMOS2020 catalog
    size = 50/3600 deg
    """
    R = ra
    D = dec
    ra_l = R - size/2
    ra_h = R + size/2
    dec_l = D - size/2
    dec_h = D + size/2
    want = (  (COSMOS_table['ALPHA_J2000']<ra_h) \
            & (COSMOS_table['ALPHA_J2000']>ra_l) \
            & (COSMOS_table["DELTA_J2000"]>dec_l) \
            & (COSMOS_table["DELTA_J2000"]<dec_h) )
    sources_ID = COSMOS_table["col1"][want] # pre-ID 
    sources_tID = COSMOS_table["ID"][want] # Tractor ID of the nearby sources
    return sources_ID, sources_tID

def find_nearby_bright_sources(ra, dec, COSMOS_table, size=50/3600):
    """
    Given ra and dec, find all nearby sources from the COSMOS2020 catalog that are in the SPHEREx reference catalog
    size = 50/3600 deg
    """
    R = ra
    D = dec
    ra_l = R - size/2
    ra_h = R + size/2
    dec_l = D - size/2
    dec_h = D + size/2
    want = (  (COSMOS_table['ALPHA_J2000']<ra_h) \
            & (COSMOS_table['ALPHA_J2000']>ra_l) \
            & (COSMOS_table["DELTA_J2000"]>dec_l) \
            & (COSMOS_table["DELTA_J2000"]<dec_h) \
            & (COSMOS_table['InSPHERxRefCat']==1)) # in the reference catalog
    sources_ID = COSMOS_table["col1"][want] # pre-ID 
    sources_tID = COSMOS_table["ID"][want] # Tractor ID of the nearby sources
    return sources_ID, sources_tID


# -------------------------------------------------------------------------------------------------------
### Tractor photometry related updated on 08/15/2024

def Gen_noiseless_img_per_chnl_fullcat(COSMOS_table_to_sim, chnl, sphx_wl, cosmos_full_phot, interp_wl_to_pix_y, Instrument, pixel_size=6.2, oversample=5, source_confusion='on', array_number=None, GaussianPSF=None, plot=False, verbose=False):
#def Gen_noiseless_img_per_chnl_fullcat(COSMOS_table_to_sim, chnl, cosmos_full_phot, interp_PSF, Instrument, pixel_size=6.2, oversample=5, source_confusion='on', array_number=None, plot=False, verbose=False):
    """
    Takes ra, dec, flux (mJy), channel number of the central source, a list of nearby sources' ID, COSMOS input table, interpolated PSF of this channel;
    main_source_Coord = SkyCoord of the main source 
    (sources_idx, sources_tID, sources_pixel_offset) = 
            direct outputs from find_nearby_sources() function
            already get rid of the central main source from the nearby source list;
            plus dx, dy offset of each nearby source
    interp_wl_to_pix_y: a list containing 5 splines mapping from wavelength to Y pixel position, assuming constant x = 1024
    Instrument = I
    pixel_size = 6.2 arcsec by default
    GaussianPSF = None or True; If True, will skip interp_wl_to_pix_y
    cosmos_full_phot = np.loadtxt("/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/Noiseless_phot_cosmos_nolines_fullCOSMOS.txt")
    Returns a cutout including all nearby sources in the oversampled pixel space.
    """
    
    ## in one channel, create a cutout in oversampled space
    pixel_size = pixel_size / 3600 # deg
    ra_min = COSMOS_table_to_sim['ALPHA_J2000'].min()
    ra_max = COSMOS_table_to_sim['ALPHA_J2000'].max()
    dec_min = COSMOS_table_to_sim['DELTA_J2000'].min()
    dec_max = COSMOS_table_to_sim['DELTA_J2000'].max()

    # Define the corners of the rectangle in RA, Dec
    corners = [
        (ra_min, dec_min),
        (ra_min, dec_max),
        (ra_max, dec_min),
        (ra_max, dec_max)
    ]


    # Convert each corner to ecliptic coordinates
    ecliptic_coords = []
    for ra, dec in corners:
        sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        ecliptic_coord = sky_coord.transform_to('barycentrictrueecliptic')
        ecliptic_coords.append(ecliptic_coord)

    # Extract ecliptic longitude and latitude from the converted coordinates
    ecliptic_longitudes = [coord.lon.deg for coord in ecliptic_coords]
    ecliptic_latitudes = [coord.lat.deg for coord in ecliptic_coords]

    # Calculate the size of the cutout in ecliptic coordinates
    ecliptic_lon_min = min(ecliptic_longitudes)
    ecliptic_lon_max = max(ecliptic_longitudes)
    ecliptic_lat_min = min(ecliptic_latitudes)
    ecliptic_lat_max = max(ecliptic_latitudes)

    ecliptic_lon_range = ecliptic_lon_max - ecliptic_lon_min
    ecliptic_lat_range = ecliptic_lat_max - ecliptic_lat_min
    dx = ecliptic_lon_range / (pixel_size)
    dy = ecliptic_lat_range / (pixel_size)

    # add buffer space for the psf kernel size (32 by 32) in the oversampled space based on the resolution
    cutout_X_oversampled_size = int(np.ceil(dx * oversample + 32 * 4 * (oversample / 5) * ((6.2/3600)/pixel_size)))
    cutout_Y_oversampled_size = int(np.ceil(dy * oversample + 32 * 4 * (oversample / 5) * ((6.2/3600)/pixel_size)))
    #print(cutout_X_oversampled_size, cutout_Y_oversampled_size)

    # create the cutout
    cutout_center = (cutout_X_oversampled_size//2+1, cutout_Y_oversampled_size//2+1)
    center_elon = ecliptic_lon_min + ecliptic_lon_range/2
    center_elat = ecliptic_lat_min + ecliptic_lat_range/2
    coord_cutout_center = SkyCoord(lon=center_elon * u.deg, lat=center_elat * u.deg, frame='barycentrictrueecliptic')

    cutout_full = np.zeros(shape=(cutout_Y_oversampled_size, cutout_X_oversampled_size))    
    
    ## add sources 
    
    # PSF kernel
    
    # Gaussian PSF
    if GaussianPSF is not None:
        if GaussianPSF is True:
            # approximate wavelengths for now
            sphx_wl_approx = np.linspace(0.75, 5., 102)
            # Gaussian PSF sigma range
            sphx_sigma = np.linspace(0.7, 1.8, 102)
            sphx_sigma_this_wl = np.interp(sphx_wl[chnl], sphx_wl_approx, sphx_sigma)
            size = Instrument.PSF.get_psf(0, 0, 1).shape[0]
            x = np.arange(size)
            y = np.arange(size)
            (x, y) = np.meshgrid(x, y)
            kernel = Gaussian2d(x, y, size//2-1, size//2-1, sphx_sigma_this_wl)
            
#             plt.imshow(kernel)
#             print(kernel.shape)
#             plt.show()
    
    else:
        array_number = sort_wavelengths_to_detectors(sphx_wl[chnl])+1
        y_pix = int(interp_wl_to_pix_y[array_number-1](sphx_wl[chnl]))
        # print('ypix = ', y_pix)
        kernel = Instrument.PSF.get_psf(x=1024, 
                                            y=y_pix, 
                                            array=array_number).T
#     print("Array number = ", array_number)
#     print("Y position = ", y_pix)
    
    # kernel = interp_PSF[chnl].T
    kernel_off_x = int(kernel.shape[0] / 2.0)
    kernel_off_y = int(kernel.shape[1] / 2.0)

    # sources to fit 
    refcat_srcs_XOFFs = []
    refcat_srcs_YOFFs = []
    refcat_srcs_names = []

    for ii in range(len(COSMOS_table_to_sim)):

        tractor_ID = COSMOS_table_to_sim['ID'][ii]
        idx = np.where(cosmos_full_phot[:, 0]==tractor_ID)[0][0]

        coord_this_source = SkyCoord(COSMOS_table_to_sim['ALPHA_J2000'][ii], 
                                     COSMOS_table_to_sim['DELTA_J2000'][ii], 
                                     unit='deg')
        coord_this_source_ecl = coord_this_source.transform_to('barycentrictrueecliptic')

        # calculate offsets from the cutout center
        # dlon, dlat = sed_Coord_ecl.spherical_offsets_to(near_coord)
        dlon = coord_this_source_ecl.lon.to("deg").value - coord_cutout_center.lon.to("deg").value
        dlat = coord_this_source_ecl.lat.to("deg").value - coord_cutout_center.lat.to("deg").value

        # pixel offset from the cutout center in the downsampled space
        dx = dlon / pixel_size
        dy = dlat / pixel_size

        # check if this is in the ref cat
        if COSMOS_table_to_sim[ii]['InSPHERxRefCat']==1:
            # convolve with the filter
            xoff = round(dx * oversample)
            yoff = round(dy * oversample)

            xlow = cutout_center[0] + xoff - kernel_off_x
            xhigh = cutout_center[0] + xoff + kernel_off_x
            ylow = cutout_center[1] + yoff - kernel_off_y
            yhigh = cutout_center[1] + yoff + kernel_off_y
            
            cutout_full[ylow:yhigh, xlow:xhigh] += (
                kernel * cosmos_full_phot[idx][3 + chnl]/1000
            ) # mJy

            refcat_srcs_XOFFs.append(xoff)
            refcat_srcs_YOFFs.append(yoff)
            refcat_srcs_names.append(tractor_ID)

        else:
            
            if source_confusion!='on':
                continue

            xoff = round(dx * oversample)
            yoff = round(dy * oversample)

            xlow = cutout_center[0] + xoff - kernel_off_x
            xhigh = cutout_center[0] + xoff + kernel_off_x
            ylow = cutout_center[1] + yoff - kernel_off_y
            yhigh = cutout_center[1] + yoff + kernel_off_y

            cutout_full[ylow:yhigh, xlow:xhigh] += (
                kernel * cosmos_full_phot[idx][3 + chnl]/1000
            ) # mJy
            

    
    if plot:
        plt.figure(figsize=(6,6))
        plt.imshow(cutout_full)
        cbar = plt.colorbar()
        cbar.set_label('mJy / pixel')
        plt.title('Noiseless Image')
        plt.xlabel("X (oversampled pixel)")
        plt.ylabel("Y (oversampled pixel)")
        plt.show()

        
    return cutout_full, cutout_center, refcat_srcs_names, refcat_srcs_XOFFs, refcat_srcs_YOFFs

# Tractor photometry updated!!! with zodi light
def Tractor_blended_full_w_zodi(Coords_offset, image_sb_MJySr, wavelength, interp_wl_to_y, cutout_center_oversample, Sky, Instrument, GaussianPSF=False, sigma=0.4, pixel_size=6.2):
    """
    Coords_offset = coordinates offset for all sources in oversampled space, ex. [(0,0), (1, 4)]
    image_sb_MJySr = downsampled noisy images in surface brightness units (MJy / Sr)
    wavelength = central wavelength of this channel 
    interp_wl_to_y = a list of 6 interpolation splines for 6 detector arrays mapping from wavelength to y pixel position, fixing x=1024
    cutout_center_oversample = coordinates of the cutout center 
    sigma = 1 sigma noise of the cutout
    GaussianPSF = True/False; if True, ignore interp_wl_to_y, use perfect Gaussian PSFs instead with continuous Neff.
    pixel_size = 6.2 arcsec by default
    """
    oversample = 5
    if Coords_offset:
        # total number of sources
        N_sources = len(Coords_offset)
        # xoffset, yoffset for each source relative to the center of the oversampled image center
        XOFFs, YOFFs = [], []
        for ii in range(len(Coords_offset)):
            XOFFs.append(Coords_offset[ii][0])
            YOFFs.append(Coords_offset[ii][1])
    else: 
        return

    
    tractor_source_list = []

    # add nearby sources
    for i in range(len(Coords_offset)):

        x_i = (cutout_center_oversample[0]+XOFFs[i]-1 + 0.5) / oversample - 0.5
        y_i = (cutout_center_oversample[1]+YOFFs[i]-1 + 0.5) / oversample - 0.5

        tractor_source_list.append(PointSource(PixPos(x_i, y_i),Flux(np.random.uniform(high=5000))))

    # PSF kernel
    array_number = sort_wavelengths_to_detectors(wavelength) + 1
    y_pix = int(interp_wl_to_y[array_number-1](wavelength))
    
    if GaussianPSF is not None:
        # use analytical gaussian PSF
        spherex_psf = SPHERExTractorPSF(Instrument.PSF, 
                                        array_number, 
                                        xshift=1024, 
                                        yshift=y_pix, 
                                        oversample=5,
                                        wavelength=wavelength)
    else:
        # use interp_wl_to_y
        spherex_psf = SPHERExTractorPSF(Instrument.PSF, 
                                        array_number, 
                                        xshift=1024, 
                                        yshift=y_pix, 
                                        oversample=5)

    pix_sr = ((pixel_size * u.arcsec) ** 2).to_value(u.sr)  # Sr / pixel 

    inverse_variance = np.zeros_like(image_sb_MJySr)+1/sigma**2
    # tractor image
    tim = tractor.Image(data=image_sb_MJySr, invvar=inverse_variance,
                        psf=spherex_psf, wcs=NullWCS(pixscale=pixel_size), photocal=LinearPhotoCal(1.),
                        sky=ConstantSky(Sky))

    for ii in range(len(tractor_source_list)):
        tractor_source_list[ii].freezeAllRecursive()
        tractor_source_list[ii].thawParam('brightness')
    tim.freezeAllRecursive() #Image parameters fixed

    trac_spherex = tractor.Tractor([tim], tractor_source_list)
    optres, umodels = trac_spherex.optimize_forced_photometry(variance=True) ## engine.py --> call optimize.py optimizer.optimize
    model  = trac_spherex.getModelImage(0)

    out = []
    for ii in range(len(tractor_source_list)):
        out.append([ii+1,tractor_source_list[ii].getParams()[0],\
                    1/np.sqrt(optres.IV[ii])])
        
    fluxres = Table(rows=out,
            names=('ID', 'Flux', 'Fluxerr'),
            meta={'name': 'tractor_result'})

    return(fluxres, model, umodels) 

# Generate one noiseless image for each source, for isolated photometry

def Gen_noiseless_img_per_chnl_onesrc(pixel_size, oversample, subpixels, chnl, sphx_wl, flux_true_mJy, Instrument, interp_wl_to_pix_y, GaussianPSF=None, plot=False):
    """
    Generate an oversampled cutout for one source (at the center) with subpixel shifts given.
    flux_true_mJy = true input flux from cosmos_full, float in mJy
    interp_wl_to_pix_y: a list containing 5 splines mapping from wavelength to Y pixel position, assuming constant x = 1024
    Instrument = I
    subpixels = (sub_x, sub_y), subpixel shifts
    pixel_size = 6.2 arcsec by default
    GaussianPSF = None or True; If True, will skip interp_wl_to_pix_y
    Returns a cutout including all nearby sources in the oversampled pixel space.
    """
    
    cutout_size = 5 # downsampled pixels (spherex)
    (sub_x, sub_y) = subpixels

    # add buffer space for the psf kernel size (32 by 32) in the oversampled space based on the resolution
    cutout_X_oversampled_size = int(np.ceil(cutout_size * oversample + 32 * 2 * (oversample / 5) * (6.2/pixel_size)))
    cutout_Y_oversampled_size = int(np.ceil(cutout_size * oversample + 32 * 2 * (oversample / 5) * (6.2/pixel_size)))

    # create the cutout
    cutout_center = (cutout_X_oversampled_size//2+1, cutout_Y_oversampled_size//2+1)
    cutout_full = np.zeros(shape=(cutout_Y_oversampled_size, cutout_X_oversampled_size))    
    
    # add source
    
    # PSF kernel
    # Gaussian PSF
    if GaussianPSF is not None:
        if GaussianPSF is True:
            # approximate wavelengths for now
            sphx_wl_approx = np.linspace(0.75, 5., 102)
            # Gaussian PSF sigma range
            sphx_sigma = np.linspace(0.7, 1.8, 102)
            sphx_sigma_this_wl = np.interp(sphx_wl[chnl], sphx_wl_approx, sphx_sigma)
            sizex = Instrument.PSF.get_psf(0, 0, 1).shape[0]
            sizey = Instrument.PSF.get_psf(0, 0, 1).shape[1]
            x = np.arange(sizex)
            y = np.arange(sizey)
            (x, y) = np.meshgrid(x, y)
            kernel = Gaussian2d(x, y, sizex//2-1, sizey//2-1, sphx_sigma_this_wl)
            
#             plt.imshow(kernel)
#             print(kernel.shape)
#             plt.show()
    
    else:
        array_number = sort_wavelengths_to_detectors(sphx_wl[chnl])+1
        y_pix = int(interp_wl_to_pix_y[array_number-1](sphx_wl[chnl]))
        # print('ypix = ', y_pix)
        kernel = Instrument.PSF.get_psf(x=1024, 
                                            y=y_pix, 
                                            array=array_number).T
#     print("Array number = ", array_number)
#     print("Y position = ", y_pix)
    
    # kernel = interp_PSF[chnl].T
    kernel_off_x = int(kernel.shape[0] / 2.0)
    kernel_off_y = int(kernel.shape[1] / 2.0)

    xlow = cutout_center[0] + sub_x - kernel_off_x
    xhigh = cutout_center[0] + sub_x + kernel_off_x
    ylow = cutout_center[1] + sub_y - kernel_off_y
    yhigh = cutout_center[1] + sub_y + kernel_off_y

    cutout_full[ylow:yhigh, xlow:xhigh] += (
        kernel * flux_true_mJy
    ) # mJy

    if plot:
        plt.figure(figsize=(6,6))
        plt.imshow(cutout_full)
        cbar = plt.colorbar()
        cbar.set_label('mJy / pixel')
        plt.title('Noiseless Image')
        plt.xlabel("X (oversampled pixel)")
        plt.ylabel("Y (oversampled pixel)")
        plt.show()

        
    return cutout_full, cutout_center

# Tractor photometry updated!!! with zodi light, only photometer one source at the center
def Tractor_blended_full_w_zodi_onesrc(image_sb_MJySr, wavelength, interp_wl_to_y, cutout_center_oversample, Sky, Instrument, GaussianPSF=False, sigma=0.4, pixel_size=6.2):
    """
    Coords_offset = coordinates offset for all sources in oversampled space, ex. [(0,0), (1, 4)]
    image_sb_MJySr = downsampled noisy images in surface brightness units (MJy / Sr)
    wavelength = central wavelength of this channel 
    interp_wl_to_y = a list of 6 interpolation splines for 6 detector arrays mapping from wavelength to y pixel position, fixing x=1024
    cutout_center_oversample = coordinates of the cutout center 
    sigma = 1 sigma noise of the cutout
    GaussianPSF = True/False; if True, ignore interp_wl_to_y, use perfect Gaussian PSFs instead with continuous Neff.
    pixel_size = 6.2 arcsec by default
    """

    tractor_source_list = []
    oversample = 5

    x_0 = (cutout_center_oversample[0]-1+0.5) / oversample - 0.5
    y_0 = (cutout_center_oversample[1]-1+0.5) / oversample - 0.5

    tractor_source_list.append(PointSource(PixPos(x_0, y_0),Flux(np.random.uniform(high=5000))))

        
    # PSF kernel
    array_number = sort_wavelengths_to_detectors(wavelength) + 1
    y_pix = int(interp_wl_to_y[array_number-1](wavelength))
    
    if GaussianPSF is not None:
        # use analytical gaussian PSF
        spherex_psf = SPHERExTractorPSF(Instrument.PSF, 
                                        array_number, 
                                        xshift=1024, 
                                        yshift=y_pix, 
                                        oversample=5,
                                        wavelength=wavelength)
    else:
        # use interp_wl_to_y
        spherex_psf = SPHERExTractorPSF(Instrument.PSF, 
                                        array_number, 
                                        xshift=1024, 
                                        yshift=y_pix, 
                                        oversample=5)

    pix_sr = ((pixel_size * u.arcsec) ** 2).to_value(u.sr)  # Sr / pixel 

    inverse_variance = np.zeros_like(image_sb_MJySr)+1/sigma**2
    # tractor image
    tim = tractor.Image(data=image_sb_MJySr, invvar=inverse_variance,
                        psf=spherex_psf, wcs=NullWCS(pixscale=pixel_size), photocal=LinearPhotoCal(1.),
                        sky=ConstantSky(Sky))

    
    for ii in range(len(tractor_source_list)):
        tractor_source_list[ii].freezeAllRecursive()
        tractor_source_list[ii].thawParam('brightness')

    tim.freezeAllRecursive() #Image parameters fixed

    trac_spherex = tractor.Tractor([tim], tractor_source_list)

    optres, umodels = trac_spherex.optimize_forced_photometry(variance=True) ## engine.py --> call optimize.py optimizer.optimize
    model  = trac_spherex.getModelImage(0)

    out = []
    for ii in range(len(tractor_source_list)):
        out.append([ii+1,tractor_source_list[ii].getParams()[0],\
                    1/np.sqrt(optres.IV[ii])])
        
    fluxres = Table(rows=out,
            names=('ID', 'Flux', 'Fluxerr'),
            meta={'name': 'tractor_result'})

    return(fluxres, model, umodels) 

# Photometry on postage stamps
def Photometer_postage_stamps_given_anImage(cutout_full, Instrument, xoffs, yoffs, names, sphx_wl, channel, crd_zodi, time_zodi, interp_wl_to_y_splines, output_table, pixel_size=6.2, stamp_size_sphx_pix=20, GaussianPSF=True, photometry=True, plot=False):
    """
    Inputs:
    cutout_full = full noiseless image (oversampled) in mJy/pixel, direct output from Gen() function.
    Instrument = SPHEREx_Instrument class.
    xoffs, yoffs = x, y offsets from the cutout center, direct output from Gen() function
    sphx_wl = SPHEREx channel centers in um
    channel = which SPHEREx channel this is
    interp_wl_to_y_splines = a list of 6 splines mapping from wavelength to y position on the detectors.
    output_table = initialized Table with 'ID', 'Flux', and 'Fluxerr' columns where 'ID' is filled with tractor IDs
    pixels_size = 6.2'' by default
    stamp_size_sphx_pix = size of the each postage stamp in spherex pixel resolution, default is 20.
    photometry = True/False: whether include noise and do photometry with Tractor or not. 
                If True, carry out the full procedure and return Tractor fit;
                If False, only return the noiseless postage stamps
    plot = True/False: whether to plot all postage stamps or not.
    
    """
    oversample = 5
    size_stamp = stamp_size_sphx_pix * oversample
    N_stamps_x = int(cutout_full.shape[1] // size_stamp)
    N_stamps_y = int(cutout_full.shape[0] // size_stamp)
    
    # add some buffer area surrounding each postage stamp in case some sources are sitting on the edges. 
    size_buffer = Instrument.PSF.get_psf(0,0,1).shape[0] 
    size_buffer_outer = size_buffer

    if plot==True:
        plt.figure(figsize=(20,20))
        
    count = 1 # for plotting purpose
    
    Stamps = []
    Centers = []
    XOFFs = []
    YOFFs = []
    XOFFs_buf = []
    YOFFs_buf = []
    NAMEs = []
    NAMEs_buf = []
    
    cutout_center = (cutout_full.shape[1]//2+1, cutout_full.shape[0]//2+1)
    x_crds = cutout_center[0] + np.array(xoffs) - 1.
    y_crds = cutout_center[1] + np.array(yoffs) - 1.
    
    for j in range(N_stamps_y):

        # x_l, x_h, y_l, y_h: for selecting sources to fit that will go into the output table
        # X_l, X_h, Y_l, Y_h: for selecting sources in the inner buffer zone, will be fitted not going into the table
        # XX_l, XX_h, YY_l, YY_h: no sources in this outer buffer zone, mainly buffer for generating unit flux models
        
        y_l = j * size_stamp
        if j==0:
            Y_l = y_l
            YY_l = y_l
        else:
            Y_l = y_l - size_buffer
            YY_l = y_l - size_buffer - size_buffer_outer

        if j == (N_stamps_y-1):
            y_h = cutout_full.shape[0]-1
            Y_h = y_h
            YY_h = y_h
        else:
            y_h = (j+1) * size_stamp
            Y_h = y_h + size_buffer
            YY_h = y_h + size_buffer + size_buffer_outer

        for i in range(N_stamps_x):

            x_l = i * size_stamp
            if i==0:
                X_l = x_l
                XX_l = x_l
            else:
                X_l = x_l - size_buffer
                XX_l = x_l - size_buffer - size_buffer_outer
            if i == (N_stamps_x-1):
                x_h = cutout_full.shape[1]-1
                X_h = x_h
                XX_h = x_h
            else:
                x_h = (i+1) * size_stamp
                X_h = x_h + size_buffer
                XX_h = x_h + size_buffer + size_buffer_outer

            # select sources in this postage stamps
            want = (x_crds>=x_l) & (x_crds<=x_h) & (y_crds>=y_l) & (y_crds<=y_h)
            want_all = (x_crds>=X_l) & (x_crds<=X_h) & (y_crds>=Y_l) & (y_crds<=Y_h)
            want_buf = ~want & want_all # sources in the buffer zone

            X_this = np.array(x_crds[want]) - (XX_l)
            Y_this = np.array(y_crds[want]) - (YY_l)
            X_this_buf = np.array(x_crds[want_buf]) - (XX_l)
            Y_this_buf = np.array(y_crds[want_buf]) - (YY_l)


            # this postage stamp

            stamp_this = cutout_full[YY_l:YY_h, XX_l:XX_h]
            Stamps.append(stamp_this)

            stamp_center = (stamp_this.shape[1]//2+1, stamp_this.shape[0]//2+1)
            Centers.append(stamp_center)

            Xoff_this = X_this - stamp_center[0]
            XOFFs.append(Xoff_this)
            Xoff_this_buf = X_this_buf - stamp_center[0]
            XOFFs_buf.append(Xoff_this_buf)

            Yoff_this = Y_this - stamp_center[1]
            YOFFs.append(Yoff_this)
            Yoff_this_buf = Y_this_buf - stamp_center[1]
            YOFFs_buf.append(Yoff_this_buf)

            name = np.array(names)[want]
            NAMEs.append(name)
            name_buf = np.array(names)[want_buf]
            NAMEs_buf.append(name_buf)
            
            if plot==True:

                plt.subplot(N_stamps_y, N_stamps_x, count)
                plt.imshow(stamp_this, vmax=0.005)
                plt.plot(X_this, Y_this, 'o', ms=2, color='red')
                plt.plot(X_this_buf, Y_this_buf, 'o', ms=2, color='blue')
                plt.title(f"{XX_l}, {XX_h}, {YY_l}, {YY_h}", fontsize=8)

            count += 1

            
    if plot==True:
        plt.show()
    
    if photometry==False:
        
        return Stamps, Centers, XOFFs, YOFFs, NAMEs, XOFFs_buf, YOFFs_buf, NAMEs_buf
    
    else:
        ## downsample, add noise, and do photometry
        pix_sr = ((pixel_size * u.arcsec) ** 2).to_value(u.sr)  # solid angle (Omega_pixel)
        srcpix2sky = oversample**2 / pix_sr * 1e-9  # from mJy to MJy / Sr
        
        
        if plot==True:
            plt.figure(figsize=(18,18))
            
        # loop through all postage stamps
        for i in range(len(Stamps)):
            stamp = Stamps[i]
            x_offs = np.array(XOFFs[i])
            y_offs = np.array(YOFFs[i])
            names_ = NAMEs[i]
            center = Centers[i]

            # with sources in the buffer zone
            x_offs = np.append(x_offs, XOFFs_buf[i]) + 1.
            y_offs = np.append(y_offs, YOFFs_buf[i]) + 1.

            if len(names_)==0:
                continue

            # convert to surface brightness units 
            stmp_noiseless_ds_flux_mJy = downscale_local_mean(stamp, (oversample, oversample)) * oversample**2 # in mJy / pixel

            # convert back to flux (uJy / pixel) and add noise, then back to surface brightness
            stmp_noiseless_ds_flux_uJy = stmp_noiseless_ds_flux_mJy * 1e3 # uJy / pixel


            (zodi_sb_Mjysr_jj, zodi_flux_uJy_jj) = calc_zodi(crd_zodi, time_zodi, sphx_wl[channel], SPsky, pix_sr)

            (stmp_noisy_ds_flux_uJy, sigma_poisson) = noisy_phot_per_chan_w_zodi(image_noiseless=stmp_noiseless_ds_flux_uJy, 
                                                                         wl=sphx_wl[channel], 
                                                                         Nobs=1, 
                                                                         Zodi=zodi_flux_uJy_jj, 
                                                                         n_samp=100 * 50,
                                                                         pixel_size=pixel_size, 
                                                                         include_poisson_noise=True)

            stmp_noisy_ds_sb_Mjysr = stmp_noisy_ds_flux_uJy * 1e-12 / pix_sr # to MJy / Sr
            sigma_poisson_sb_Mjysr = sigma_poisson * 1e-12 / pix_sr # to MJy / Sr

        #     plt.imshow(stamp)
        #     plt.colorbar()
        #     print(x_offs, y_offs, names_)
        #     plt.plot(x_offs+center[0], y_offs+center[1], 'o', ms=2, color='red')
        #     plt.show()

            x_i = (center[0]+x_offs-1 + 0.5) / oversample - 0.5
            y_i = (center[1]+y_offs-1 + 0.5) / oversample - 0.5

#             plt.imshow(stmp_noiseless_ds_flux_mJy)
#             plt.plot(x_i, y_i, 'o', ms=2, color='red')
#             plt.colorbar()
#             plt.show()


            # tractor photometry 
            coordinates = list(zip(x_offs, y_offs))
            # print("  N sources to fit = ", len(coordinates))

            fluxres, model, umodels = Tractor_blended_full_w_zodi(Coords_offset=coordinates, 
                                                  image_sb_MJySr=stmp_noisy_ds_sb_Mjysr - np.nanmedian(stmp_noisy_ds_sb_Mjysr), 
                                                  wavelength=sphx_wl[channel], 
                                                  interp_wl_to_y=interp_wl_to_y_splines,
                                                  cutout_center_oversample=center, 
                                                  sigma=sigma_poisson_sb_Mjysr, 
                                                  Instrument=Instrument,
                                                  GaussianPSF=GaussianPSF,
                                                  Sky=0.,
                                                  pixel_size=pixel_size)
            
            fluxres = fluxres[:(len(fluxres)-len(XOFFs_buf[i]))] # remove srcs in the buffer zone
            fluxres['Flux'] = fluxres['Flux'] * pix_sr * 1e9 # in mJy
            fluxres['Fluxerr'] = fluxres['Fluxerr'] * pix_sr * 1e9 # in mJy
            
            if plot==True:
#                 plt.figure(figsize=(10,4.5))
#                 plt.subplot(1,2,1)
#                 plt.imshow(stmp_noisy_ds_sb_Mjysr - np.median(stmp_noisy_ds_sb_Mjysr))
#                 plt.colorbar()
#                 plt.subplot(1,2,2)
#                 plt.imshow(stmp_noisy_ds_sb_Mjysr - np.median(stmp_noisy_ds_sb_Mjysr) - model)
#                 plt.colorbar()
#                 plt.show()

                plt.subplot(N_stamps_y, N_stamps_x, i+1)
                plt.imshow(stmp_noisy_ds_sb_Mjysr - np.median(stmp_noisy_ds_sb_Mjysr) - model)
                plt.colorbar()
            
            # append to the table
            ind = np.isin(output_table['ID'], NAMEs[i])
            output_table[f'Flux{channel+1}'][ind] = fluxres['Flux']
            output_table[f'Fluxerr{channel+1}'][ind] = fluxres['Fluxerr']

        if plot==True:
            plt.show()

        return
    
# # return x, y offsets of all sources in the given table
def Gen_noiseless_img_per_chnl_fullcat_allGivenSrcs(COSMOS_table_to_sim, chnl, sphx_wl, cosmos_full_phot, interp_wl_to_pix_y, Instrument, pixel_size=6.2, oversample=5, array_number=None, GaussianPSF=None, plot=False, verbose=False):
    """
    Return coordinates of all of the sources in the input catalog, regardless of reference catalog
    --------------
    Takes ra, dec, flux (mJy), channel number of the central source, a list of nearby sources' ID, COSMOS input table, interpolated PSF of this channel;
    main_source_Coord = SkyCoord of the main source 
    (sources_idx, sources_tID, sources_pixel_offset) = 
            direct outputs from find_nearby_sources() function
            already get rid of the central main source from the nearby source list;
            plus dx, dy offset of each nearby source
    interp_wl_to_pix_y: a list containing 5 splines mapping from wavelength to Y pixel position, assuming constant x = 1024
    Instrument = I
    pixel_size = 6.2 arcsec by default
    GaussianPSF = None or True; If True, will skip interp_wl_to_pix_y
    cosmos_full_phot = np.loadtxt("/Users/gemmahuai/Desktop/CalTech/SPHEREx/Redshift/Noiseless_phot_cosmos_nolines_fullCOSMOS.txt")
    Returns a cutout including all nearby sources in the oversampled pixel space.
    """
    
    ## in one channel, create a cutout in oversampled space
    pixel_size = pixel_size / 3600 # deg
    ra_min = COSMOS_table_to_sim['ALPHA_J2000'].min()
    ra_max = COSMOS_table_to_sim['ALPHA_J2000'].max()
    dec_min = COSMOS_table_to_sim['DELTA_J2000'].min()
    dec_max = COSMOS_table_to_sim['DELTA_J2000'].max()

    # Define the corners of the rectangle in RA, Dec
    corners = [
        (ra_min, dec_min),
        (ra_min, dec_max),
        (ra_max, dec_min),
        (ra_max, dec_max)
    ]

    # Convert each corner to ecliptic coordinates
    ecliptic_coords = []
    for ra, dec in corners:
        sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        ecliptic_coord = sky_coord.transform_to('barycentrictrueecliptic')
        ecliptic_coords.append(ecliptic_coord)

    # Extract ecliptic longitude and latitude from the converted coordinates
    ecliptic_longitudes = [coord.lon.deg for coord in ecliptic_coords]
    ecliptic_latitudes = [coord.lat.deg for coord in ecliptic_coords]

    # Calculate the size of the cutout in ecliptic coordinates
    ecliptic_lon_min = min(ecliptic_longitudes)
    ecliptic_lon_max = max(ecliptic_longitudes)
    ecliptic_lat_min = min(ecliptic_latitudes)
    ecliptic_lat_max = max(ecliptic_latitudes)

    ecliptic_lon_range = ecliptic_lon_max - ecliptic_lon_min
    ecliptic_lat_range = ecliptic_lat_max - ecliptic_lat_min
    dx = ecliptic_lon_range / (pixel_size)
    dy = ecliptic_lat_range / (pixel_size)

    # add buffer space for the psf kernel size (32 by 32) in the oversampled space based on the resolution
    cutout_X_oversampled_size = int(np.ceil(dx * oversample + 32 * 4 * (oversample / 5) * ((6.2/3600)/pixel_size)))
    cutout_Y_oversampled_size = int(np.ceil(dy * oversample + 32 * 4 * (oversample / 5) * ((6.2/3600)/pixel_size)))
    #print(cutout_X_oversampled_size, cutout_Y_oversampled_size)

    # create the cutout
    cutout_center = (cutout_X_oversampled_size//2+1, cutout_Y_oversampled_size//2+1)
    center_elon = ecliptic_lon_min + ecliptic_lon_range/2
    center_elat = ecliptic_lat_min + ecliptic_lat_range/2
    coord_cutout_center = SkyCoord(lon=center_elon * u.deg, lat=center_elat * u.deg, frame='barycentrictrueecliptic')

    cutout_full = np.zeros(shape=(cutout_Y_oversampled_size, cutout_X_oversampled_size))    
    
    ## add sources 
    
    # PSF kernel
    
    # Gaussian PSF
    if GaussianPSF is not None:
        if GaussianPSF is True:
            # approximate wavelengths for now
            sphx_wl_approx = np.linspace(0.75, 5., 102)
            # Gaussian PSF sigma range
            sphx_sigma = np.linspace(0.7, 1.8, 102)
            sphx_sigma_this_wl = np.interp(sphx_wl[chnl], sphx_wl_approx, sphx_sigma)
            size = Instrument.PSF.get_psf(0, 0, 1).shape[0]
            x = np.arange(size)
            y = np.arange(size)
            (x, y) = np.meshgrid(x, y)
            kernel = Gaussian2d(x, y, size//2-1, size//2-1, sphx_sigma_this_wl)
            
#             plt.imshow(kernel)
#             print(kernel.shape)
#             plt.show()
    
    else:
        array_number = sort_wavelengths_to_detectors(sphx_wl[chnl])+1
        y_pix = int(interp_wl_to_pix_y[array_number-1](sphx_wl[chnl]))
        # print('ypix = ', y_pix)
        kernel = Instrument.PSF.get_psf(x=1024, 
                                        y=y_pix, 
                                        array=array_number).T
#     print("Array number = ", array_number)
#     print("Y position = ", y_pix)
    
    # kernel = interp_PSF[chnl].T
    kernel_off_x = int(kernel.shape[0] / 2.0)
    kernel_off_y = int(kernel.shape[1] / 2.0)

    # sources to fit 
    refcat_srcs_XOFFs = []
    refcat_srcs_YOFFs = []
    refcat_srcs_names = []

    for ii in range(len(COSMOS_table_to_sim)):

        tractor_ID = COSMOS_table_to_sim['ID'][ii]
        idx = np.where(cosmos_full_phot[:, 0]==tractor_ID)[0][0]

        coord_this_source = SkyCoord(COSMOS_table_to_sim['ALPHA_J2000'][ii], 
                                     COSMOS_table_to_sim['DELTA_J2000'][ii], 
                                     unit='deg')
        coord_this_source_ecl = coord_this_source.transform_to('barycentrictrueecliptic')

        # calculate offsets from the cutout center
        # dlon, dlat = sed_Coord_ecl.spherical_offsets_to(near_coord)
        dlon = coord_this_source_ecl.lon.to("deg").value - coord_cutout_center.lon.to("deg").value
        dlat = coord_this_source_ecl.lat.to("deg").value - coord_cutout_center.lat.to("deg").value

        # pixel offset from the cutout center in the downsampled space
        dx = dlon / pixel_size
        dy = dlat / pixel_size


        # convolve with the filter
        xoff = round(dx * oversample)
        yoff = round(dy * oversample)

        xlow = cutout_center[0] + xoff - kernel_off_x
        xhigh = cutout_center[0] + xoff + kernel_off_x
        ylow = cutout_center[1] + yoff - kernel_off_y
        yhigh = cutout_center[1] + yoff + kernel_off_y

        cutout_full[ylow:yhigh, xlow:xhigh] += (
            kernel * cosmos_full_phot[idx][3 + chnl]/1000
        ) # mJy

        refcat_srcs_XOFFs.append(xoff)
        refcat_srcs_YOFFs.append(yoff)
        refcat_srcs_names.append(tractor_ID)
            

    
    if plot:
        plt.figure(figsize=(6,6))
        plt.imshow(cutout_full)
        cbar = plt.colorbar()
        cbar.set_label('mJy / pixel')
        plt.title('Noiseless Image')
        plt.xlabel("X (oversampled pixel)")
        plt.ylabel("Y (oversampled pixel)")
        plt.show()

        
    return cutout_full, cutout_center, refcat_srcs_names, refcat_srcs_XOFFs, refcat_srcs_YOFFs
    
