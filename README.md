# deep-field-phot-on-maps
Perform photometry on galaxies on the deep-field sky maps for SPHEREx

__Inputs__: 
1. Catalog of galaxies for which we do photometry
    * source ID
    * position RA, DEC
    * (flux?, see problems)
2. Noise realization maps
    * A DC realization map
    * A read noise realization map
    * A true zodi map (for baseline calculation?)
    * A ZL realization map w/ photon noise
    * __A galaxy map__ (???)
3. Noise sigma maps
    * A DC sigma map
    * A read noise sigma map
    * A ZL photon noise sigma map (? optional)
4. PSF!!! likely per channel / varying with position?

---
__Procedure__:

(Per fiducial channel, assuming no galaxy map) 
1. Define a cutout, in the same space as the PSF; calculate ra, dec coords of the center and edges
2. Convolve flux with the PSF; place it into the cutout
    * Given flux, PSF
    * Assuming the LVF has been included in the flux
    * Also add nearby sources if there's any, transforming ra, dec to cutout coordinates. (Might have to deal with oversampling/downsampling source position...)
3. Pull out the sub-ZLmap; add onto the cutout
    * Given the cutout's ra, dec coords, pull out a sub-patch from the true ZL map of the same size.
    * Transform the sub-map into the PSF coord space 
    * Add the ZL map onto the cutout
4. Calculate a photon noise map due to ZL + source; add onto the cutout
5. Pull out the sub-DC & read noise realization maps; add onto the cutout
    * From noise realization maps (input #2), same calculation in step #3, pull out sub-patches. 
6. Sum up noise sigmas from step 3 & 4 & 5 --> total variance map
    * Photon noise sigma map = noiseless map from step #2
    * DC, read noise sigma maps from input #3
7. Subtract baseline (true zodi sub-map + mean noise map)
8. Do photometry:
    1. Tractor on: 
        * Given cutout to fit + source position (cutout space) + variance map + PSF model
        * Need to rewrite SPHERExTractorPSF class in tractor_utils.py. Currently we use this class to generate a PSF model given an x,y pixel position and an array number, directly passing it to Tractor as their unit flux model. But using sky maps, PSFs are likely some effective models incorporating multiple observations, also with better resolution (3''). 
    2. Tractor off:
        * Use do_psf_phot() function in quickcatalog.py with very few changes hopefully.
9. Output photometry results
10. Repeat step #1-9 for all fiducial channels.



---
__Problems__:
1. For the galaxy map, are spectral filters (LVFs) already incorporated? 
2. PSF model (which space? on the detector plane | ra, dec space | ecliptic coords)