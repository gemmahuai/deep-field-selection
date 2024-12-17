# deep-field-phot-on-maps
Perform photometry on galaxies on the deep-field sky maps for SPHEREx

__Inputs__: 
1. Catalog of galaxies for photometry
    * source ID
    * position RA, DEC
    * (brightness? TBD, need further clarification.)
2. Noise realization maps
    * A DC realization map
    * A read noise realization map
    * A true zodi light map (for baseline calculation?)
    * A ZL realization map w/ photon noise? (not necessary)
    * __A galaxy map__ (? unclear, needs clarification.)
3. Noise sigma maps
    * A DC sigma map
    * A read noise sigma map
    * A ZL photon noise sigma map (optional)
4. __PSF__ potentially per channel and varying with position?

---
__Procedure__:

(Per fiducial channel, assuming no galaxy map) 
1. Define a cutout surrounding a given source.
    * In the same coordinate space as the PSFs;
    * Calculate the ra, dec coords of the cutout's center and edges.
3. Convolve flux with PSF and populate the cutout
    * Given source flux, PSF;
    * Assuming the LVF effects have been included;
    * Place the primary source to photometer at the cutout center;;
    * Add nearby sources if there's any, by transforming their ra, dec coords into the cutout coordinates. (Might have to deal with oversampled/downsampled source position...)
4. Extract and add sub-ZL map to the cutout
    * Using the RA and DEC coordinates of the cutout, extract a sub-patch from the true zodiacal light (ZL) map of the same size as the cutout.
    * Transform the sub-map into the PSF coord space 
    * Add the ZL map to the cutout
5. Calculate a photon Poisson noise map due to ZL + source; add to the cutout
6. Extract and add sub-DC and read noise maps
    * Extract sub-patches from the DC and read noise realization maps (input #2), following the same cutout extraction process as in step #3.
    * Add these maps to the cutout.
7. Calculate a total noise variance map from noise sigma maps in step 3 & 4 & 5
    * Photon noise variance map = noiseless cutout from step #2
    * DC, read noise sigma maps from input #3
8. Subtract baseline (true ZL + mean noise map)
9. Do photometry:
    1. Tractor on: 
        * Inputs: cutout to fit, source position (cutout space), variance map, PSF model
        * Need to rewrite SPHERExTractorPSF class in tractor_utils.py. Currently we use this class to generate a PSF model given an x,y pixel position and an array number, directly passing it to Tractor as their unit flux model. But using sky maps, PSFs are likely some effective models incorporating multiple observations, also with higher resolution (3''). 
    2. Tractor off:
        * Use do_psf_phot() function in quickcatalog.py with very few changes hopefully.
10. Save photometry results
11. Repeat step #1-9 for all fiducial channels.



---
__Problems__:
1. For the galaxy map, are spectral filters (LVFs) already incorporated?
2. PSF model 
    * Oversampling / downsampling?
    * Resolution?
    * In which coordinate system? detector plane? ra,dec? ecliptic coords?
    * Averaged over multiple orientations?
    * Naive question - any conversion factor from RA, DEC coords to detector plane coords (likely in which we do photometry)?
3. Does the map contain pointing ditherings? 
      
