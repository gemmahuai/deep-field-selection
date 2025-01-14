# deep-field-phot-on-maps
Perform photometry on galaxies on the deep-field sky maps for SPHEREx

__Inputs__: 
1. Catalog of galaxies for photometry
    * source ID
    * position RA, DEC
    * True flux density or surface brightness
2. Noise realization maps
    * A few dark current (DC) realization maps
    * A few read noise realization maps
    * A few photon noise realization maps (due to ZL + galaxies + stars)
    * A true zodiacal light (ZL) map (for baseline calculation)
    * A galaxy map
3. Noise sigma maps (calculated from the given noise realization maps)
    * A DC sigma map
    * A read noise sigma map
    * A ZL photon noise sigma map
    * A total noise sigma map
4. Effective __PSF__ per channel
    * Approximatedly, Ari's PSF class in the galaxy formation pipeline, averaged, symmetrical PSFs interpolated across channels; see ./scripts/PSF
    * Shortcut? Extract PSFs from stars.


---
__Procedure__:

(Per fiducial channel,)
1. Co-add all maps together for the deg^2 region around NEP.
    * Galaxy map
    * The true ZL map
    * A DC realization map
    * A read noise realization map
    * A photon noise realization map
2. Calculate a noise sigma map
    * Evaluate DC noise sigma from Shuang-Shuang's 10 realizations
    * Evaluate read noise sigma from 10 realiz
    * Evaluate photon noise sigma map from 10 realiz
    * Add noise sigmas in quadruture --> a noise sigma map for all components
3. Extract a cutout surrounding a given source (ra, dec)
4. Subtract the local baseline in the cutout
5. Do photometry:
    


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
    * In which coordinate system? detector pixels? ra,dec? ecliptic coords?
    * Averaged over multiple orientations?
3. Does the map contain pointing ditherings? 


# Deep Field Selection

## Goal
Come up with a color-magnitude cut that goes deeper selecting sources to be photometered in the deep field. While going deeper, we need to keep confusion under control as it has non-negligible effects in the deep field. 
(Jan 2025)

## Consideration
* Confusion from photometered sources
* Confusion from sub-threshold sources
* Varying sensitivity with ecliptic latitude → forget about this one, we’ll have a uniform cut for the entire deep field region…
* Maybe consider DGL effects 

## Simulation Outline

### Input data
* COSMOS 110k cross-matched catalog with SPHEREx RefCat v0.6 (csv files in https://caltech.app.box.com/file/1689176492841 )
* COSMOS 166k cross-matched high resolution SEDs (corrected ones https://caltech.app.box.com/folder/302074292575  ). Note: to extract SEDs for the cross-matched 110k sources, you want to load this file (https://caltech.app.box.com/file/1746943081834 ) and apply the column “xmatched_LS_110k” boolean array, selecting 110k from the full 166k catalog.
* Updated survey plan (depending on the simulation details, probably need to truncate down to the first 6 months or 2-3 months for faster 
computation)

### Procedure
1. __Isolated QuickCatalog Photometry + Redshift__
    1. Find a coordinate pair (ecliptic latitude + longitude, converted to RA, DEC, that gives median number of observations in the deep field) at which we place every source and do photometry. This could be done using a deep field hit map or a noise map from Shuang-Shuang.
    2. Use COSMOS cross-matched 110k high resolution SEDs and replace their positions with the calculated, fixed median coordinate. We probably need to choose a small sample of sources (a couple thousands) for faster simulation, but still occupying a large color-magnitude parameter space, depending on QC performance in the deep field and the chosen coordinates. 
    3. Feed QuickCatalog with the truncated 6 month survey plan (could be 2-3 months depending on the spectral coverage).
    4. Run QuickCatalog on each individual COSMOS source in isolation, with Tractor off. Save both output tables, SPHEREx_Catalog and Truth_Catalog for later calculation. Such isolated photometry will give us optimal photometry and redshift results as if there were no confusion.
    5. Modify secondary or primary photometry to scale it up to the full 2-year observation: for each measurement, we scale the flux error by sqrt of the ratio of the number of observations (say, sigma / sqrt( 2year / 6month)). And we extract true primary fluxes (already convolved with filters) from the Truth_Catalog and perturb them by Gaussian noise G(true_flux, sigma_scaled). 
    6. Run Photo-z on the QuickCatalog secondary photometry.
2. __Color - Magnitude Contours__
    1. Scatter plot of LS z magnitude (x) vs. LS z-w1 color (y) and color code it by redshift uncertainty.
    2. Identify some ‘contour’ color-mag cuts, similar to the full-sky preselection analysis.
    3. For each of the contour cut, perform Step #3 as follows.
3. __Tractor Photometry (with Confusion) + Redshift__
    1. For each of the contour color-mag cuts, select the corresponding ‘reference catalog’ from the cross-matched 110k to photometer; the remaining sources contribute to sub-threshold source confusion. 
    2. We offset the COSMOS coordinates to the deep field so that the central COSMOS field aligns with the chosen coordinates earlier in step 1.
    3. Choose a small sample of sources in the ref cat to photometer. We turn on the Tractor this time, photometering blended, close neighbors at the same time. Note that for each source to be photometered, we also photometer refcat neighbors within nearby 5 SPHEREx pixels, using the full 2-year survey plan. But we ignore sub-threshold sources in this run.
    4. To account for the un-photometered source confusion, we construct a confusion library from sub-threshold sources (among the 166k catalog) given a contour cut, using Gemma’s tools. 
    5. Gemma will perform the standard confusion injection into secondary photometry from step 3b output. 
    6. We run Photo-z on the resultant photometry, obtaining redshift estimates with confusion effects. (Finally, with each contour cut in step 2, we would have corresponding redshift measurements with confusion.)
4. __Finalize a Color-Mag Cut__
    1. We compare simulation results from step 3f and 1f, studying confusion impacts. For example, fractional loss of the number of sources in each science bin due to confusion; fractional loss vs magnitude, redshift bias in each bin due to confusion, redshift error bias, shift in the color-mag space, and so on.
    2. Pick a contour level that returns acceptable bias and loss in the full photometry simulation. 

### Problem
* What survey plan shall we use? Full 2 years or truncated 6 months / shorter? This might be a question for Jean as Jean is an expert at all infrastructures:) 
* Probably can get away with the 6 months survey plan. Could do a few sources with 2y for consistency checks if needed.
What COSMOS catalog shall we use? Cross-matched 110k catalog with the SPHEREx RefCat v0.6, or the original 166k cross-matched to RC with no position/magnitude quality cuts? I think we should use the 166k catalog matched to SPHEREx RC with no further quality cuts since it’s deeper.



      
