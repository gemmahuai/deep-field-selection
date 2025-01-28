"""
Module of `tractor`-related utility functions and classes for use in common

"""

# import modules
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tractor
from astrometry.util.miscutils import get_overlapping_region
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage
from skimage.transform import downscale_local_mean
from SPHEREx_InstrumentSimulator.psf import PSF
from tractor import ConstantSky, Flux, LinearPhotoCal, NullWCS, PixPos, PointSource
from tractor.galaxy import GalaxyShape
from tractor.sersic import SersicGalaxy, SersicIndex

import inspect

from matplotlib import pyplot as plt


# FIXME: `frame_width` and `keep_frame_sources` are not used in the function
def tractor_fit_spherex(
    spherex_cutout,
    cutout_wcs,
    spherex_variance_cutout,
    source_list,
    array_number,
    cutout_center,
    sp_inst,
    frame_width=2,
    keep_frame_sources=False,
    return_fit=False,
    in_table=True,
):
    """
    Compute forced photometry with Tractor of sources in a SPHEREx cutout.

    Parameters
    ----------
    spherex_cutout : `~astropy.nddata.Cutout2D` image
        A small (~10x10 pixel) subregion of a SPHEREx image. WCS info should be
        retained.

    cutout_wcs : WCS object
        WCS for the cutout.

    spherex_variance_cutout : 2D cutout image
        Variance for the same small (~10x10 pixel) subregion of a SPHEREx image.

    source_list : `~astropy.table.Table` or dict of list-like
        Sources to fit. Table includes: ID, RA, DEC, SersicRadius (arcsec),
        SersicAB, SersicNu, SersicPhi (deg), X, Y. For sources treated as point
        sources, set Sersic Radius to 0 or a negative number. If X, Y are not
        NaN, they are treated as the source coordinates and the WCS is
        bypassed.

    array : {1, 2, 3, 4, 5, 6}
        SPHEREx array number (1-6)

    cutout_center : tuple
        Position (x,y) in pixels of the cutout on full SPHEREx array. Used to
        query the PSF.

    sp_inst : `~SPHEREx_InstrumentSimulator.instrument.Instrument`
        SPHEREx simulator instrument model. Contains the PSF.

    frame_width : SPHEREx pixels, optional
        If sources fall within this number of pixels of the cutout edge, they
        are included in the fit but their photometry is not considered valid,
        and is not returned by default. Default is 2.

    keep_frame_sources : bool
        If `True`, photometry of sources in the outer frame of the cutout is
        returned. Default is `False`.

    return_fit : bool, optional
        If `True`, fit model and residual image are returned. Default is `False`.

    in_table : bool, optional
        If `True`, the output is an `astropy.table.Table`. Otherwise, a ``N``
        by ``3`` array, where ``N`` is the number of sources and ``3`` is in
        the order of ID, flux, and flux error. Default is `True`.

    Returns
    -------
    flux/fluxerr : `~astropy.table.Table` or `~numpy.ndarray`
        Fluxes/errors of the sources provided in `source_list`.

    fit : FITS structure
        If return_fit=True, returns the best-fit model and residual image

    """
    from astropy.coordinates import SkyCoord

    spherex_psf = SPHERExTractorPSF(
        sp_inst.PSF, array_number, xshift=cutout_center[0], yshift=cutout_center[1]
    )

    tractor_source_list = []
    _x0 = source_list["X"][0]
    for _x, _y, _ra, _dec, _rs, _phi, _ab, _nu in zip(
        source_list["X"],
        source_list["Y"],
        source_list["RA"],
        source_list["DEC"],
        source_list["SersicRadius"],
        source_list["SersicPhi"],
        source_list["SersicAB"],
        source_list["SersicNu"],
    ):
        _flux = Flux(np.random.uniform(high=5000))
        if _rs <= 0:
            # FIXME: is it really _x0 in the line below (not _x)?
            if np.isnan(_x0):  # Start off with point source
                pixloc = cutout_wcs.world_to_pixel(SkyCoord(_ra, _dec, unit="deg"))
                _src = PointSource(PixPos(*pixloc), _flux)
            else:
                # Use the provided X,Y coordinates of the source
                _src = PointSource(PixPos(_x, _y), _flux)

        # Here and below we should also allow the option of extended sources with
        # X,Y provided..not yet done
        else:
            # Start off with extended source
            pixloc = cutout_wcs.world_to_pixel(SkyCoord(_ra, _dec, unit="deg"))
            # FIXME: is the line below correct? I think `cutout_wcs`, not `cutout`
            angle = np.arccos(cutout.wcs.wcs.pc[0, 0]) * 180 / np.pi
            _src = SersicGalaxy(
                PixPos(*pixloc),
                _flux,
                GalaxyShape(_rs, _ab, _phi - angle),
                SersicIndex(_nu),
            )

        # Freeze position for all sources (pure forced photometry)
        _src.freezeAllRecursive()
        _src.thawParam("brightness")
        tractor_source_list.append(_src)

    tim = tractor.Image(
        data=spherex_cutout,
        invvar=1 / spherex_variance_cutout,
        psf=spherex_psf,
        wcs=NullWCS(pixscale=6.2),  # FIXME: fix this hard-coded value
        photocal=LinearPhotoCal(1.0),
        sky=ConstantSky(0.0),
    )

    tim.freezeAllRecursive()  # Image parameters fixed

    # Build Tractor object
    trac_spherex = tractor.Tractor([tim], tractor_source_list)

    # Optimize the fit with Tractor, positions frozen
    # start_time = timeit.default_timer()
    optres = trac_spherex.optimize_forced_photometry(variance=True)

    if return_fit:
        image_hdu_mod_final = fits.PrimaryHDU(trac_spherex.getModelImage(0))
        image_hdu_chi_final = fits.ImageHDU(trac_spherex.getChiImage(0))
        hdul = fits.HDUList([image_hdu_mod_final, image_hdu_chi_final])

    # Build the output table with fluxes and errors
    out = []
    for ii in range(0, len(tractor_source_list)):
        out.append(
            [
                float(source_list["ID"][ii]),
                tractor_source_list[ii].getParams()[0],
                1 / np.sqrt(optres.IV[ii]),
            ]
        )
        # NOTE: Can't we save `tractor_source_list[ii].getParams()[0]` as,
        # e.g., `fluxlist` in the previous for loop?

    if in_table:
        fluxres = Table(
            rows=out, names=("ID", "Flux", "Fluxerr"), meta={"name": "tractor_result"}
        )
    else:
        fluxres = np.atleast_2d(out)

    if return_fit:
        return fluxres, hdul
    else:
        return fluxres


def lanczos_filter(order, x):
    """Lanczos kernel function

    Parameters
    ----------
    order : int
        Positive integer determining the size of the kernel.
    x : array_like
        Abscissae of the kernel.

    Returns
    -------
    out : array_like
        Lanczos kernel.

    Notes
    -----
    This is a simplified copy of the function, `lanczos_filter`, in
    `util.miscutils` module of `astrometry.net
    <https://github.com/dstndstn/astrometry.net/>`_.

    """

    x = np.atleast_1d(x)
    nz = np.logical_and(x != 0, np.logical_and(x < order, x > -order))
    nz = np.flatnonzero(nz)

    out = np.zeros(x.shape, dtype=float)

    pinz = np.pi * x.flat[nz]
    out.flat[nz] = order * np.sin(pinz) * np.sin(pinz / order) / (pinz**2)
    out[x == 0] = 1.0

    return out


def lanczos_shift_image(img, dx, dy):
    """Shift the given image with Lanczos resampling

    Parameters
    ----------
    img : array_like
        Input 2-dimensional image to shift.
    dx, dy : float
        Amount of shift along the x- and y-axis, respectively (axis ``1`` and
        ``0`` in default Numpy ndarray, respectively).
    order : int, optional
        Order of Lanczos filter. Default is 3.

    Returns
    -------
    outimg : array_like
        Shifted 2-dimensional image

    Notes
    -----
    This is a simplified copy of the function, `lanczos_shift_image`, in `psf`
    module of `the Tractor <https://github.com/dstndstn/tractor/>`_.

    """

    # "order" of Lanczos kernel
    l_order = 3
    _a = np.arange(-l_order, l_order + 1)
    l_x = lanczos_filter(l_order, _a + dx)
    l_y = lanczos_filter(l_order, _a + dy)

    l_x /= l_x.sum()
    l_y /= l_y.sum()

    sx = ndimage.correlate1d(img, l_x, axis=1, mode="constant")
    outimg = ndimage.correlate1d(sx, l_y, axis=0, mode="constant")

    return outimg


def _subimg_center(height, width, subimg):
    subimg_h, subimg_w = subimg.shape

    if height >= subimg_h:
        pad_y0 = (height - subimg_h) // 2
        subimg_cy = pad_y0 + np.floor((subimg_h - 1.0) / 2)
    else:
        pad_y0 = 0
        cut = (subimg_h - height) // 2
        subimg = subimg[cut : cut + height, :]
        subimg_cy = np.floor((subimg_h - 1.0) / 2) - cut

    if width >= subimg_w:
        pad_x0 = (width - subimg_w) // 2
        subimg_cx = pad_x0 + np.floor((subimg_w - 1.0) / 2)
    else:
        pad_x0 = 0
        cut = (subimg_w - width) // 2
        subimg = subimg[:, cut : cut + width]
        subimg_cx = np.floor((subimg_w - 1.0) / 2) - cut

    subimg_h, subimg_w = subimg.shape

    pad = np.zeros((height, width), subimg.dtype)
    pad[pad_y0 : pad_y0 + subimg_h, pad_x0 : pad_x0 + subimg_w] = subimg

    return pad, subimg_cx, subimg_cy

## Gemma modified 
def _get_hires_psf(px, py, xshift, yshift, psf, array, oversample=5, gemma_1st_order_sim=False):
    if gemma_1st_order_sim==False:
        px_hires = (px + xshift + 0.5) * oversample - 0.5
        py_hires = (py + yshift + 0.5) * oversample - 0.5
        chunk_size_hires = int(2048 * oversample / 3)
        chunk_ix = min(px_hires // chunk_size_hires, 2.0)
        chunk_iy = min(py_hires // chunk_size_hires, 2.0)
        psf_region_ref_x = chunk_ix * (2048 / 2)
        psf_region_ref_y = chunk_iy * (2048 / 2)

    elif gemma_1st_order_sim==True:
        # use the input xshift, yshift position to extract the exact psf, not dividing into chunks
        psf_region_ref_x = xshift
        psf_region_ref_y = yshift

    return psf(psf_region_ref_x, psf_region_ref_y, array=array).T


class SPHERExTractorPSF(tractor.ducks.Params, tractor.ducks.ImageCalibration):
    """Tractor PSF object based on the SPHEREx instrument simulator

    Parameters
    ----------
    psf : `~SPHEREx_InstrumentSimulator.psf.PSF`
        Instantiated SPHEREx PSF object.
    array : {1, 2, 3, 4, 5, 6}
        ID of the SPHEREx detector array. Should be one of 1, 2, 3, 4, 5, and
        6.
    xshift, yshift : int, optional
        x- and y-axis shift in the SPHEREx detector pixel coordinates. Used
        when get the PSF from `inst` in a cropped part of SPHEREx observation
        image. Default is 0.
    wavelength: float, optional
        if given in micron, then generate a perfect Gaussian PSF based on the wavelenth given. 

    Notes
    -----
    Note that this currently does not work with extended sources.

    """

    # instantiated SPHEREx PSF object
    _psf = None
    # ID of the SPHEREx detector array
    array = None
    # x-axis shift in the SPHEREx detector pixel coordinates
    xshift = 0
    # y-axis shift in the SPHEREx detector pixel coordinates
    yshift = 0
    # radius of PSF in the unit of pixels
    radius = 1

    def __init__(self, spherex_psf, array, xshift=0, yshift=0, oversample=5, wavelength=None):
        if not isinstance(spherex_psf, PSF):
            raise TypeError(
                "input parameter `spherex_psf` not a "
                "SPHEREx_InstrumentSimulator.psf.PSF object"
            )
        self._psf = spherex_psf
        self.array = int(array)
        self.xshift = xshift
        self.yshift = yshift
        # gemma add
        self.oversample = oversample
        if wavelength:
            self.wave = wavelength
        else:
            self.wave = 0


## Gemma changed the oversampling factor on 07/24/2024  5 -> 10
        # set the shape of PSF taking into account the oversampling
        self.psf_shape = [
            int(x / self.oversample) for x in self._psf.get_psf(0, 0, array=self.array).shape
        ]
        # set the radius of PSF taking into account the oversampling
        self.radius = max(self.psf_shape) * 0.5 / self.oversample

        return

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"{self.__class__.__name__}: array={self.array}, "
            f"xshift={self.xshift}, yshift={self.yshift}"
        )

    @property
    def shape(self):
        return self.psf_shape

    def copy(self):
        return self.__class__(
            self._psf, self.array, xshift=self.xshift, yshift=self.yshift
        )

    def getRadius(self):
        return self.radius
    
    def GaussianPSF(self, x, y, x0, y0):
        # approximate sphx wavelengths
        sphx_wl = np.linspace(0.75, 5., 102)
        # Gaussian PSF sigma range
        sphx_sigma = np.linspace(0.7, 1.8, 102)
        sphx_sigma_this_wl = np.interp(self.wave, sphx_wl, sphx_sigma)
        # print("Gaussian sigma in utils = ", sphx_sigma_this_wl)
        g = np.exp( - ((x-x0)**2+(y-y0)**2) / (2*sphx_sigma_this_wl**2))
        g /= np.sum(g)
        return(g)
        

    def getPointSourcePatch(
        self, px, py, minval=0, radius=None, modelMask=None, **kwargs
    ):
        """Return a patch of PSF of a point source at the given pixel coordinates

        Notes
        -----
        This obtains high-resolution, 5-times oversampled PSF from SPHEREx
        instrument simulator, resamples it for sub-pixel shift, and downscales
        it by a factor of 5 to get the detector-pixel-scaled PSF.

        Parameters
        ----------
        px, py : float
            x- and y-axis pixel coordinate of the point source, respectively.
        minval : float, optional
        radius : float, optional
        modelMask : optional

        Returns
        -------
        patch : `tractor.patch.Patch`
            Patch of PSF

        """

        if radius is not None:
            raise NotImplementedError("input parameter `radius` not " "supported yet")
        if modelMask is not None:
            raise NotImplementedError(
                "input parameter `modelMask` not " "supported yet"
            )

        # # get 5-times high-resolution PSF
        # hires_psf = _get_hires_psf(
        #     px, py, self.xshift, self.yshift, self._psf.psf, self.array, oversample=self.oversample
        # )
        # gemma modified
        hires_psf = _get_hires_psf(
            px, py, self.xshift, self.yshift, self._psf.psf, self.array, oversample=self.oversample, gemma_1st_order_sim=False
        )
        if self.wave!=0:
            x = np.arange(self.psf_shape[0])
            y = np.arange(self.psf_shape[1])
            (x, y) = np.meshgrid(x,y)
            hires_psf = self.GaussianPSF(x, y, self.psf_shape[0]//2-1, self.psf_shape[1]//2-1)
            # plt.figure()
            # plt.imshow(hires_psf)
            # plt.show()

        # pixel coordinates of the left-lower corner of "first" pixel of
        # full-scale PSF in the scene with resolution same to that of
        # full-scale PSF
        # For simplicity, assume that the "left-lower" corner of a pixel has
        # integer index in this scene.
        # We use a complex equation for displacement from the given pixel
        # coordinates (`px`, `py`), because the centroid of PSF is at the
        # center of left-lower pixel of the exact center of image, if the image
        # is even-shaped.
        x0_hires = (px + 0.5) * self.oversample - (hires_psf.shape[1] - 1) // 2 - 0.5
        y0_hires = (py + 0.5) * self.oversample - (hires_psf.shape[0] - 1) // 2 - 0.5

        # integer parts of `x0_hires` and `y0_hires`
        ix0_hires = round(x0_hires)
        iy0_hires = round(y0_hires)

        # float part of `x0_hires` and `y0_hires`
        dx_hires = x0_hires - ix0_hires
        dy_hires = y0_hires - iy0_hires

        # subpixel shift of the hires-scale PSF
        hires_psf = lanczos_shift_image(hires_psf, dx_hires, dy_hires) # this is fine, normal Neff
        # print("1st operation shape: ", hires_psf.shape)
        # print(hires_psf.max(), np.sum(hires_psf))
        # print(1 / np.sum(hires_psf**2))
        # test = downscale_local_mean(hires_psf, (self.oversample, self.oversample))
        # test = test / np.sum(test)
        # print(test.shape, 1 / np.sum(test**2))

        # pad zero-value pixels around the hires-scale PSF to align it to the
        # real resolution scene
        npad_left = ix0_hires % self.oversample
        hires_psf = np.c_[np.zeros((hires_psf.shape[0], npad_left)), hires_psf]

        npad_bottom = iy0_hires % self.oversample
        hires_psf = np.r_[np.zeros((npad_bottom, hires_psf.shape[1])), hires_psf] # Neff already changed
        # print("\n2nd operation shape: ", hires_psf.shape)
        # print(hires_psf.max(), np.sum(hires_psf))
        # print(1 / np.sum(hires_psf**2))
        # test = downscale_local_mean(hires_psf, (self.oversample, self.oversample))
        # test = test / np.sum(test)
        # print(test.shape, 1 / np.sum(test**2))

        if nover := hires_psf.shape[1] % self.oversample > 0:
            npad_right = self.oversample - nover
            hires_psf = np.c_[hires_psf, np.zeros((hires_psf.shape[0], npad_right))]

        if nover := hires_psf.shape[0] % self.oversample > 0:
            npad_top = self.oversample - nover
            hires_psf = np.r_[hires_psf, np.zeros((npad_top, hires_psf.shape[1]))]

        # pixel coordinates of the "first" pixel of padded hires-scale PSF
        # in the scene with resolution same to the hires-scale PSF
        x0_hires = ix0_hires - npad_left
        y0_hires = iy0_hires - npad_bottom

        # They should be multiple of `self.downsampling_factor`.
        assert x0_hires % self.oversample == 0
        assert y0_hires % self.oversample == 0

        # downsampling
        outimg = downscale_local_mean(hires_psf, (self.oversample, self.oversample))
        outimg /= outimg.sum()
        ix0 = x0_hires // self.oversample
        iy0 = y0_hires // self.oversample


        # # test = tractor.patch.Patch(ix0, iy0, outimg).getImage()
        # # test = outimg.copy() # already smaller Neff
        # print(inspect.getfile(tractor.patch.Patch))
        # plt.imshow(test)
        # plt.colorbar()
        # plt.show()
        # print(test.shape)
        # print("Sum = ", np.sum(test))
        # print("Neff = ", 1 / np.sum(test**2))

        return tractor.patch.Patch(ix0, iy0, outimg)

    def getFourierTransformSize(self, radius):
        """Calculate the power-of-two size next to the given `radius`."""
        size = 2 ** int(np.ceil(np.log2(2 * radius)))
        return size

    def getFourierTransform(self, px, py, radius):
        """Fourier transform of PSF, with the next-power-of-2 size up from `radius`.

        Parameters
        ----------
        px : float
        py : float
        radius : float

        Returns
        -------
        p : array_like
            Fourier transform of PSF.
        (x0, y0) : tuple of floats
            Pixel coordinates of the PSF center in the PSF subimage.
        (imh, imw) : tuple of ints
            Shape of the padded PSF image.
        (v, w) : tuple of floats
            v=np.fft.rfftfreq(imw), w=np.fft.fftfreq(imh)

        """

        size = self.getFourierTransformSize(radius)

        pad, cx, cy = self._padInImage(size, size, px, py)

        p = np.fft.rfft2(pad)
        p = p.astype(np.complex64)

        pad_height, pad_width = pad.shape

        v = np.fft.rfftfreq(pad_width)
        w = np.fft.fftfreq(pad_height)

        return p, (cx, cy), (pad_height, pad_width), (v, w)

    def _padInImage(self, height, width, px, py, img=None):
        """Embed PSF image into shape (`height`, `width`).

        Parameters
        ----------
        height, width : int
            Embed PSF image into a larger or smaller image of shape, (`height`,
            `width`).
        px, py : float
        img : array_like, optional

        Returns
        -------
        pad : array_like
        cx : int
            *x*-axis coordinate of the PSF center in `pad`.
        cy : int
            *y*-axis coordinate of the PSF center in `pad`.

        Notes
        -----
        Return pixel coordinates follow the FITS standard with 0-based indices.

        """

        if img is None:
            point_source_patch = self.getPointSourcePatch(px, py)
            subimg = point_source_patch.patch
        else:
            subimg = img.copy()

        # Center of PSF is at the "exact" center of the image, `pad`, if the
        # image is odd-shaped.  It is at the center of lower-left pixel (with
        # origin = lower) of the exact center of the image, if the image is
        # even-shaped.

        return _subimg_center(height, width, subimg)

    def getFourierTransform5xHiRes(self, px, py, radius):
        ### Gemma changed from 5 to 10
        size = self.getFourierTransformSize(self.oversample * radius)

        pad, cx, cy = self._padInImage5xHiRes(size, size, px, py)

        p = np.fft.rfft2(pad)
        p = p.astype(np.complex64)

        pad_height, pad_width = pad.shape

        v = np.fft.rfftfreq(pad_width)
        w = np.fft.fftfreq(pad_height)

        return p, (cx, cy), (pad_height, pad_width), (v, w)

    def _padInImage5xHiRes(self, height, width, px, py, img=None):
        if img is None:
            # get 5-times high-resolution PSF
            subimg = _get_hires_psf(
                px,
                py,
                self.xshift,
                self.yshift,
                self._psf.get_psf,
                self.array,
                oversample=self.oversample,
                gemma_1st_order_sim=True
            ) # Gemma changed from oversample, added the last argument
            subimg /= subimg.sum()
        else:
            subimg = img.copy()

        # Center of PSF is at the "exact" center of the image, `pad`, if the
        # image is odd-shaped.  It is at the center of lower-left pixel (with
        # origin = lower) of the exact center of the image, if the image is
        # even-shaped.
        return _subimg_center(height, width, subimg)


def tractor_ProfileGalaxy_getUnitFluxModelPatch_new(
    self,
    img,
    px=None,
    py=None,
    minval=0.0,
    modelMask=None,
    force_halfsize=None,
    oversample = 5, # gemma added
    **kwargs,
):

    if px is None or py is None:
        px, py = img.getWcs().positionToPixel(self.getPosition(), self)

    psf = img.getPsf()

    if hasattr(psf, "getFourierTransform5xHiRes"):
        if modelMask is None:
            # choose the patch size
            halfsize = self._getUnitFluxPatchSize(img, px=px, py=py, minval=minval)
            # find overlapping pixels to render
            outx, inx = get_overlapping_region(
                int(np.floor(px - halfsize)),
                int(np.ceil(px + halfsize + 1)),
                0,
                img.getWidth(),
            )
            outy, iny = get_overlapping_region(
                int(np.floor(py - halfsize)),
                int(np.ceil(py + halfsize + 1)),
                0,
                img.getHeight(),
            )
            if (not inx) or (not iny):
                # no overlap
                return None
            x0, x1 = outx.start, outx.stop
            y0, y1 = outy.start, outy.stop
        else:
            x0, y0 = modelMask.x0, modelMask.y0

        # FFT in 5-times larger scale -----------------------------------------
        if modelMask is None:
            # Avoid huge galaxies
            # Let `halfsize` be smaller than or equal to the longer side of
            # image.
            imsz = max(img.shape)
            halfsize = min(halfsize, imsz)
        else:
            # modelMask sets the sizes
            mh, mw = modelMask.shape
            x1 = x0 + mw
            y1 = y0 + mw

            halfsize = max(mh / 2.0, mw / 2.0)
            # How far from the source center to furthest modelMask edge?
            # FIXME -- add 1 for Lanczos margin?
            halfsize = max(
                halfsize,
                max(max(1 + px - x0, 1 + x1 - px), max(1 + py - y0, 1 + y1 - py)),
            )
            psfh, psfw = psf.shape
            halfsize = max(halfsize, max(psfw / 2.0, psfh / 2.0))
            if force_halfsize is not None:
                halfsize = force_halfsize

            # Is the source center outside the modelMask?
            sourceOut = px < x0 or px > x1 - 1 or py < y0 or py > y1 - 1

            if sourceOut:
                # FFT, modelMask, source is outside the box.
                neardx, neardy = 0.0, 0.0
                if px < x0:
                    neardx = x0 - px
                if px > x1:
                    neardx = px - x1
                if py < y0:
                    neardy = y0 - py
                if py > y1:
                    neardy = py - y1
                nearest = np.hypot(neardx, neardy)
                if nearest > self.getRadius():
                    return None

                # How far is the furthest point from the source center?
                farw = max(abs(x0 - px), abs(x1 - px))
                farh = max(abs(y0 - py), abs(y1 - py))
                bigx0 = int(np.floor(px - farw))
                bigx1 = int(np.ceil(px + farw))
                bigy0 = int(np.floor(py - farh))
                bigy1 = int(np.ceil(py + farh))
                bigw = 1 + bigx1 - bigx0
                bigh = 1 + bigy1 - bigy0
                boffx = x0 - bigx0
                boffy = y0 - bigy0
                assert bigw >= mw
                assert bigh >= mh
                assert boffx >= 0
                assert boffy >= 0
                bigMask = np.zeros((bigh, bigw), bool)
                if modelMask.mask is not None:
                    bigMask[boffy : boffy + mh, boffx : boffx + mw] = modelMask.mask
                else:
                    bigMask[boffy : boffy + mh, boffx : boffx + mw] = True
                bigMask = tractor.patch.ModelMask(bigx0, bigy0, bigMask)
                # print('Recursing:', self, ':', (mh,mw), 'to', (bigh,bigw))
                bigmodel = self._realGetUnitFluxModelPatch(
                    img, px, py, minval, modelMask=bigMask
                )
                return tractor.patch.Patch(
                    x0, y0, bigmodel.patch[boffy : boffy + mh, boffx : boffx + mw]
                )

        # get the FT of 5-times high-resolution PSF
        # (pxx5, pyx5) -- pixel coordinates in 5-times scaled-up image
        pxx5 = (px + 0.5) * oversample - 0.5
        pyx5 = (py + 0.5) * oversample - 0.5

        # P -- FT of PSF
        # (cx, cy) -- pixel coordinates of the PSF center
        # (pH, pW) -- height and width of PSF
        # (v, w) -- DFT frequencies long x- and y-axes
        P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform5xHiRes(px, py, halfsize)

        # (dxx5, dyx5) -- pixel coordinates of the first pixel of PSF in
        #                 5-times scaled-up image
        dxx5 = pxx5 - cx
        dyx5 = pyx5 - cy
        if modelMask is not None:
            # The Patch we return *must* have this origin.
            ix0 = x0
            iy0 = y0
            # integer portions of (ix0, iy0) in 5-times scaled-up image
            # ix0x5 = int(np.round((ix0 + 0.5) * oversample - 2.5))
            # iy0x5 = int(np.round((iy0 + 0.5) * oversample - 2.5))
            ix0x5 = int(np.round((ix0 + 0.5) * oversample - 0.5 * oversample))
            iy0x5 = int(np.round((iy0 + 0.5) * oversample - 0.5 * oversample))
            # difference that we have to handle by shifting the 5-times
            # scaled-up model image
            muxx5 = dxx5 - ix0x5
            muyx5 = dyx5 - iy0x5
            # We will handle the integer portion by computing a shifted image
            # and copying it into the result.
            sxx5 = int(np.round(muxx5))
            syx5 = int(np.round(muyx5))
            # the subpixel portion will be handled with a Lanczos interpolation
            muxx5 -= sxx5
            muyx5 -= syx5
        else:
            # integer portions of pixel coordinates
            ix0x5 = int(np.round(dxx5))
            iy0x5 = int(np.round(dyx5))
            # subpixel portions of pixel coordinates
            muxx5 = dxx5 - ix0x5
            muyx5 = dyx5 - iy0x5
            # pixel coordinates of the first pixel of downscaled PSF in real
            # image
            ix0 = ix0x5 // oversample
            iy0 = iy0x5 // oversample
            #
            sxx5 = syx5 = 0

        # At this point, mux,muy are both in [-0.5, 0.5].
        assert abs(muxx5) <= 0.5
        assert abs(muyx5) <= 0.5

        wcs_ = tractor.NullWCS(pixscale=img.getWcs().pixscale / oversample)
        img_ = img.__class__(
            data=img.data,
            inverr=img.inverr,
            wcs=wcs_,
            psf=img.psf,
            sky=img.sky,
            photocal=img.photocal,
            name=img.name,
            time=img.time,
        )
        fftmix = self._getShearedProfile(img_, px, py)

        if fftmix is not None:
            Fsum = fftmix.getFourierTransform(v, w, zero_mean=True)
            G = np.fft.irfft2(Fsum * P)

            assert G.shape == (pH, pW)

            # Lanczos-3 interpolation in the same way we do for pixelized PSFs.
            from tractor.psf import lanczos_shift_image

            G = G.astype(np.float32)
            lanczos_shift_image(G, muxx5, muyx5, inplace=True)
        else:
            G = np.zeros((pH, pW), np.float32)

        # downscaling after padding zero-value pixels
        Gpad = np.zeros(
            5 * np.ceil([s / oversample + 1 for s in G.shape]).astype(int), dtype=G.dtype
        )

        pad_x0 = ix0x5 % oversample + (sxx5 % oversample)
        pad_y0 = iy0x5 % oversample + (syx5 % oversample)
        Gpad[pad_y0 : pad_y0 + G.shape[0], pad_x0 : pad_x0 + G.shape[1]] = G

        G = downscale_local_mean(Gpad, (oversample, oversample))
        G /= G.sum()

        if modelMask is not None:
            gh, gw = G.shape
            mh, mw = modelMask.shape

            sx = sxx5 // oversample
            sy = syx5 // oversample

            if gh % 2 == 0 and mh % 2 == 1:
                sx += 1
                ix0 -= 1
            if gw % 2 == 0 and mw % 2 == 1:
                sy += 1
                iy0 -= 1

            if sx != 0 or sy != 0:
                yi, yo = get_overlapping_region(-sy, -sy + mh - 1, 0, gh - 1)
                xi, xo = get_overlapping_region(-sx, -sx + mw - 1, 0, gw - 1)
                # shifted
                # FIXME -- are yo,xo always the whole image?  If so, optimize
                shG = np.zeros((mh, mw), G.dtype)
                shG[yo, xo] = G[yi, xi]

                G = shG

            if gh > mh or gw > mw:
                G = G[:mh, :mw]

            assert G.shape == modelMask.shape
        else:
            # Clip down to suggested "halfsize"
            if x0 > ix0:
                G = G[:, x0 - ix0 :]
                ix0 = x0
            if y0 > iy0:
                G = G[y0 - iy0 :, :]
                iy0 = y0
            gh, gw = G.shape
            if gw + ix0 > x1:
                G = G[:, : x1 - ix0]
            if gh + iy0 > y1:
                G = G[: y1 - iy0, :]

        patch = tractor.Patch(ix0, iy0, G)

    else:
        patch = self._realGetUnitFluxModelPatch(
            img, px, py, minval, modelMask=modelMask, **kwargs
        )

    if patch is not None and modelMask is not None:
        assert patch.shape == modelMask.shape

    return patch
