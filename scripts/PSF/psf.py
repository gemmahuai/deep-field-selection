# Adapted from Ari's script in the SPHEREx-L4-Galaxy-Formation pipeline.
# Used for performing photometry on the deep field maps

import numpy as np
from psfparams import PSFParams
import os
from astropy import wcs
import astropy.units as u
from spherex_parameters import load_spherex_parameters,\
    INSTRUMENT_PARAM_FILEPATH,SPS_PARAM_FILEPATH


class PSF:
    def __init__(self, psfparams='default', psf_filename='/Users/gemmahuai/anaconda3/envs/spherexsim_tractor/lib/python3.11/site-packages/SPHEREx_Simulator_Data/psf/simulated_PSF_database_centered_v3_og.fits'):
        '''
        This is a class to amalgamate PSF estimates, average them within each
        spectral channel and symmetrize them. The PSF dataset is coordinated
        with the SkySimulator. This class is used in ConvCube and is not intended
        for an end user. The class creates PSFs only for channels that have
        PSF estimates. Coordinated lists of channels, PSFs and wavelengths are
        given in the attributes .ch, .psf and .wl. The platescale is in 
        .platescale in units of arcsec.

        Parameters
        ----------
        psfparams [PSFParams object]: Defines the PSF data products. Defaults
            to 'default', which uses PSFParams().
        '''
        if psfparams == 'default':
            self.psfparams = PSFParams()
        else:
            self.psfparams = psfparams

        self.psf_fn = psf_filename
        
        ## mapmaker relevant params
        self.nArrays = 6 # number of arrays
        self.inst_params = load_spherex_parameters(INSTRUMENT_PARAM_FILEPATH)
        chs_per_band = 17
        self.chs_per_band = chs_per_band
        # chs_per_band ignore the edge detectors that form only fractional
        # channels. Incorporate those in the EBL channel definitions.
        self.chs_per_array = chs_per_band + 2
        # Parameters for handling PSF model:
        self.npixPerSide = 2048 # detector pixels on a side of the array
        self.mmPerPix = 36.9/2048 # mm corresponding to npixPerSide
        self.gap_size_pixels = 22.4*60/6.2 # gap size between arrays in pixels


        def get_psf_products():
            print("\nget psf estimate")
            psf_estimates = self.__get_psf_estimates()
            print("\nget psf ch")
            self._psf_ch = self.__get_psf_ch(psf_estimates)
            print("\nget psf data")
            chs,psfs_asym = self.__get_psf_data(psf_estimates)
            self.ch = chs
            self._psf = psfs_asym
            platescale = psf_estimates.psf_platescale # arcsec
            self.platescale = platescale
            # oversampling factor for PSF symmetrization
            print("\nget_oversample")
            self._oversample = self.__get_oversample() 
            print("\nget sym psf data")
            psf = self.__get_sym_psf_data()
            print("\nget wl")
            wl = self.__get_wl()
            return chs,psfs_asym,platescale,psf,wl

        chs,psfs_asym,platescale,psf,wl = get_psf_products()

        self.ch = chs
        self._psf = psfs_asym
        self.platescale = platescale
        self.psf = psf
        self.wl = wl


    def __get_oversample(self):
        if self.psfparams.psf_type == 'ext':
            oversample = 1
        else:
            oversample = 5
        return oversample

    def __get_wl(self):
        ch = self.ch
        wl = []
        for i in range(len(ch)):
            ch_i = ch[i]
            wl.append(self.__get_wavelength(ch_i))
        return wl


    def __get_sym_psf_data(self):
        n = len(self.ch)

        psfs = []
        for i in range(n):
            print('Symmetrizing PSF %d of %d' % (i+1,n))
            psfs.append(self.__get_sym_psf(i))
        return psfs
        
    def __get_sym_psf(self,i):
        n = 24
        dtheta = 2*np.pi/n
        
        oversample = self._oversample

        for j in range(n):
            print('Rotation %d of %d' % (j+1,n))
            
            theta = dtheta*j
            rotated_psf = self.__get_rotated_psf(i,theta)
            rotated_psf[np.where(np.isnan(rotated_psf))] = 0

            if j == 0:
                sym_psf = np.zeros_like(rotated_psf)
            sym_psf += rotated_psf
            
        from skimage.transform import downscale_local_mean
        sym_psf = downscale_local_mean(sym_psf,(oversample,oversample))/n
        return sym_psf

    def __get_rotated_psf(self,i,theta):
        psf_array = self._psf[i]

        ctype = ['RA---SIN','DEC--SIN']
        
        w_in = wcs.WCS(naxis=2)
        w_in.wcs.ctype = ctype
        nx_in,ny_in = psf_array.shape
        w_in.array_shape = (nx_in,ny_in)
        w_in.wcs.crpix = [nx_in/2 + 0.5, ny_in/2 + 0.5]
        cdelt_in = self.platescale[i]/3600
        w_in.wcs.cdelt = np.array(2*[cdelt_in])
        w_in.wcs.crota = [0,np.rad2deg(theta)]

        oversample = self._oversample
        
        w_rot = wcs.WCS(naxis=2)
        w_rot.wcs.ctype = ctype
        nx_rot = nx_in*oversample
        ny_rot = ny_in*oversample
        w_rot.array_shape = (nx_rot,ny_rot)
        w_rot.wcs.crpix = [nx_rot/2 + 0.5, ny_rot/2 + 0.5]
        w_rot.wcs.cdelt = np.array(2*[cdelt_in/oversample])
        
        import reproject as rp
        rotated_psf,f = rp.reproject_interp((psf_array.T,w_in),w_rot,\
                                            shape_out=(ny_rot,nx_rot))
        
        return rotated_psf.T

    def __get_psf_data(self,psf_estimates):
        psf_dict = self.__get_psf_dict(psf_estimates)
        psf_chs = list(psf_dict.keys())
        
        nArrays = self.nArrays
        nSubchs = self.chs_per_array

        chs = []
        psfs = []
        for array in range(1,nArrays+1):
            for subch in range(nSubchs):
                ch = (array,subch)
                if ch in psf_chs:
                    chs.append(ch)
                    psfs.append(psf_dict[ch])
        return chs,psfs

    def __get_psf_estimates(self):

        import SPHEREx_InstrumentSimulator as SPinst
        psf_SkySim = SPinst.PSF(self.psf_fn,normalize=False)
        return psf_SkySim

    def __get_psf_ch_unique(self,psf_ch):
        psf_ch_unique = []
        for i in range(len(psf_ch)):
            ch_i = psf_ch[i]
            if ch_i not in psf_ch_unique:
                psf_ch_unique.append(ch_i)
        return psf_ch_unique

    def __get_avg_psf(self,psf_estimates,ch):
        psf_ch = self._psf_ch

        n = 0
        made_acc_psf = False
        for i in range(len(psf_ch)):            
            if ch == psf_ch[i]:
                if not made_acc_psf:
                    acc_psf = np.zeros_like(psf_estimates.get_psf_data(i+1))
                    made_acc_psf = True
                
                acc_psf += psf_estimates.get_psf_data(i+1)
                n += 1
        avg_psf = acc_psf/n
        return avg_psf

    def __get_psf_dict(self,psf_estimates):
        psf_ch = self._psf_ch
        psf_ch_unique = self.__get_psf_ch_unique(psf_ch)
        d = {}
        for ch in psf_ch_unique:
            d[ch] = self.__get_avg_psf(psf_estimates,ch)
        return d

    def __get_psf_ch(self,psf_estimates):
        nArrays = self.nArrays
        nSubchs = self.chs_per_array

        psf_x = psf_estimates.psf_x
        psf_y = psf_estimates.psf_y
        psf_array = psf_estimates.psf_array
        nPSFs = len(psf_x)

        psf_ch = nPSFs*[None]
        for array in range(1,nArrays+1):
            for subch in range(nSubchs):
                print('Array %d, subch. %d' % (array,subch))
                ch_ndxs,wl_range = self.__get_channel_def(array,subch)

                ch_tups = []
                xs,ys = ch_ndxs
                for i in range(len(xs)):
                    ch_tups.append((xs[i],ys[i]))
                
                for i in range(nPSFs):
                    psf_array_i = psf_array[i]
                    if psf_array_i == array:
                        x,y = self.__get_xy_pix(psf_array_i,psf_x[i],psf_y[i])
                        if (x,y) in ch_tups:
                            psf_ch[i] = (array,subch)
        return psf_ch

    def __get_xy_pix(self,array,xmm,ymm):
        npixPerSide = self.npixPerSide
        mmPerPix = self.mmPerPix
        gap_size = self.gap_size_pixels

        xPixFromCen = xmm/mmPerPix
        yPixFromCen = ymm/mmPerPix

        x = xPixFromCen + npixPerSide/2
        y = yPixFromCen + npixPerSide/2
        if array in [1,4]:
            x = x + npixPerSide + gap_size
        elif array in [3,6]:
            x = x - npixPerSide - gap_size

        x = int(x)
        y = int(y)
        return x,y
    
    def __get_psf_wl(self,psf_estimates):
        psf_array = psf_estimates.psf_array
        psf_x = psf_estimates.psf_x
        psf_y = psf_estimates.psf_y

        psf_wl = np.empty(x.shape)
        for i in range(1,self.nArrays+1):
            wl_array = self.__get_wl_array(i)
            for j in range(len(psf_array)):
                array = psf_array[j]
                if j == i:
                    psf_wl[j] = wl_array[psf_x[j],psf_y[j]]
        return psf_wl

    def get_psf_interp(self,wl):
        ''' Interpolate from stored PSFs to a chosen wavelength `wl` [Quantity
        with units of length]. '''
        wl_psf_full = self.wl

        # bracket target wavelength; if impossible, choose two closest
        n_psf_full = len(wl_psf_full)
        imax = n_psf_full-1
        for i in range(n_psf_full):
            wl_tmp = wl_psf_full[i]
            if wl_tmp > wl:
                imax = i
                break
        if imax == 0:
            imax = 1
        imin = imax-1

        wl_psf_min = wl_psf_full[imin]
        wl_psf_max = wl_psf_full[imax]

        psf_min = self.psf[imin]
        psf_max = self.psf[imax]

        # linear interpolation
        dwl = wl_psf_max - wl_psf_min
        dpsf = psf_max - psf_min
        slope = dpsf/dwl

        psf_interp = slope*(wl-wl_psf_min) + psf_min
        return psf_interp
    

## more interpolation methods
    def interp_images_1d(self, wl):
        ''' 
        Interpolate from stored PSFs to a chosen wavelength `wl`
        
        INPUT:
        - wl: where to interpolate, wavelength in um
        
        OUTPUT:
        The interpolated NxN image.
        '''
        
        # get points
        imgs = np.array(self.psf)
        Z1 = np.array([self.wl[i].value for i in range(len(self.wl))])
        n = imgs.shape[1]
        points = (Z1, np.arange(n), np.arange(n))
        
        # get interpolated points
        xi = np.rollaxis(np.mgrid[:n, :n], 0, 3).reshape((n**2, 2))
        xi = np.c_[np.repeat(wl,n**2), xi]
        #print(xi.shape)
        
        # interpolate
        from scipy.interpolate import interpn
        img_interp = interpn(points, imgs, xi, method='linear', bounds_error=False, fill_value=None)
        img_interp = img_interp.reshape((n, n))
        #print(np.sum(img_interp))
        
        return(img_interp)

## interpolate multiple PSFs at a time

    def create_interpolator(self, interp_order):
        """
        Create an interpolator that can be fed into interpolate_at() to calculate a stack of interpolated PSFs

        INPUT:
        - interp_order: string, order of interpolation, could be 'linear', ....

        OUTPUT:
        - An interpolator.
        """

        if len(self.psf) != len(self.wl):
            raise ValueError("Length of Z1 must match the first dimension of images.")
        
        # wavelength array, unitless
        Z1 = np.array([self.wl[i].value for i in range(len(self.wl))])

        # Define the interpolator
        from scipy.interpolate import RegularGridInterpolator
        interpolator = RegularGridInterpolator(
            points=(Z1, np.arange(np.array(self.psf).shape[1]), np.arange(np.array(self.psf).shape[2])),
            values=np.array(self.psf),
            method=interp_order,
            bounds_error=False,
            fill_value=None
        )
        return interpolator
    
    def interpolate_at(self, interpolator, wls):
        """
        Interpolates the stack of images at the specified wavelengths 'wls'.

        INPUT:
        - wls: single value or array of wavelengths interpolate, in um.

        OUTPUT:
        - Interpolated images at each wls position, with shape (len(wls), N, N)
        if wls is an array, or (N, N) if wls is a scalar.

        """
        Z1_interp = np.atleast_1d(wls) 
        Nimage = self.psf[0].shape[0]
        # generate the grid of (i, j) positions for each Z1_interp
        grid = np.stack(np.meshgrid(np.arange(Nimage), np.arange(Nimage), indexing="ij"), axis=-1)
        grid = grid.reshape(-1, 2) 

        # add the Z1_interp dimension to the grid
        full_grid = np.array([
            np.hstack((np.full((grid.shape[0], 1), z), grid))
            for z in Z1_interp
        ]).reshape(-1, 3)

        # interpolate at all points
        interpolated = interpolator(full_grid).reshape(len(Z1_interp), Nimage, Nimage)
        return interpolated if len(Z1_interp) > 1 else interpolated[0]

        


### add methods from mapmakerparams.py
    def __get_channel_def(self,array,subch):
        '''
        Return pixel indices and wavelength range for the channel defined by
        an array and a subchannel.

        Parameters
        ----------
        array [int]: Array ID from [1,2,3,4,5,6]
        subch [int]: Subchannel index. This is 0-indexed in 
                     [0,1,2,...,chs_per_array-1].

        Returns
        -------
        ch_ndxs [tuple]: Pixel indices as tuple with x indices (as array) in 
                         the first element and y indices in the second.
        wl_range [tuple]: Tuple giving wavelength range with minimum wavelength
                          (as astropy quantity with units) in the first element
                          and maximum wavelength in the second.
        '''
        
        # inst_params = self.__get_inst_params()
        filter_dict = self.inst_params['filter']
        lambda_str_dict = filter_dict['lambda_str']
        lambda_end_dict = filter_dict['lambda_end']

        wl_array_min = lambda_str_dict['value'][array-1]
        wl_array_max = lambda_end_dict['value'][array-1]

        ndivs = self.chs_per_band

        wl_array = self.__get_wl_array(array)
        nx,ny = wl_array.shape

        ix_mid = int(nx/2)
        wl_mid = wl_array[ix_mid,:] # middle column
        iy_include = np.where((wl_mid >= wl_array_min)*\
                              (wl_mid <= wl_array_max))
        iy_mid_array_min = np.min(iy_include)
        iy_mid_array_max = np.max(iy_include)

        delta_iy_mid = (iy_mid_array_max + 1 - iy_mid_array_min)/ndivs
        iy_mid_ch_min = iy_mid_array_min - 0.5 + (ndivs-subch)*delta_iy_mid
        iy_mid_ch_max = iy_mid_ch_min + delta_iy_mid
        if subch == ndivs + 1:
            wl_ch_max = np.max(wl_array)
        else:
            wl_ch_max = wl_mid[int(np.ceil(iy_mid_ch_min))]
        if subch == 0:
            wl_ch_min = np.min(wl_array)
        else:
            wl_ch_min = wl_mid[int(np.ceil(iy_mid_ch_max))]

        if subch == 0:
            bool_gtr = (wl_array >= wl_ch_min)
        else:
            bool_gtr = (wl_array > wl_ch_min)
            
        ch_ndxs = np.where((wl_array <= wl_ch_max)*bool_gtr)
        wl_range = (wl_ch_min*u.um,wl_ch_max*u.um)
        return ch_ndxs,wl_range
    
    def __get_wl_array(self,array,lvf_model='default'):
        if lvf_model == 'default':
            from SPHEREx_InstrumentSimulator import roc_lvf
            lvf_model = roc_lvf
        
        camera_dict = self.inst_params['nominal_camera_model']
        nx = camera_dict['npix_x']['value']
        ny = camera_dict['npix_y']['value']
        ix,iy = np.indices((nx,ny))
        wl_array = lvf_model(ix,iy,array=array,central_bandpass_only=True)[0]

        from SPHEREx_InstrumentSimulator import monochromatic_response
        if lvf_model == monochromatic_response:
            ''' monochromatic returns a single wavelength so convert to array
            actually wl_array is a length-2 list of wavelength and
            transmission '''
            wl = wl_array[0] 
            wl_array = wl*np.ones(ix.shape)
        
        return wl_array # um but returned as float, not Quantity
    
### add relevant methods from detparams.py
    def __get_wavelength(self, channel):
        '''
        Returns the average wavelength (as an astropy quantity with units of 
            length) for the pixels defined by self.channel.
        '''        
        from SPHEREx_InstrumentSimulator import Tabular_Bandpass
        lvf_model = Tabular_Bandpass()
        # assume lvf_type = 'tabular

        if type(channel) != list:
            channel = [channel]
        acc_w = 0
        acc_wl = 0
        for ch in channel:
            array,subchannel = ch
            wl_array = self.__get_wl_array(array,lvf_model=lvf_model)

            if subchannel is None:
                wl_cropped_array = wl_array
            else:
                if type(subchannel) == range:
                    subchannel = list(subchannel)
                
                wl_cropped_array = np.empty(wl_array.shape)
                wl_cropped_array[:] = np.nan
                if type(subchannel) != list:
                    subchannel = [subchannel]
                for subch in subchannel:
                    ndxs,wl_range = self.__get_channel_def(array,subch)
                    wl_cropped_array[ndxs] = wl_array[ndxs]
            acc_wl += np.nansum(wl_cropped_array)
            acc_w += len(np.where(~np.isnan(wl_cropped_array))[0])
        wl = (acc_wl/acc_w)
        return wl*u.um