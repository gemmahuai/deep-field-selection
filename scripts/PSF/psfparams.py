import params_tools as pt

class PSFParams:
    def __init__(self,psf_type='auto',**kwargs):
        '''
        Define PSF to be used in detector definition.
        
        Parameters
        ----------
        psf_type [str]: The type of PSF to use. Defaults to 
                'core', which uses on the core PSF. 
            Can also choose 
                'ext', which uses the extended PSF,
                'full', which uses the sum of 'core' and 'ext',
                None, in which case there is no PSF convlution, or
                'auto', which uses the core PSF for EBL, full PSF
                for sources, and None for the other components.
            For direct mosaics, these PSFs are symmetrized. For
            exposure-based mosaics, the PSFs are used with all
            asymmetries and position dependences.

        If psf_type is not None, need the optional argument
            psfrecparams [PSFRecParams object]: Defines the PSF 
                reconstruction algorithm that constructs the PSF models.
                Defaults to None, which uses the input PSF models. Should
                be able to define different reconstruction algorithms for
                core and extended PSF.
        '''
        self.psf_type = psf_type

        defaultDict = self.__get_defaultDict()
        pt.set_attrs(self,defaultDict,**kwargs)
        self.defaultDict = defaultDict

    def __get_attr_label(self,attr,str_type):
        if attr == 'psf_type':
            psf_type = self.psf_type
            if psf_type is None:
                return 'noPSF'
            else:
                return psf_type
        elif attr == 'psfrecparams':
            psfrecparams = self.psfrecparams
            if psfrecparams is not None:
                return pt.run_method(psfrecparams,str_type)
            else:
                return ''

    def __get_params_str(self,str_type):
        def get_attr_label(attr):
            return self.__get_attr_label(attr,str_type)

        return pt.get_params_str(self,get_attr_label,self.defaultDict,\
                                            str_type)

    def label(self):
        return self.__get_params_str('label')

    def subdir(self):
        return self.__get_params_str('subdir')

    def __get_defaultDict(self):
        defaultDict = {}
    
        if self.psf_type is not None:
            defaultDict['psfrecparams'] = None

        return defaultDict

    def is_reconstructed_psf(self):
        if hasattr(self,'psfrecparams'):
            if self.psfrecparams is not None:
                return True
        return False

    def get_psf_type_eff(self,simparams):
        if self.psf_type == 'auto':
            if simparams.do_conv():
                if simparams.is_static_sources():
                    psf_type_eff = 'full'
                elif simparams.is_eblflatsky():
                    psf_type_eff = 'core'
            else:
                psf_type_eff = None
        else:
            psf_type_eff = self.psf_type
        return psf_type_eff