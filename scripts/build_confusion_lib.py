"""

generates confusion library from sub-threshold sources.

@author: gemmahuai
01/20/2025

"""

import sys
import re
import os
import itertools

import numpy as np
import astropy.units as u
from astropy.table import hstack, Table
from astropy.io import fits

import SPHEREx_SkySimulator as SPsky
from SPHEREx_SkySimulator import QuickCatalog
from SPHEREx_SkySimulator import Catalog_to_Simulate



class ConfusionLibrary():
    
    def __init__(self, 
                 catalog, 
                 ra_colname,
                 dec_colname,
                 id_colname,
                 idx_refcat, 
                 Pointings,
                 Instrument,
                 Scene, 
                 Channels, 
                 sed_path = "/Users/gemmahuai/Desktop/CalTech/SPHEREx/COSMOS_ALL_HIRES_SEDS_010925/"):
        self.catalog = catalog
        self.ra_colname = ra_colname
        self.dec_colname = dec_colname
        self.id_colname
        self.idx_refcat = idx_refcat
        self.P = Pointings
        self.I = Instrument
        self.Scene = Scene
        self.Channels = Channels
        self.sed_path = sed_path
        
    def find_sed_fits_file(self, 
                           index, 
                           ID):
        if index < 60000:
            DIR = self.sed_path + "0_60000/"
        elif index < 120000:
            DIR = self.sed_path + "60000_120000/"
        else:
            DIR = self.sed_path + "120000_166041/"
        return f"{DIR}cosmos2020_hiresSED_FarmerID_{ID:07d}.fits"

    def write_output(self, 
                     params_insert, 
                     fluxes, 
                     flux_errs, 
                     filename, 
                     NewFile=False):
        
        # check units
        if isinstance(fluxes, u.Quantity):
            try:
                # convert the unit of fluxes to mJy
                fluxes = fluxes.to_value(u.mJy)
            except u.UnitConversionError as e:
                    raise e   

            
        if isinstance(flux_errs, u.Quantity):
            try:
                # convert the unit of fluxes to mJy
                flux_errs = flux_errs.to_value(u.mJy)
            except u.UnitConversionError as e:
                    raise e   

        
        spectrum = list(itertools.chain(*zip(fluxes, flux_errs)))  # in mJy
        
        # insert galaxy parameters at the beginning, with the same order as in params_insert;
        # for example, params_insert = [<source_id>, <ra>, <dec>, <N_neighbors>]
        for param in reversed(params_insert):
            spectrum.insert(0, param)

        # output this spectrum
        mode = "w" if NewFile else "a"
        with open(filename, mode) as f:
            if not NewFile:
                f.write("\n")
            f.write(" ".join(map(str, spectrum)))
            
            
    def gen_zero_sed(self):
        
        """
        Generate a high resolution zero SED dictionary, for confusion photometry run.
        
        """
        
        # open a high res sed fits file, borrow the wavelength array
        
        _sed = self.find_sed_fits_file(index=0,
                                       ID=self.catalog[self.id_colname][0])
        
        # create a dictionary to store the zero sed
        zero_sed = {}
        
        try: 
            with fits.open(_sed, memmap=True) as hdul:
                zero_sed['lambda'] = hdul[1].data['lambda']
                header = hdul[1].header
                
                if 'TTYPE1' in header:
                    _ttype1 = header['TTYPE1']
                    _ttype2 = header['TTYPE2']
                
                unit1 = getattr(u, header['TUNIT1'])
                unit2 = getattr(u, header['TUNIT2'])
                
                # append wavelength array
                zero_sed[_ttype1] = hdul[1].data[_ttype1] * unit1
                # append zero fluxes
                zero_sed[_ttype2] = np.zeros_like(hdul[1].data[_ttype1]) * unit2
                
        except:
            # evenly spaced samples
            d_lambda = 1e-4
            zero_sed['lambda'] = np.arange(0.5, 5.5, d_lambda) * u.um
            
            # add zero fluxes
            zero_sed['FLUX'] = np.zeros_like(np.arange(0.5, 5.5, d_lambda)) * u.mJy
        
        return zero_sed
        
    def __call__(self):
        
        ## generate zero sed
        self.zero_spec = self.gen_zero_sed()
        
        
        
    def Coord_gen_rand(self, 
                       Npix, 
                       N_positions):
        
        a_l = self.catalog[self.ra_colname].min()
        a_h = self.catalog[self.ra_colname].max()
        d_l = self.catalog[self.dec_colname].min()
        d_h = self.catalog[self.dec_colname].max()

        size = 6.2 * Npix / 3600  # deg extent
        # randomly drawn ra, dec position
        R, D = np.array([]), np.array([])
        i = 0 # counting purpose
        
        # If there's no nearby sub-threshold neighbors, it's very likely that 
        # it's in /on the edge of a star-masked region.
        # Then skip and move on to the next randomly drawn position.
        while len(R) < N_positions:
            ra = np.random.uniform(low=a_l, high=a_h)
            dec = np.random.uniform(low=d_l, high=d_h)
            
            ra_l, ra_h = ra - size / 2, ra + size / 2
            dec_l, dec_h = dec - size / 2, dec + size / 2

            # count how many faint neighbors there are
            N_rand = len(np.where((self.catalog[self.ra_colname] <= ra_h) & 
                                  (self.catalog[self.ra_colname] >= ra_l) &
                                  (self.catalog[self.dec_colname] >= dec_l) & 
                                  (self.catalog[self.dec_colname] <= dec_h))[0])
            if N_rand > 0:
                R = np.append(R, ra)
                D = np.append(D, dec)
                
            i += 1 # move on to the next one
            
        # check if there's enough random blank position generated, matching N_positions
        if (i != N_positions) or (len(R) != N_positions) or (len(D) != N_positions):
            raise ValueError('Dimension mismatch!')
            
        return R, D

    
    def photometer_one_spot(self,
                            index, 
                            Npix, 
                            Use_Tractor=True, 
                            blank_spectrum=None,
                            verbose=False):

        """
        Photometer a random blank spot within the catalog field.
        
        ----------
        Inputs:
        
        index: int,
               index of this current object being photometered.
               
        Npix: int,
              number of pixels surrounding a chosen blank spot within which 
              we include sub-threshold sources.
              
        Use_Tractor: True - turn on Tractor in QuickCatalog;
                     False - turn off Tractor.
                     
        blank_spectrum: str,
                        file path to the zero SED fits file.
        
        verbose: True - print each step;
                 False - do photometry in silence :)
        
        """
        

        # generate one random position
        (ra, dec) = self.Coord_gen_rand(Npix, 
                                        N_positions=1,
                                        ra_colname=self.ra_colname,
                                        dec_colname=self.dec_colname)
        ra = ra[0]
        dec = dec[0]

        print('\nRA = {}'.format(ra), 'DEC = {}'.format(dec))

        # select nearby sources
        size = 6.2*Npix/3600 # extent in deg
        ra_l = ra - size/2
        ra_h = ra + size/2
        dec_l = dec - size/2
        dec_h = dec + size/2
        want = (self.catalog[self.ra_colname]<ra_h) &\
               (self.catalog[self.ra_colname]>ra_l) &\
               (self.catalog[self.dec_colname]>dec_l) &\
               (self.catalog[self.dec_colname]<dec_h)

        # source index in the provided catalog
        near_idx = np.where(want)[0] 
        # original source index in the untruncated catalog for filename extraction purpose
        near_og_idx = self.catalog["col1"][want] 
        # Tractor ID, or other survey ID of the nearby sources
        near_ID = self.catalog[self.id_colname][want] 

        ## Remove bright sources above threshold, in the reference catalog.
        # nearby sub-threshold sources, index in the given catalog
        near_rmv_idx = np.array([], dtype=int) 
        # nearby sub-threshold sources, original source index in the untruncated catalog
        near_rmv_og_idx = np.array([], dtype=int) 
        # nearby sub-threshold sources, survey ID
        near_rmv_ID = np.array([], dtype=int) 

        for i in range(len(near_idx)):

            if self.idx_refcat[near_idx[i]] == False: 

                #print(near_idx[i])
                near_rmv_idx = np.append(near_rmv_idx, near_idx[i])
                near_rmv_og_idx = np.append(near_rmv_og_idx, near_og_idx[i])
                near_rmv_ID = np.append(near_rmv_ID, near_ID[i])
        print('  Number of sub-threshold neighbors = ', len(near_rmv_idx))



        # list of sub-threshold source name, passed to QC do_not_fit
        source_name = []
        for j in range(len(near_rmv_idx)):
            tID = "{}".format(near_rmv_ID[j]).zfill(7)
            source_name.append('catalog_'+tID)

        # initiate QC w/ tractor
        QC = QuickCatalog(self.P, 
                          self.I, 
                          self.Scene, 
                          Use_Tractor=Use_Tractor, 
                          spectral_channel_table=self.Channels,
                          do_not_fit=source_name,
                          subpixel_offset_x=0, 
                          subpixel_offset_y=0)

        Sources_to_Simulate = Catalog_to_Simulate()

        # the Central random blanck spot to fit
        if blank_spectrum is not None:
            # use the input high resolution zero SED
            Sources_to_Simulate.load_single(name='C', 
                                            ra=ra*u.deg, 
                                            dec=dec*u.deg, 
                                            inputpath=blank_spectrum)
        else:
            # use the default one, derived from __call__()
            Sources_to_Simulate.load_single_from_arrays(name='C',
                                                        ra=ra*u.deg,
                                                        dec=dec*u.deg,
                                                        wave=self.zero_spec['lambda'],
                                                        flux=self.zero_spec['FLUX'])


        # load in all nearby sub-threshold sources
        for i in range(len(near_rmv_idx)):

            if verbose is True: 
                print("\nLoad sub-threshold neighbors...")
                print("  survey ID = ", self.catalog[self.id_colname][near_rmv_idx[i]])

            ra_sub = self.catalog[self.ra_colname][near_rmv_idx[i]]
            dec_sub = self.catalog[self.dec_colname][near_rmv_idx[i]]

            # hires sed filename
            filename = self.find_sed_fits_file(index = near_rmv_og_idx[i], 
                                               ID = self.catalog[self.id_colname][near_rmv_idx[i]])

            Sources_to_Simulate.load_single(name=source_name[i],
                                            ra=ra_sub*u.deg, 
                                            dec=dec_sub*u.deg,
                                            inputpath=filename)

            if verbose is True: print("  filename = ", filename)

        if verbose is True: print("Start photometry ...")

        ## QC measurement
        try: 
            SPHEREx_Catalog, Truth_Catalog = QC(Sources_to_Simulate)

            # extract photometry for the central spot
            this = SPHEREx_Catalog['SOURCE_ID']=='C'
            self.SPHEREx_Catalog = SPHEREx_Catalog[this]
            self.Truth_Catalog = Truth_Catalog[this]

            # return SPHEREx_Catalog, Truth_Catalog
        except:
            print(f"QuickCatalog failed, skip this source {index} and return its ID ")
            return index # Return the simulation index for tracking
        
        print(f'[Photometry {index}] Done.')


    def collate_QuickCatalog_primary(self, output_filename):
        
        """
        Collate output from method self.photometer_one_spot();
        Save primary photometry
        
        --------
        Inputs:
        
        output_filename: str,
                         name of the individual primary photometry file (for photo-z input).

        """
        method_name = f"save_level3_primary"

        # get the method and call it
        method = getattr(SPsky, method_name, None)
        
        if method is None:
            raise AttributeError(f"Method {method_name} does not exist")

            
        method(self.SPHEREx_Catalog,
               output_filename,
               pointing_table=self.P.pointing_table,
               )
        return None

    
    
    def collate_QuickCatalog_secondary(self, 
                                       output_filename,
                                       file_intermediate,
                                       NewFile=False):
        """
        Collate output from method self.photometer_one_spot();
        Save secondary photometry
        
        --------
        Inputs:
        
        output_filename: str,
                         name of the combined secondary photometry file.
        file_intermediate: str,
                         name of the individual output secondary photometry file,
                         if ends with "_del.parq", this intermediate file will be removed later.
                         For example, 'secondary_phot_id{}.parq'.format(index)
        NewFile: True: create a new combined secondary photometry file
                 False: append a secondary photometry spectrum to an existing combined file.
                
        
        """
        
        # extract the index 
        index = re.findall(r'\d+', file_intermediate)
        index = int(index[0])

        # construct the method name dynamically
        method_name = f"save_level3_secondary"

        # get the method and call it
        method = getattr(SPsky, method_name, None)
        if method is None:
            raise AttributeError(f"Method {method_name} does not exist")


        # save to an intermediate secondary file
        method(self.SPHEREx_Catalog, 
               self.Channels, 
               self.I, 
               file_intermediate, 
               pointing_table=self.P.pointing_table, 
               fluxerr_from_weights=True
               )

        secondary_tbl = Table.read(file_intermediate, format="parquet")

        # delete the intermediate file if file_intermediate ends with "_del.parq"
        if file_intermediate.split(".")[0].endswith("del"):
            try:
                os.remove(file_intermediate)
            except FileNotFoundError:
                print(f"File '{file_intermediate}' not found.")

        # save fluxes and fluxerrs
        f = secondary_tbl['flux_allsky'][0] / 1000 # mJy
        fe = secondary_tbl['flux_err_allsky'][0] / 1000 # mJy

        # find source survey ID by matching RA, DEC, for general purposes;
        # for the confusion library, insert confusion spectrum index. 
        idx = np.where(np.isclose(secondary_tbl['ra'][0], self.catalog[self.ra_colname], rtol=1e-7) 
               & np.isclose(secondary_tbl['dec'][0], self.catalog[self.dec_colname], rtol=1e-7))[0]
        
        if idx.size>0:
            source_ID = self.catalog[self.id_colname][idx]
        else:
            source_ID = index
            

        params_list = [source_ID, secondary_tbl['ra'][0], secondary_tbl['dec'][0]]

        # write / append to the output txt file as photoz input
        print("writing to", output_filename)
        self.write_output(params_list, 
                          fluxes = f, 
                          flux_errs = fe, 
                          filename = output_filename, 
                          NewFile=NewFile)

        return None
    
    

    
    
