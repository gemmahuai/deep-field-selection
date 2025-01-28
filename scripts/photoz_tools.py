
'''
Useful tools for photo-z in-house code
by Yongjung Kim
Last updated on Apr. 9, 2024

'''

import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy import interpolate

# Reddening laws
# Allen76 (MW)
def allen_k(wave):

    Rv = 3.1
    wv_vec = [1000.0,1110.0,1250.0,1430.0,1670.0,2000.0,2220.0,2500.0,2850.0,
              3330.0,3650.0,4000.0,4400.0,5000.0,5530.0,6700.0,9000.0,10000.0,
              20000.0,100000.0]
    #k vector
    k_vec = [4.20,3.70,3.30,3.00,2.70,2.80,2.90,2.30,1.97,1.69,
             1.58,1.45,1.32,1.13,1.00,0.74,0.46,0.38,0.11,0.00]

    f = interpolate.interp1d(wv_vec, k_vec, kind="cubic")

    Np = len(wave)
    k = np.zeros(Np)

    for i in range(Np):
        if ((wave[i] >= wv_vec[0]) & (wave[i] <= wv_vec[19])):
            # assign values via interpolation and shift based on Rv
            k[i] = f(wave[i]) * Rv # changed from + Rv : 6/8/2023

        elif wave[i] < wv_vec[0]:
            # bluer wavelengths get attenuated by constant * delta_lambda (dex) + offset
            k[i] = 4.791107 * ( np.log10(wv_vec[0]) - np.log10(wave[i]) ) + k_vec[0] + Rv

        else:
            k[i] = 0 #redder wavelengths set to 0
    return k

# Prevot+84 (SMC)
def prevot_k(wave):

    Rv=2.72

    wv_vec = [1275,1330,1385,1435,1490,1545,1595,1647,1700,1755,1810,1860,1910,2000,2115,     2220,2335,2445,2550,2665,2778,2890,2995,3105,3704,4255,5291,12500,16500,
              22000,24000,26000,28000,30000,32000,34000,36000,38000,40000]

    ee_vec = [13.54,12.52,11.51,10.8,9.84,9.28,9.06,8.49,8.01,7.71,7.17,6.9,6.76,6.38,5.85,     5.3,4.53,4.24,3.91,3.49,3.15,3,2.65,2.29,1.81,1,0,-2.02,-2.36,-2.47,-2.51,-2.55,
              -2.59,-2.63,-2.67,-2.71,-2.75,-2.79,-2.83]

    f = interpolate.interp1d(wv_vec, ee_vec, kind="cubic")

    #k=make_array(n_elements(wave),VALUE=0,/DOUBLE)
    k = np.zeros(len(wave))

    short = np.where((wave < 34500) & (wave > 1275))
    scnt = np.count_nonzero((wave < 34500) & (wave > 1275))

    if scnt > 0:
        k[short] = f(wave[short])+Rv

    long = np.where(wave >= 34500)
    lcnt = np.count_nonzero(wave >= 34500)

    if lcnt > 0:
        k[long]=0

    vshort = np.where(wave < 1275)
    vscnt = np.count_nonzero(wave < 1275)

    if vscnt > 0:
        k[vshort] = 24.151865 * (np.log10(wv_vec[0]) - np.log10(wave[vshort])) + ee_vec[0] + Rv

    return k

# Calzetti+00 (SB)
def calz_k(wave):

    #Input wave assumed Angstrom
    Rv=4.05

    w = wave*1.0e-4 #convert to micron
    iwave = 1.0/w

    k = np.zeros(len(wave))

    short = np.where(w < 0.63)
    scnt = np.count_nonzero(w < 0.63)

    medium = np.where(w >= 0.63)
    mcnt = np.count_nonzero(w >= 0.63)

    if scnt > 0:
        k[short] = (2.659*(- 2.156 + (1.509*iwave[short]) - (0.198*iwave[short]**2) + (0.011*iwave[short]**3)))+Rv

    if mcnt > 0:
        k[medium] = (2.659*(-1.857+(1.040*iwave[medium])))+Rv

    bad = np.where(k < 0)
    bcnt = np.count_nonzero(k < 0)

    if bcnt > 0:
        k[bad]=0

    return k

# Seaton79 (MW)
def seaton_k(wave):

    Rv=3.1

    w = wave*1.0e-4 #convert to micron
    iwave = 1.0/w

    ee = np.zeros(len(wave))

    lo=4.595
    gamma=1.051
    c1=-0.38
    c2=0.74
    c3=3.96
    c4=0.26

    short = np.where(iwave >= 5.9)
    scnt = np.count_nonzero(iwave >= 5.9)

    if scnt > 0:
        ee[short] = c1 + c2*iwave[short] + c3/( (iwave[short] - (lo**2)*w[short])**2  +         gamma**2)+ c4*(0.539*((iwave[short] - 5.9)**2)+0.0564*((iwave[short]-5.9)**3))

    medium = np.where( (iwave < 5.9) & (iwave >= 2.74))
    mcnt = np.count_nonzero((iwave < 5.9) & (iwave >= 2.74))

    if mcnt > 0:
        ee[medium] = c1 + c2*iwave[medium] + c3/( (iwave[medium] - (lo**2)*w[medium])**2                                                  + gamma**2)

    long = np.where(iwave < 2.74)
    lcnt = np.count_nonzero(iwave < 2.74)

    if lcnt > 0:
        ee[long] = allen_k(wave[long])-Rv

    k=ee+Rv

    return k

# Fitzpatrick86 (MW)
def fitzpatrick_k(wave):

    Rv=3.1

    #w = wave*1.0e-4 #convert to micron
    #iwave = 1.0/w

    #ee = np.zeros(len(wave))

    lo=4.608
    gamma=0.994
    c1=-0.69
    c2=0.89
    c3=2.55
    c4=0.50

    #Conversion of IDL
    #short = np.where(iwave >= 5.9)
    #scnt = np.count_nonzero(iwave >= 5.9)

    #if scnt > 0:
    #    ee[short] = c1 + c2*iwave[short] + c3/( (iwave[short] - (lo**2)*w[short])**2  + \
    #    gamma**2)+ c4*(0.539*((iwave[short] - 5.9)**2)+0.0564*((iwave[short]-5.9)**3))

    #medium = np.where( (iwave < 5.9) & (iwave >= 3.0))
    #mcnt = np.count_nonzero((iwave < 5.9) & (iwave >= 3.0))

    #if mcnt > 0:
     #   ee[medium] = c1 + c2*iwave[medium] + c3/( (iwave[medium] - (lo**2)*w[medium])**2 \
    #                                             + gamma**2)

    #long = np.where(iwave < 3.0)
    #lcnt = np.count_nonzero(iwave < 3.0)

    #if lcnt > 0:
    #    ee[long] = allen_k(wave[long])-Rv

    #k=ee+Rv


    ##C++ code conversion follows here

    Np = len(wave)
    k = np.zeros(Np)

    ak = allen_k(wave)

    #for (int i = 0; i < Np; ++i)
    for i in range(Np):

        w = wave[i]*1.0e-4 #convert to micron
        iwave = 1.0/w

        if (iwave >= 5.9):
            k[i] = c1 + c2*iwave + c3/( (iwave - (lo**2)*w)**2 + gamma**2)+ c4*(0.539*((iwave - 5.9)**2)+0.0564*((iwave-5.9)**3))

            k[i] += Rv

        elif (iwave < 5.9) & (iwave >= 3.0):
            k[i] = c1 + c2*iwave + c3/( (iwave - (lo**2)*w)**2 + gamma**2)
            k[i] += Rv

        else:
            k[i] = ak[i] # for redder data, keep allen_k values

    return k

# Applying reddening laws
def apply_dust(wave,flux,ebv,law):

    #apply a dust curve to a flux vector
    #first vector should be wavelength, second flux, third ebv value
    #dust laws are as follows
    # 1 = SB Calzetti et al. 2000
    # 2 = SMC Prevot et al. 1984
    # 3 = MW Allen 1976
    # 4 = MW Seaton 1979
    # 5 = LMC Fitzpatrick 1986

    ## DCM - numbers changed to match sed.cpp

    if law == 1:
        a = calz_k(wave)*ebv
    if law == 2:
        a = prevot_k(wave)*ebv
    if law == 3:
        a = allen_k(wave)*ebv
    if law == 4:
        a = seaton_k(wave)*ebv
    if law == 5:
        a = fitzpatrick_k(wave)*ebv

    if ((law < 0) | (law > 5)):
        print('Invalid dust law, no correction applied!')
        a = np.zeros(len(wave))

    return (10**(-0.4*a)*flux)

# Dust attenuation law (Madau95)
def madau_teff1(x,z):

    teff1=0.0

    wv=1216.0
    zwv=wv*(1.+z)

    if (x < zwv):
        teff1=0.0037*(x/wv)**3.46

    wv=1026.0
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.00177*(x/wv)**3.46

    wv=973.0
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.00106*(x/wv)**3.46

    wv=950.0
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.000584*(x/wv)**3.46

    wv=938.1
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.00044*(x/wv)**3.46

    wv=931.0
    zwv=wv*(1.+z)
    if(x < zwv):
        teff1=teff1+0.00040*(x/wv)**3.46

    wv=926.5
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.00037*(x/wv)**3.46

    wv=923.4
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.00035*(x/wv)**3.46

    wv=921.2
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.00033*(x/wv)**3.46

    wv=919.6
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.00032*(x/wv)**3.46

    wv=918.4
    zwv=wv*(1.+z)
    if (x < zwv):
        teff1=teff1+0.00031*(x/wv)**3.46

    return teff1

# Dust attenuation law (Madau95)
def madau_teff2(x,z):

    zlambda=912.0*(1.0 + z)
    teff2=0.0
    if (x < zlambda):
        xc=x/912.0
        xem=1.0 + z
        teff2=0.25*xc**3*(xem**0.46 - xc**0.46)
        teff2=teff2 + 9.4*xc**1.5*(xem**0.18 - xc**0.18)
        teff2=teff2 - 0.7*xc**3*(xc**(-1.32) - xem**(-1.32))
        teff2=teff2 - 0.023*(xem**1.68 - xc**1.68)

    if (teff2 < 0.0):
        teff2=0.0

    return teff2

# Dust attenuation law (Madau95)
def madau_teff(x,y):

    tf= madau_teff1(x,y) + madau_teff2(x,y)
    teff = np.exp(-tf)

    return teff

# Applying dust attenuation by neutral hydrogen (Madua95)
def apply_madau(wave,flux,z):
    att = np.zeros(len(wave))
    for i in range(len(wave)):
        att[i]=madau_teff(wave[i]*(1.0+z),z)

    return (att*flux)


# Read fitcat output as astropy table
def read_fitcat_output(fitcat_output_file_path):
    fitcat_columns = ['ID',
                      'RA',
                      'DEC',
                      'imag',
                      'zout',
                      'ntmp',
                      'ebv',
                      'rlaw',
                      'fnorm',
                      'chi2']
    
    tbl = Table.read(fitcat_output_file_path, format='ascii.no_header', names=fitcat_columns)

    return tbl


# Read photo-z output as astropy table
def read_photoz_output(photoz_output_file_path):
    photoz_columns = ['ID', #0
                      'RA',
                      'DEC',
                      'z_best',
                      'z_err_std',
                      'skewness',
                      'kurtosis',
                      'pdf_sum',
                    #   'minchi_best', #10
                    #   'meanchi_best',
                    #   'ntmp_best',
                    #   'ebv_best',
                    #   'rlaw_best',
                    #   'fnorm_best', #15
                    #   'z_mp',
                      'density_mp',
                      'z_mp',
                      'minchi_mp',
                      'minchi_best',
                      'meanchi_best',
                      'ntmp_mp',
                      'ntmp_best',
                    #   'ebv_mp', #20
                    #   'rlaw_mp',
                    #   'fnorm_mp',
                      'tflux_per_filter',
                      'wflux_per_tweight',
                      'twsflux_per_tweight', #25
                      'twsflux',
                      'tsflux_per_tnoise',
                      'tflux_per_sqrt_tweight',
                      'snr_per_filter']
    
    tbl = Table.read(photoz_output_file_path, guess=False, fast_reader=False, format='ascii.no_header', names=photoz_columns)

    return tbl


# Compute redshit statistics
def compute_redshift_stats(zml_select, zspec_select, sigma_z_select=None, nsig_outlier=3, outlier_pct=15):
    # The original code is provided by Richard Feder
    # Note that the input sigma_z_select should be sigma_z/(1+zout), not sigma_z

    #statistics
    Ngal = len(zml_select)
    arg_bias = zml_select-zspec_select
    arg_std = arg_bias / (1. + zspec_select)
    bias = np.median(arg_std)
    NMAD = 1.4821 * np.median( np.abs(arg_std-bias))

    h,b = np.histogram(arg_std,bins=np.arange(np.min(arg_std),np.max(arg_std),0.01))
    bc = b[:-1]+0.5*np.diff(b)
    def gaussian(x, amplitude, mean, stddev): # Gaussian fitting to estimate sigma well
        return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)
    fit_parameters, _ = curve_fit(gaussian, bc, h, p0=[1, np.mean(arg_std), np.std(arg_std)])
    sig_dz1z = np.abs(fit_parameters[2])

    if sigma_z_select is not None:
        cond_outl = ( abs(arg_std) > nsig_outlier*sigma_z_select)
    else:
        cond_outl = ( abs(arg_std) > 0.01*outlier_pct)
    outl_rate = len(arg_std[cond_outl]) / float(Ngal)
    return arg_bias, arg_std, bias, NMAD, sig_dz1z, cond_outl, outl_rate


   

# Get best-fit template from the photo-z code output parameters
def get_best_template(input_sed_file_path, zout, ebv, rlaw, fnorm, lambda_unit='um',flux_unit='mag'):

    twave0, tflux0 = np.loadtxt(input_sed_file_path,unpack=True,usecols=[0,1])
    tflux1 = apply_dust(twave0, tflux0, ebv, rlaw)
    tflux2 = apply_madau(twave0, tflux1, zout)
    c = 3e18 # m/s
    tflux = tflux2 * twave0 * twave0 * fnorm / c
    twave = twave0 * (1+zout)

    if lambda_unit == 'Angstrom':
        twave = twave
    elif lambda_unit == 'um':
        twave = twave / 1e4
    else:
        raise ValueError('lambda_unit must be "Angstrom" or "um"')

    if flux_unit == 'mag':
        tmag = -2.5 * np.log10(tflux) + 23.9
        return twave, tmag
    elif flux_unit == 'mJy':
        return twave, tflux * 1e-3
    else:
        raise ValueError('flux_unit must be "mag" or "mJy"')
    
# Read PDF files as dictionary
def read_pdf_dict(pdf_file_path):

    # Columns: "# id xpos ypos redshifts: 0 ..."
    pdf_dict = {}

    with open(pdf_file_path,'r') as pf:
        columns = pf.readline().strip().split()
        redshifts = np.array([float(c) for c in columns[5:]])

        for line in pf:
            values = [x for x in line.strip().split()]
            redshift_values = np.array([float(z) for z in values[3:]])
            pdf_dict[str(values[0])]= redshift_values

    return redshifts, pdf_dict

# Make input file for makegrid 
#  - This is a simple version and one can manually change the input files,
#    but need to keep the format
def make_input_for_grid(name,zmax,zstep,tempmax,ebvmax,ebvstep,dustlawmax,normfac,emlinefac):
    
    thisRA = 150.0 # RA, DEC, ID, and magnitude can be arbitrary
    thisDEC = 2.0
    thisID = 1
    thismag = 25.0
    # thismag = 20.0
    thisz = 0.0
    thistemp = 1
    thisebv = 0.0
    thisdustlaw = 1
    
    f = open(name,"w")
    
    for i in range(int(zmax/zstep)+1):
        
        for j in range(tempmax):
            
            for k in range(int(ebvmax/ebvstep)+1):
                
                for l in range(dustlawmax):
                    
                    thisz = i * zstep
                    thistemp = j+1
                    thisebv = k * ebvstep
                    thisdustlaw = l+1
                    
                    f.write("%d %6.1f %6.1f %6.1f %5.3f %d %4.2f %d %6.2e %6.1f\n" % \
                    (thisID,thisRA,thisDEC,thismag,thisz,thistemp,thisebv,thisdustlaw,normfac,emlinefac))
    f.close()             
    
    return

# Make unity template priors file
def make_unity_template_prob(template_prob_filename,tempmax):
    f = open(template_prob_filename,'w')
    for i in range(tempmax):
        f.write('%d %6.1f\n'%(i+1, 1.0))
    f.close()
