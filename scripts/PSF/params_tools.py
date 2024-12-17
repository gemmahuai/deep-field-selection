import inspect,os,copy
import astropy.units as u
import numpy as np

# attributes for which to make separate subdirectories
sep_attrs = ['radec_center','boundaryparams','Aalpha',\
             'Abeta','ellparams',\
             'iBin','rlz','gridparams','detparams','coaddparams',\
             'lognormal','universe','zrange','fov','zmin','zmax',\
             'logmlim','sourcemaskparams','apoparams',\
             'maskparams','catalogparams','galparams']

def set_attrs(struct,defaultDict,**kwargs):
    # full set of keys of which kwargs specifies a subset
    keys = defaultDict.keys()

    for kw in kwargs.keys():
        if kw in keys: # ignore extraneous keyword arguments
            setattr(struct,kw,kwargs[kw])
        '''else:
            print('\nIgnore extraneous argument \'%s\'' % (kw))'''

    # fill in the rest with default values
    for attr in keys:
        if not hasattr(struct,attr):
            defaultValue = defaultDict[attr]
            if type(defaultValue) == str:
                apostrophe = '\''
            else:
                apostrophe = ''
            '''print('Set default for \'%s\': %s%s%s' \
                  % (attr,apostrophe,defaultValue,apostrophe))'''
            setattr(struct,attr,defaultValue)

def run_method(struct,method_name):
    method = getattr(struct,method_name)
    return method()

def set_params_labels(struct,label_func,defaultDict):    
    struct.label = get_params_label(struct,label_func,defaultDict)
    struct.subdir = get_params_subdir(struct,label_func,defaultDict)

def get_params_str(struct,label_func,defaultDict,str_type):
    if str_type == 'label':
        str_func = get_params_label
    elif str_type == 'subdir':
        str_func = get_params_subdir
    return str_func(struct,label_func,defaultDict)
    
def get_params_subdir(struct,label_func,defaultDict):
    label_list_req = get_label_list_req(struct,label_func)
    label_list_opt = get_label_list_opt(struct,label_func,defaultDict)

    for i in range(len(label_list_req)):
        label_req_temp = label_list_req[i]
        if i == 0:
            d = label_req_temp
        else:
            d = os.path.join(d,label_req_temp)
    for i in range(len(label_list_opt)):
        d = os.path.join(d,label_list_opt[i])
    return d

def get_params_label(struct,label_func,defaultDict):
    label_list_req = get_label_list_req(struct,label_func)
    label = label_list_to_label(label_list_req)
    
    label_list_opt = get_label_list_opt(struct,label_func,defaultDict)
    label_opt = label_list_to_label(label_list_opt)

    if label_opt != '':
        if label == '':
            label = label_opt
        else:
            label = '%s_%s' % (label,label_opt)
    return label

def label_list_to_label(label_list):
    n = len(label_list)
    label = ''
    for i in range(n):
        sublabel = label_list[i]
        if (i == 0) or (sublabel == '') or (label == ''):
            fillStr = ''
        else:
            fillStr = '_'
        label += '%s%s' % (fillStr,sublabel)
    return label

def separate_arg_label(attr,args,label_func):
    if attr in args:
        args.remove(attr)
        label = label_func(attr)
    else:
        label = None
    return label,args

def get_args_label_list(args,label_func):
    label_list = []
    for attr in sep_attrs:
        sep_label,args = separate_arg_label(attr,args,label_func)
        if sep_label is not None:
            label_list.append(sep_label)

    label = ''
    n_args = len(args)
    for i in range(n_args):
        arg = args[i]
        attr_label = label_func(arg)
        if (i == 0) or (attr_label == '') or (label == ''):
            fillStr = ''
        else:
            fillStr = '_'
        label += '%s%s' % (fillStr,attr_label)

    if label != '':
        label_list = [label] + label_list
    elif len(label_list) == 0 and n_args > 0:
        ''' If the label is still an empty string but there were actually
        contributing arguments, make sure that the label list contains
        at least an empty string. '''
        label_list = ['']

    return label_list

def get_label_list_req(struct,label_func):
    req_args = get_req_args(struct)
    return get_args_label_list(req_args,label_func)

def get_label_list_opt(struct,label_func,defaultDict):
    kws = list(defaultDict.keys())
    return get_args_label_list(kws,label_func)

def get_req_args(struct):
    fullargspec = inspect.getfullargspec(struct.__init__)
    args = fullargspec.args
    args.remove('self')
    return args

def get_wh_label(wh,decimals):
    w,h = wh
    return 'w%.*f_h%.*f' % (decimals,w,decimals,h)

def get_reso_label(reso,decimals):
    return 'reso%.*f' % (decimals,reso)

def get_radec_label(radec_center,decimals):
    if type(radec_center) == str:
        return radec_center
    else:
        ra,dec = radec_center
        return 'ra%.*f_dec%.*f' % (decimals,ra,decimals,dec)

def get_rlz_label(rlz):
    return 'rlz%04d' % (rlz)

def get_bin_label(iBin):
    return 'bin%d' % (iBin)

def get_label_from_strlist(str_list):
    nstrs = len(str_list)
    if nstrs == 0:
        label = ''
    else:
        label = str_list[0]
        for i in range(1,nstrs):
            label += '_%s' % (str_list[i])
    return label

def get_subdir_from_strlist(str_list):
    nstrs = len(str_list)
    if nstrs == 0:
        subdir = ''
    else:
        subdir = str_list[0]
        for i in range(1,nstrs):
            subdir = os.path.join(subdir,str_list[i])
    return subdir

def get_Bbeta_label(Bbeta,B_sigfigs,beta_decimals):
    B,beta = Bbeta
    return 'B%.*e_beta%.*f' % (B_sigfigs-1,B,beta_decimals,beta)

def get_wl_label(wl):
    return '%.3fum' % ((wl.to(u.um)).value)

def get_Aalpha_label(Aalpha,A_sigfigs,alpha_decimals):
    A,alpha = Aalpha
    return 'A%.*e_alpha%.*f' % (A_sigfigs-1,A,alpha_decimals,alpha)

def get_lmax_label(lmax):
    return 'lmax%d' % (lmax)

def get_A0_label(A0,sigfigs):
    if A0 == 0:
        numstr = '0'
    else:
        numstr = '%.*e' % (sigfigs-1,np.abs(A0))

    if A0 < 0:
        signstr = '-'
    else:
        signstr = ''

    return 'A0_%s%s' % (signstr,numstr)

def get_roll_label(roll,decimals):
    return 'roll%.*f' % (decimals,roll)

def get_array_label(array):
    return 'a%d' % (array)

def get_fov_label(fov):
    return 'fov_%.1f' % (fov)

def get_pix_label(pix):
    return 'pix_%.1f' % (pix)

def get_Lbox_label(Lbox):
    return 'box_%d' % (Lbox)

def get_dims_label(dims):
    return 'dim_%d' % (dims)

def get_zmin_label(zmin):
    return 'zmin_%.3f' % (zmin)

def get_zmax_label(zmax):
    return 'zmax_%.3f' % (zmax)

def get_zlim_label(zlim):
    return 'z_%.2f_%.2f' % (zlim[0],zlim[1])

def get_logmlim_label(logmlim):
    return 'm_%.2f_%.2f' % (logmlim)

def get_lognormal_label(lognormal):
    if lognormal:
        return 'ln'
    else:
        return ''

def get_zrange_label(zrange,universe,decimals):
    if universe is not None and zrange is not None:
        return 'z%.*f_%.*f' % (decimals,zrange[0],decimals,zrange[1])
    else:
        return ''

def get_version_label(version):
    return version

def get_numcounts_label(numcounts):
    if numcounts:
        return 'numcounts'
    else:
        return ''

def get_mag_label(mag,decimals):
    return 'mag%.*f' % (decimals,mag)

def restrict_sigfigs(x,n):
    '''
    Round an input value to a given number of significant figures.

    Parameters
    ----------
    x [float]: Input value to be rounded.
    n [int]: Number of significant figures to allow.

    Example: restrict_sigfigs(45.871,2) --> 4.6e1
    '''
    if x == 0:
        return 0

    xabs = np.abs(x)
    xabs_new = round(xabs,-int(np.floor(np.log10(xabs))) + (n-1))
    return np.sign(x)*xabs_new

def restrict_decimals(x,n):
    '''
    Round an input value to a given number of decimal places.

    Parameters
    ----------
    x [float]: Input value to be rounded.
    n [int]: Number of decimal places to allow.

    Example: restrict_decimals(45.871,1) --> 45.9
    '''
    return np.around(x,decimals=n)

def get_mask_subdir(maskparams,apoparams='auto'):
    if type(maskparams) != list:
        maskparams_list = [maskparams]
    else:
        maskparams_list = maskparams

    for i in range(len(maskparams_list)):
        mp = copy.deepcopy(maskparams_list[i])

        if apoparams != 'auto':
            mp.apoparams = apoparams

        mp_dir = mp.subdir()

        if i == 0:
            d = mp_dir
        else:
            d = os.path.join(d,mp_dir)
    return d