#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  23 13:18:13 2021

@author: julianarhee
"""
import re
import glob
import os
import numpy as np

# ###############################################################
# General
# ###############################################################
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def isnumber(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    except TypeError:
        return False

    return True

def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    from: @FlorianBrucker:
    https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    '''
    n = int(n)
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return str(n) + suffix


def add_datakey(sdata):
    if 'fovnum' not in sdata.keys():
        sdata['fovnum'] = [int(re.findall(r'FOV(\d+)_', x)[0]) for x in sdata['fov']]

    sdata['datakey'] = ['%s_%s_fov%i' % (session, animalid, fovnum)
                              for session, animalid, fovnum in \
                                zip(sdata['session'].values,
                                    sdata['animalid'].values,
                                    sdata['fovnum'].values)]
    return sdata


def split_datakey(df):
    df['animalid'] = [s.split('_')[1] for s in df['datakey'].values]
    df['fov'] = ['FOV%i_zoom2p0x' % int(s.split('_')[2][3:]) for s in df['datakey'].values]
    df['session'] = [s.split('_')[0] for s in df['datakey'].values]
    return df

def split_datakey_str(s):
    session, animalid, fovn = s.split('_')
    fovnum = int(fovn[3:])
    return session, animalid, fovnum

def add_meta_to_df(tmpd, metainfo):
    for v, k in metainfo.items():
        tmpd[v] = k
    return tmpd


def convert_columns_byte_to_str(df):
#     str_df = df.select_dtypes([np.object])
#     str_df = str_df.stack().str.decode('utf-8').unstack()
#     for col in str_df:
#         df[col] = str_df[col]
    new_columns = dict((c, c.decode("utf-8") ) for c in df.columns.tolist())
    df = df.rename(columns=new_columns)
    return df




# ###############################################################
# Visual stimulation functions 
# ###############################################################

def get_pixel_size():
    # Use measured pixel size from PSF (20191005, most recent)
    # ------------------------------------------------------------------
    xaxis_conversion = 2.3 #1  # size of x-axis pixel, goes with A-P axis
    yaxis_conversion = 1.9 #89  # size of y-axis pixels, goes with M-L axis
    return (xaxis_conversion, yaxis_conversion)

def get_screen_dims():
    screen_x = 59.7782*2 #119.5564
    screen_y =  33.6615*2. #67.323
    resolution = [1920, 1080] #[1024, 768]
    deg_per_pixel_x = screen_x / float(resolution[0])
    deg_per_pixel_y = screen_y / float(resolution[1])
    deg_per_pixel = np.mean([deg_per_pixel_x, deg_per_pixel_y])
    screen = {'azimuth_deg': screen_x,
              'altitude_deg': screen_y,
              'azimuth_cm': 103.0,
              'altitude_cm': 58.0,
              'resolution': resolution,
              'deg_per_pixel': (deg_per_pixel_x, deg_per_pixel_y)}

    return screen


# ###############################################################
# Calculation functions 
# ###############################################################

def convert_range(oldval, newmin=None, newmax=None, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

def CoM(df_):
    '''
    Calculate center of mass from coords x0, y0 in dataframe df_
    '''
    x = df_['x0'].values
    y = df_['y0'].values
    m=np.ones(df_['x0'].shape)
    cgx = np.sum(x*m)/np.sum(m)
    cgy = np.sum(y*m)/np.sum(m)
    
    return cgx, cgy


def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb

def convert_uint16(tracemat):
    offset = 32768
    arr = np.zeros(tracemat.shape, dtype='uint16')
    arr[:] = tracemat + offset
    return arr


# Calculationgs
def get_empirical_ci(stat, ci=0.95):
    p = ((1.0-ci)/2.0) * 100
    lower = np.percentile(stat, p) #max(0.0, np.percentile(stat, p))
    p = (ci+((1.0-ci)/2.0)) * 100
    upper = np.percentile(stat, p) # min(1.0, np.percentile(x0, p))
    #print('%.1f confidence interval %.2f and %.2f' % (alpha*100, lower, upper))
    return lower, upper



# Test function for module  
def _test():
    assert add('1', '1') == 2

if __name__ == '__main__':
    _test()

