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
def get_pixels_per_deg():
    screen = get_screen_dims()
    pix_per_degW = screen['resolution'][0] / screen['azimuth_deg']
    pix_per_degH = screen['resolution'][1] / screen['altitude_deg']
    screen_res = screen['resolution']
    pix_per_deg = np.mean([pix_per_degW, pix_per_degH])

    return pix_per_deg, screen_res

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

# -----------------------------------------------------------------------------
# Screen:
# -----------------------------------------------------------------------------

def get_lin_coords(resolution=[1080, 1920], cm_to_deg=True, 
                   xlim_degrees=(-59.7, 59.7), ylim_degrees=(-33.6, 33.6)):
    """
    **From: https://github.com/zhuangjun1981/retinotopic_mapping (Monitor initialiser)

    Parameters
    ----------
    resolution : tuple of two positive integers
        value of the monitor resolution, (pixel number in height, pixel number in width)
    dis : float
         distance from eyeball to monitor (in cm)
    mon_width_cm : float
        width of monitor (in cm)
    mon_height_cm : float
        height of monitor (in cm)
    C2T_cm : float
        distance from gaze center to monitor top
    C2A_cm : float
        distance from gaze center to anterior edge of the monitor
    center_coordinates : tuple of two floats
        (altitude, azimuth), in degrees. the coordinates of the projecting point
        from the eye ball to the monitor. This allows to place the display monitor
        in any arbitrary position.
    visual_field : str from {'right','left'}, optional
        the eye that is facing the monitor, defaults to 'left'
    """
    mon_height_cm = 58.
    mon_width_cm = 103.
    # resolution = [1080, 1920]
    visual_field = 'left'
    
    C2T_cm = mon_height_cm/2. #np.sqrt(dis**2 + mon_height_cm**2)
    C2A_cm = mon_width_cm/2.
    
    # distance form projection point of the eye to bottom of the monitor
    C2B_cm = mon_height_cm - C2T_cm
    # distance form projection point of the eye to right of the monitor
    C2P_cm = -C2A_cm #mon_width_cm - C2A_cm

    map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]),
                                           range(resolution[0]))

    if visual_field == "left":
        #map_x = np.linspace(C2A_cm, -1.0 * C2P_cm, resolution[1])
        map_x = np.linspace(C2P_cm, C2A_cm, resolution[1])

    if visual_field == "right":
        map_x = np.linspace(-1 * C2A_cm, C2P_cm, resolution[1])

    map_y = np.linspace(C2T_cm, -1.0 * C2B_cm, resolution[0])
    old_map_x, old_map_y = np.meshgrid(map_x, map_y, sparse=False)

    lin_coord_x = old_map_x
    lin_coord_y = old_map_y
    
    
    if cm_to_deg:
        xmin_cm = lin_coord_x.min(); xmax_cm = lin_coord_x.max();
        ymin_cm = lin_coord_y.min(); ymax_cm = lin_coord_y.max();
        
        xmin_deg, xmax_deg = xlim_degrees
        ymin_deg, ymax_deg = ylim_degrees
        
        lin_coord_x = convert_range(lin_coord_x, oldmin=xmin_cm, oldmax=xmax_cm, 
                                           newmin=xmin_deg, newmax=xmax_deg)
        lin_coord_y = convert_range(lin_coord_y, oldmin=ymin_cm, oldmax=ymax_cm, 
                                           newmin=ymin_deg, newmax=ymax_deg)
    return lin_coord_x, lin_coord_y


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

