#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 16:24:13 2020

@author: julianarhee
"""

import re
import matplotlib as mpl
mpl.use('agg')

import glob
import os
import shutil
import traceback
import json
import cv2
import h5py

import math
#import skimage
import time

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import tifffile as tf
import _pickle as pkl
import matplotlib.colors as mcolors
import sklearn.metrics as skmetrics 
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable
import analyze2p.retinotopy.utils as retutils
import analyze2p.extraction.rois as roiutils
#from pipeline.python.coregistration import align_fov as coreg
import analyze2p.aggregate_datasets as aggr
import analyze2p.utils as hutils
import analyze2p.plotting as pplot

#from scipy import misc,interpolate,stats,signal
import scipy.stats as spstats
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.morphology import binary_dilation
from scipy.interpolate import SmoothBivariateSpline



from skimage.color import label2rgb
#from skimage.measure import label, regionprops, find_contours
import skimage.measure as skmeasure
from skimage.measure import block_reduce
#pl.switch_backend('TkAgg')

# ------------------------------------------------------------------------------
# Data processing and loading
# ------------------------------------------------------------------------------
def load_roi_assignments(datakey, retinorun='retino_run1', 
                            rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(datakey)
 
    seg_basedir = glob.glob(os.path.join(rootdir, animalid, session, 
                            'FOV%i_*' % fovn, retinorun, 
                            'retino_analysis/segmentation'))[0]
    results_fpath = os.path.join(seg_basedir, 'roi_assignments.json')
    
    assert os.path.exists(results_fpath), \
                "Assignment results not found: %s" % results_fpath
    with open(results_fpath, 'r') as f:
        roi_assignments = json.load(f)
   
    return roi_assignments #, roi_masks_labeled


def load_segmentation_results(datakey, retinorun='retino_run1', 
                                rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(datakey)
 
    retino_seg_dir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn,
                            retinorun,'retino_analysis', 'segmentation'))[0]
 
    results_fpath = os.path.join(retino_seg_dir, 'results.pkl')
    assert os.path.exists(results_fpath), "Seg-results not found: %s" % results_fpath
    with open(results_fpath, 'rb') as f:
        results = pkl.load(f, encoding='latin1')

    params_fpath = os.path.join(retino_seg_dir, 'params.json')
    assert os.path.exists(params_fpath), "Seg-params not found: %s" % params_fpath
    with open(params_fpath, 'r') as f:
        params = json.load(f)
    
    return results, params


def load_processed_maps(datakey, retinorun='retino_run1', 
                        rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    results_dir = glob.glob(os.path.join(rootdir, animalid, session, \
                            'FOV%i_*' % fovn, retinorun, 'retino_analysis',
                            'segmentation'))
    assert len(results_dir)==1, "No segmentation results, %s" % retinorun
     
    processedmaps_fpath = os.path.join(results_dir[0], 'processed_maps.npz')
    pmaps = np.load(processedmaps_fpath)
    
    processingparams_fpath = os.path.join(results_dir[0], 'processing_params.json')
    with open(processingparams_fpath, 'r') as f:
        pparams = json.load(f)

    return pmaps, pparams


def get_processed_maps(datakey, retinorun='retino_run1', 
                        analysis_id=None, create_new=False, 
                        pix_mag_thr=0.002, delay_map_thr=1, 
                        rootdir='/n/coxfs01/2p-data'):
    if not create_new:
        try:
            pmaps, pparams = load_processed_maps(datakey, retinorun, rootdir=rootdir)
        except Exception as e:
            print(e)
            print(" -- procssing maps now...")
            create_new=True

    if create_new:
        # Load data metainfo
        print("Current run: %s" % retinorun)
        retinoid, RETID = retutils.load_retino_analysis_info(datakey, retinorun, 
                                                             use_pixels=True)
        data_id = '_'.join([datakey, retinorun, retinoid])
        print("DATA ID: %s" % data_id)
       
        analysis_basedir = RETID['DST'].split('/analysis')[0] 
        curr_dst_dir = os.path.join(analysis_basedir, 'segmentation')
        if not os.path.exists(curr_dst_dir):
            os.makedirs(curr_dst_dir)
            print(curr_dst_dir)
        # Load MW info and SI info
        mwinfo = retutils.load_mw_info(datakey, retinorun)
        scaninfo = retutils.get_protocol_info(datakey, run=retinorun) 
        trials_by_cond = scaninfo['trials']
     
        # Get run results
        magratio, phase, trials_by_cond = retutils.fft_results_by_trial(RETID)
        d2 = scaninfo['pixels_per_line']
        d1 = scaninfo['lines_per_frame']
        print("Original dims: [%i, %i]" % (d1, d2))
        ds_factor = int(RETID['PARAMS']['downsample_factor'])
        print('Data were downsampled by %i.' % ds_factor)

        # Get pixel size
        pixel_size = hutils.get_pixel_size()
        pixel_size_ds = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)

        # #### Get maps
        abs_vmin, abs_vmax = (-np.pi, np.pi)

        _, absolute_az, absolute_el, delay_az, delay_el = retutils.absolute_maps_from_conds(
                                                        magratio, phase, 
                                                        trials_by_cond=trials_by_cond,
                                                        mag_thr=pix_mag_thr, 
                                                        dims=(d1, d2),
                                                        ds_factor=ds_factor)

        fig = retutils.plot_phase_and_delay_maps(absolute_az, absolute_el, 
                                                  delay_az, delay_el,
                                                  cmap='nipy_spectral', 
                                                  vmin=abs_vmin, vmax=abs_vmax)
        pplot.label_figure(fig, data_id)
        pl.savefig(os.path.join(curr_dst_dir, 'delay_map_filters.png'))


        # #### Filter where delay map is not uniform (Az v El)
        filt_az, filt_el = retutils.filter_by_delay_map(absolute_az, absolute_el, 
                                                    delay_az, delay_el, 
                                                    delay_map_thr=delay_map_thr, 
                                                    return_delay=False)
        filt_azim_r = retutils.transform_2p_fov(filt_az, pixel_size_ds, normalize=False)
        filt_elev_r = retutils.transform_2p_fov(filt_el, pixel_size_ds, normalize=False)

        # Save processing results + params
        processedmaps_fpath = os.path.join(curr_dst_dir, 'processed_maps.npz')
        np.savez(processedmaps_fpath, 
                 absolute_az=absolute_az, absolute_el=absolute_el,
                 filtered_az=filt_az, filtered_el=filt_el,
                 filtered_az_scaled=filt_azim_r, filtered_el_scaled=filt_elev_r)

        # load
        pmaps = np.load(processedmaps_fpath)
        processedparams_fpath = os.path.join(curr_dst_dir, 'processing_params.json')
        pparams = {'pixel_mag_thr': pix_mag_thr,
                    'ds_factor': ds_factor,
                    'delay_map_thr': delay_map_thr,
                    'dims': (d1, d2),
                    'pixel_size': pixel_size,
                    'retino_id': retinoid, 
                    'retino_run': retinorun}

        with open(processedparams_fpath, 'w') as f:
            json.dump(pparams, f, indent=4)

    return pmaps, pparams


# --------------------------------------------------------------------
# Smoothing/Map processing
# --------------------------------------------------------------------
from scipy import ndimage as nd
def fill_nans(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def fill_and_smooth_nans_missing(img, kx=1, ky=1):
    '''
    Smooths image and fills over NaNs. 
    Useful for dealing with holes from neuropil masks
    '''
    y, x = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    x = x.astype(float)
    y = y.astype(float)
    z = img.copy()

    xx = x.copy()
    yy = y.copy()
    xx[np.isnan(z)] = np.nan
    yy[np.isnan(z)] = np.nan

    xx=xx.ravel()
    xx=(xx[~np.isnan(xx)])
    yy=yy.ravel()
    yy=(yy[~np.isnan(yy)])
    zz=z.ravel()
    zz=(zz[~np.isnan(zz)])

    xnew = np.arange(x.min(), x.max()+1) #np.arange(9,11.5, 0.01)
    ynew = np.arange(y.min(), y.max()+1) #np.arange(10.5,15, 0.01)

    f = SmoothBivariateSpline(xx, yy, zz, kx=kx,ky=ky)
    znew=np.transpose(f(xnew, ynew))
    znew[np.isnan(z.T)] = np.nan

    # make sure limits are preserved
    orig_min = np.nanmin(img)
    orig_max = np.nanmax(img)
    zfinal = znew.copy()
    zfinal[znew<orig_min] = orig_min
    zfinal[znew>orig_max] = orig_max
 
    return zfinal.T #znew.T #a


def smooth_maps(start_az, start_el, smooth_fwhm=12, smooth_spline=(1,1), 
                fill_nans=True,
                smooth_spline_x=None, smooth_spline_y=None, target_sigma_um=25, 
                start_with_transformed=True, use_phase_smooth=False, ds_factor=2):

    pixel_size = hutils.get_pixel_size()
    pixel_size_ds = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)

    smooth_spline_x = smooth_spline[0] if smooth_spline_x is None else smooth_spline_x
    smooth_spline_y = smooth_spline[1] if smooth_spline_y is None else smooth_spline_y

    um_per_pix = np.mean(pixel_size) if start_with_transformed \
                    else np.mean(pixel_size_ds)
    smooth_fwhm = int(round(target_sigma_um/um_per_pix))  # int(25*pix_per_deg) #11
    sz=smooth_fwhm*2 #smooth_fwhm
    print("Target: %i (fwhm=%i, k=(%i, %i))" \
            % (target_sigma_um, smooth_fwhm, smooth_spline_x, smooth_spline_y)) #, sz)
    smooth_type = 'phasenan'if use_phase_smooth else 'gaussian'

    print("start", np.nanmin(start_az), np.nanmax(start_az))
    if use_phase_smooth:
        azim_smoothed = retutils.smooth_phase_nans(start_az, smooth_fwhm, sz)
        elev_smoothed = retutils.smooth_phase_nans(start_el, smooth_fwhm, sz)
    else:
        azim_smoothed = retutils.smooth_neuropil(start_az, smooth_fwhm=smooth_fwhm)
        elev_smoothed = retutils.smooth_neuropil(start_el, smooth_fwhm=smooth_fwhm)
    print("smoothed", np.nanmin(azim_smoothed), np.nanmax(azim_smoothed))

    if fill_nans:
        azim_fillnan=None
        elev_fillnan=None
        try:
            azim_fillnan = fill_and_smooth_nans_missing(azim_smoothed, 
                                            kx=smooth_spline_x, ky=smooth_spline_x)
        except Exception as e: # sometimes if too filled, fails w ValueError
            traceback.print_exc()
            print("[AZ] Bad NaN fill. Try a smaller target_smooth_um value")
             #azim_fillnan = fill_and_smooth_nans(azim_smoothed, 
             #                              kx=smooth_spline_x, ky=smooth_spline_x)
        try:
             elev_fillnan = fill_and_smooth_nans_missing(elev_smoothed, 
                                            kx=smooth_spline_y, ky=smooth_spline_y)
        except Exception as e:
            print("[EL] Bad NaN fill. Try a smaller target_smooth_um value")
        print("fillnan", np.nanmin(azim_fillnan), np.nanmax(azim_fillnan))
    else:
        
        azim_fillnan = retutils.smooth_neuropil(azim_smoothed, smooth_fwhm=smooth_fwhm)
        elev_fillnan = retutils.smooth_neuropil(elev_smoothed, smooth_fwhm=smooth_fwhm)
    # Transform FOV to match widefield
    az_fill = retutils.transform_2p_fov(azim_fillnan, pixel_size, normalize=False) \
                                        if not start_with_transformed else azim_fillnan
    el_fill = retutils.transform_2p_fov(elev_fillnan, pixel_size, normalize=False) \
                                        if not start_with_transformed else elev_fillnan
    print("fillnan", np.nanmin(az_fill), np.nanmax(az_fill))

    azim_ = {'smoothed': azim_smoothed, 'nan_filled': azim_fillnan, 'final': az_fill}
    elev_ = {'smoothed': elev_smoothed, 'nan_filled': elev_fillnan, 'final': el_fill}

    return azim_, elev_ 


# --------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------
def plot_smoothing_results(sm_azim, sm_elev, fig=None, axn=None):
    '''plot smoothing results (smoothed, final)'''
    if axn is None:
        fig, axn = pl.subplots(1, 4, figsize=(7,3))
    az_min, az_max = np.nanmin(sm_azim['input']), np.nanmax(sm_azim['input'])
    el_min, el_max = np.nanmin(sm_elev['input']), np.nanmax(sm_elev['input'])
    plotn=0
    for ai, skey in enumerate(['smoothed', 'final']):
        ax=axn[plotn]
        if plotn==0:
            ax.set_title('Azimuth', loc='left')
        im1 = sm_azim[skey].copy()
        ax.imshow(im1, cmap='Spectral', vmin=az_min, vmax=az_max)

        ax=axn[plotn+2]
        if plotn==0:
            ax.set_title('Elevation', loc='left')
        im2 = sm_elev[skey].copy()
        ax.imshow(im2, cmap='Spectral', vmin=el_min, vmax=el_max)
        plotn+=1
    pl.subplots_adjust(left=0.05, right=0.85, wspace=0.2, bottom=0.2, top=0.8)
    
    for ax in axn.flat:
        ax.axis('off')
    return fig, axn


def plot_processing_steps_pixels(filt_az, azim_smoothed, azim_fillnan, az_fill,
                            filt_el, elev_smoothed, elev_fillnan, el_fill,
                            cmap_phase='nipy_spectral', show_cbar=False,
                            vmin=-np.pi, vmax=np.pi, full_cmap_range=True,
                            smooth_fwhm=7, delay_map_thr=1, smooth_spline=(1,1)):
    '''visualize retinomap processing steps'''

    if isinstance(smooth_spline, tuple):
        smooth_spline_x, smooth_spline_y = smooth_spline
    else:
        smooth_spline_x = smooth_spline
        smooth_spline_y = smooth_spline

    fig, axn = pl.subplots(2,4, figsize=(7,4))

    ax = axn[0,0]
    if full_cmap_range:
        im0=ax.imshow(filt_az, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    else:
        im0=ax.imshow(filt_az, cmap=cmap_phase)
    ax.set_ylabel('Azimuth')
    ax.set_title('abs map (delay thr=%.2f)' % delay_map_thr)
    if show_cbar:
        pplot.colorbar(im0)


    ax = axn[0, 1]
    if full_cmap_range:
        im0=ax.imshow(azim_smoothed, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    else:
        im0=ax.imshow(azim_smoothed, cmap=cmap_phase)
    ax.set_title('spatial smooth (%i)' % smooth_fwhm)
    if show_cbar:
        pplot.colorbar(im0)


    ax = axn[0, 2]
    if full_cmap_range:
        im0 = ax.imshow(azim_fillnan, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    else:
        im0 = ax.imshow(azim_fillnan, cmap=cmap_phase)
    ax.set_title('filled NaNs (spline=(%i, %i))' % (smooth_spline_x, smooth_spline_y))
    if show_cbar:
        pplot.colorbar(im0)


    ax = axn[0, 3]
    if full_cmap_range:
        im0 = ax.imshow(az_fill, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    else:
        im0 = ax.imshow(az_fill, cmap=cmap_phase)
    ax.set_title('final')
    if show_cbar:
        pplot.colorbar(im0)

    ax = axn[1, 0]
    if full_cmap_range:
        im1=ax.imshow(filt_el, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    else:
        im1=ax.imshow(filt_el, cmap=cmap_phase)
    ax.set_ylabel('Altitude')
    if show_cbar:
        pplot.colorbar(im1)

    ax = axn[1, 1]
    if full_cmap_range:
        im1=ax.imshow(elev_smoothed, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    else:
        im1=ax.imshow(elev_smoothed, cmap=cmap_phase) 
    if show_cbar:
        pplot.colorbar(im1)


    ax = axn[1, 2]
    if full_cmap_range:
        im1=ax.imshow(elev_fillnan, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    else:
        im1=ax.imshow(elev_fillnan, cmap=cmap_phase)
    #ax.set_title('filled NaNs')
    if show_cbar:
        pplot.colorbar(im1)


    ax = axn[1, 3]
    if full_cmap_range:
        im1= ax.imshow(el_fill, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    else:
        im1 = ax.imshow(el_fill, cmap=cmap_phase)
    #ax.set_title('final')
    if show_cbar:
        pplot.colorbar(im1)

    pl.subplots_adjust(wspace=0.2, hspace=0.2, left=0.1, right=0.95)
    for ax in axn.flat:
        pplot.turn_off_axis_ticks(ax, despine=False)

    return fig


# ####################################################################
# --------------------------------------------------------------------
# segmentation
# -------------------------------------------------------------------- 
def segment_areas(img_az, img_el, sign_map_thr=0.5):

    # Calculate gradients
    # ---------------------------------------------------------
    h_map = img_el.copy()
    v_map = img_az.copy()
    [h_gy, h_gx] = np.array(gradient_phase(h_map))
    [v_gy, v_gx] = np.array(gradient_phase(v_map))

    h_gdir = np.arctan2(h_gy, h_gx) # gradient direction
    v_gdir = np.arctan2(v_gy, v_gx)

    # Create sign map
    # ---------------------------------------------------------
    gdiff = v_gdir-h_gdir
    gdiff = (gdiff + math.pi) % (2*math.pi) - math.pi

    #O=-1*np.sin(gdiff)
    O=np.sin(gdiff) # LEFT goes w/ BOTTOM.  RIGHT goes w/ TOP.
    S=np.sign(O) # Discretize into patches

    # Calculate STD, and threshold to separate areas (simple morph. step)
    # ---------------------------------------------------------
    O_sigma = np.nanstd(O)
    S_thr = np.zeros(np.shape(O))
    S_thr[O>(O_sigma*sign_map_thr)] = 1
    S_thr[O<(-1*O_sigma*sign_map_thr)] = -1
    
    return O, S_thr

def segment_and_label(S_thr, min_region_area=500):

    # Create segmented + labeled map
    # ---------------------------------------------------------
    filled_smap = fill_nans(S_thr)
    labeled_image_tmp, n_labels = skmeasure.label(
                                 filled_smap, background=0, return_num=True)

    image_label_overlay = label2rgb(labeled_image_tmp) #, image=segmented_img) 
    print(labeled_image_tmp.shape, image_label_overlay.shape)
    rprops_ = skmeasure.regionprops(labeled_image_tmp, filled_smap)
    region_props = [r for r in rprops_ if r.area > min_region_area]
    
    # Relabel image
    labeled_image = np.zeros(labeled_image_tmp.shape)
    for ri, rprop in enumerate(region_props):
        new_label = int(ri+1)
        labeled_image[labeled_image_tmp==rprop.label] = new_label
        rprop.label = new_label
        region_props[ri] = rprop
        
    return region_props, labeled_image 


def do_morphological_steps(S, close_k=31, open_k=131, dilate_k=31):

    # Morphological closing
    kernel =  np.ones((close_k, close_k))
    closing_s1 = cv2.morphologyEx(S, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Morphological opening
    ernel = np.ones((open_k, open_k))
    opening_s1 = cv2.morphologyEx(closing_s1, cv2.MORPH_OPEN, kernel, iterations=1)
    # Morphological dilation
    kernel = np.ones((dilate_k, dilate_k))
    dilation = cv2.dilate(opening_s1, kernel, iterations=1)
    # dilation = cv2.morphologyEx(opening_1, cv2.MORPH_CLOSE, kernel, iterations=niter)

    return S, closing_s1, opening_s1, dilation

def plot_morphological_steps(S, closing_s1, opening_s1, dilation,
                            close_k=None, open_k=None, dilate_k=None):
    # Plot steps
    f, axf = pl.subplots(1,4) #pl.figure()
    axn = axf.flat
    ax=axn[0]
    ax.set_title("sign map")
    ax.imshow(S,cmap='jet')

    ax=axn[1]
    im=ax.imshow(closing_s1, cmap='jet')
    ax.set_title('closing (%i)' % close_k)

    ax=axn[2]
    im=ax.imshow(opening_s1, cmap='jet')
    ax.set_title('opening (%i)' % open_k)

    ax=axn[3]
    im=ax.imshow(dilation, cmap='jet')
    pplot.colorbar(im)
    ax.set_title('dilation (%i)' % dilate_k)

    return f


def plot_segmentation_steps(img_az, img_el, surface=None, O=None, S_thr=None, params=None,
                            cmap='viridis', labeled_image=None, region_props=None, 
                            label_color='w', lw=1):
    
    sign_map_thr = 0 if params is None else params['sign_map_thr']

    fig, axf = pl.subplots(2, 3, figsize=(8,8))
    axn = axf.flat

    ax=axn[0]; #ax.set_title(proc_info_str, loc='left', fontsize=12)
    im0 = ax.imshow(surface, cmap='gray'); ax.axis('off');

    ax=axn[1]
    im0 = ax.imshow(img_az, cmap=cmap); ax.axis('off');
    pplot.colorbar(im0, label='az')

    ax=axn[2]
    im0 = ax.imshow(img_el, cmap=cmap); ax.axis('off');
    pplot.colorbar(im0, label='el')

    ax=axn[3]; ax.set_title('Sign Map, O');
    im0 = ax.imshow(O, cmap='jet'); ax.axis('off');
    
    ax=axn[4]; ax.set_title('Visual Field Patches\n(std_thr=%.2f)' % sign_map_thr)
    im = ax.imshow(S_thr, cmap='jet'); ax.axis('off');

    cbar_ax = fig.add_axes([0.35, 0.1, 0.3, 0.02])
    cbar_ticks = np.linspace(-1, 1, 5)
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=cbar_ticks)
    cbar_ax.tick_params(size=0)

    ax = axn[5]
    ax.imshow(labeled_image)
    for ri, region in enumerate(region_props): 
        ax.text(region.centroid[1], region.centroid[0], 
                '%i' % region.label, fontsize=24, color=label_color)
#     for index in np.arange(0, len(region_props)):
#         label = region_props[index].label
#         contour = skmeasure.find_contours(labeled_image == label, 0.5)[0]
#         ax.plot(contour[:, 1], contour[:, 0], label_color)
    contours = skmeasure.find_contours(labeled_image, level=0)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], label_color, lw=lw)
        
    ax.set_title('Labeled (%i patches)' % len(region_props))
    ax.axis('off')

    return fig

def plot_labeled_areas(filt_azim_r, filt_elev_r, surface_2p, label_keys,
                        labeled_image_2p, labeled_image_incl, region_props, 
                        cmap_phase='nipy_spectral', pos_multiplier=(1,1)):

    fig, axn = pl.subplots(1,3, figsize=(9,3))
    ax=axn[0]
    ax.imshow(surface_2p, cmap='gray')
    ax.set_title('Azimuth')
    im0 = ax.imshow(filt_azim_r, cmap=cmap_phase)
    pplot.colorbar(im0)
    ax = overlay_all_contours(labeled_image_2p, ax=ax, lw=2, lc='k')
    pplot.turn_off_axis_ticks(ax, despine=False)

    ax=axn[1]
    ax.imshow(surface_2p, cmap='gray')
    im1=ax.imshow(filt_elev_r, cmap=cmap_phase)
    pplot.colorbar(im1)
    ax.set_title('Elevation')
    ax = overlay_all_contours(labeled_image_2p, ax=ax, lw=2, lc='k')
    pplot.turn_off_axis_ticks(ax, despine=False)

    ax=axn[2]
    ax.imshow(surface_2p, cmap='gray')
    labeled_image_incl_2p = cv2.resize(labeled_image_incl, \
                                (surface_2p.shape[1], surface_2p.shape[0]))
    ax.imshow(labeled_image_incl_2p, cmap='jet', alpha=0.5)
    ax = overlay_all_contours(labeled_image_2p, ax=ax, lw=2, lc='k')

    area_ids = [k[1] for k in label_keys]
    for region in region_props:
        if region.label in area_ids:
            region_name = str([k[0] for k in label_keys if k[1]==region.label][0])
            ax.text(region.centroid[1]*pos_multiplier[0], 
                    region.centroid[0]*pos_multiplier[1], 
                    '%s (%i)' % (region_name, region.label), fontsize=18, color='r')
            print(region_name, region.area)
    ax.set_title('Labeled (%i patches)' % len(area_ids))
    pplot.turn_off_axis_ticks(ax, despine=False)

    return fig

def overlay_all_contours(labeled_image, ax=None, lc='w', lw=2):
    if ax is None:
        fig, ax = pl.subplots()
    
    label_ids = [l for l in np.unique(labeled_image) if l!=0]
    #print(label_ids)
    #for label in label_ids: # range(1, labeled_image.max()):
        #contour = skmeasure.find_contours(labeled_image == label, 0.5)[-1]
        #print(label, len(contour))
        #ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)
    contours = skmeasure.find_contours(labeled_image, level=0)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)

    return ax





# Calculation gradients
## =======================================================================
# Gradient functions
# =======================================================================
import math

def py_ang(v1, v2):
    '''Returns the angle in radians between vectors v1 and v2'''
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def gradient_phase(f, *varargs, **kwargs):
    """
    Return the gradient of an N-dimensional array.
    The gradient is computed using second order accurate central differences
    in the interior and either first differences or second order accurate
    one-sides (forward or backwards) differences at the boundaries. The
    returned gradient hence has the same shape as the input array.
    Parameters
    ----------
    f : array_like
        An N-dimensional array containing samples of a scalar function.
    varargs : scalar or list of scalar, optional
        N scalars specifying the sample distances for each dimension,
        i.e. `dx`, `dy`, `dz`, ... Default distance: 1.
        single scalar specifies sample distance for all dimensions.
        if `axis` is given, the number of varargs must equal the number of axes.
    edge_order : {1, 2}, optional
        Gradient is calculated using N\ :sup:`th` order accurate differences
        at the boundaries. Default: 1.
        .. versionadded:: 1.9.1
    axis : None or int or tuple of ints, optional
        Gradient is calculated only along the given axis or axes
        The default (axis = None) is to calculate the gradient for all the axes of the input array.
        axis may be negative, in which case it counts from the last to the first axis.
        .. versionadded:: 1.11.0
    Returns
    -------
    gradient : list of ndarray
        Each element of `list` has the same shape as `f` giving the derivative
        of `f` with respect to each dimension.
    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 11, 16], dtype=np.float)
    >>> np.gradient(x)
    array([ 1. ,  1.5,  2.5,  3.5,  4.5,  5. ])
    >>> np.gradient(x, 2)
    array([ 0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])
    For two dimensional arrays, the return will be two arrays ordered by
    axis. In this example the first array stands for the gradient in
    rows and the second one in columns direction:
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=np.float))
    [array([[ 2.,  2., -1.],
            [ 2.,  2., -1.]]), array([[ 1. ,  2.5,  4. ],
            [ 1. ,  1. ,  1. ]])]
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> dx = np.gradient(x)
    >>> y = x**2
    >>> np.gradient(y, dx, edge_order=2)
    array([-0.,  2.,  4.,  6.,  8.])
    The axis keyword can be used to specify a subset of axes of which the gradient is calculated
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=np.float), axis=0)
    array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]])
    """
    f = np.asanyarray(f)
    N = len(f.shape)  # number of dimensions

    axes = kwargs.pop('axis', None)
    if axes is None:
        axes = tuple(range(N))
    # check axes to have correct type and no duplicate entries
    if isinstance(axes, int):
        axes = (axes,)
    if not isinstance(axes, tuple):
        raise TypeError("A tuple of integers or a single integer is required")

    # normalize axis values:
    axes = tuple(x + N if x < 0 else x for x in axes)
    if max(axes) >= N or min(axes) < 0:
        raise ValueError("'axis' entry is out of bounds")

    if len(set(axes)) != len(axes):
        raise ValueError("duplicate value in 'axis'")

    n = len(varargs)
    if n == 0:
        dx = [1.0]*N
    elif n == 1:
        dx = [varargs[0]]*N
    elif n == len(axes):
        dx = list(varargs)
    else:
        raise SyntaxError(
            "invalid number of arguments")

    edge_order = kwargs.pop('edge_order', 1)
    if kwargs:
        raise TypeError('"{}" are not valid keyword arguments.'.format(
                                                  '", "'.join(kwargs.keys())))
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D', 'm', 'M']:
        otype = 'd'

    # Difference of datetime64 elements results in timedelta64
    if otype == 'M':
        # Need to use the full dtype name because it contains unit information
        otype = f.dtype.name.replace('datetime', 'timedelta')
    elif otype == 'm':
        # Needs to keep the specific units, can't be a general unit
        otype = f.dtype

    # Convert datetime64 data into ints. Make dummy variable `y`
    # that is a view of ints if the data is datetime64, otherwise
    # just set y equal to the array `f`.
    if f.dtype.char in ["M", "m"]:
        y = f.view('int64')
    else:
        y = f

    for i, axis in enumerate(axes):

        if y.shape[axis] < 2:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least two elements are required.")
        
        # Numerical differentiation: 1st order edges, 2nd order interior
        if y.shape[axis] == 2 or edge_order == 1:
            
            # Use first order differences for time data
            out = np.empty_like(y, dtype=otype)

            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(2, None)
            slice3[axis] = slice(None, -2)
            # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
            out[slice1] = (y[slice2] - y[slice3])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi
            out[slice1]=out[slice1]/2.0

            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            # 1D equivalent -- out[0] = (y[1] - y[0])
            out[slice1] = (y[slice2] - y[slice3])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            # 1D equivalent -- out[-1] = (y[-1] - y[-2])
            out[slice1] = (y[slice2] - y[slice3])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi

        # Numerical differentiation: 2st order edges, 2nd order interior
        else:
            # Use second order differences where possible
            out = np.empty_like(y, dtype=otype)

            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(2, None)
            slice3[axis] = slice(None, -2)
            # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
            out[slice1] = (y[slice2] - y[slice3])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi
            out[slice1] = out[slice1]/2

            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            # 1D equivalent -- out[0] = -(3*y[0] - 4*y[1] + y[2]) / 2.0
            out[slice1] = -(3.0*y[slice2] - 4.0*y[slice3] + y[slice4])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi
            out[slice1]=out[slice1]/2.0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            slice4[axis] = -3
            # 1D equivalent -- out[-1] = (3*y[-1] - 4*y[-2] + y[-3])
            out[slice1] = (3.0*y[slice2] - 4.0*y[slice3] + y[slice4])
            out[slice1] = (out[slice1] + math.pi) % (2*math.pi) - math.pi
            out[slice1]=out[slice1]/2.0

        # divide by step size
        out /= dx[i]
        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len(axes) == 1:
        return outvals[0]
    else:
        return outvals


def image_gradient(img):
    '''
    Calculate 2d gradient, plus mean direction, etc. Return as dict.
    '''
    # Get gradient
    gdy, gdx = np.gradient(img)
    
    # 3) Calculate the magnitude
    gradmag = np.sqrt(gdx**2 + gdy**2)

    # 3) Take the absolute value of the x and y gradients
    #abs_gdx = np.absolute(gdx)
    #abs_gdy = np.absolute(gdy)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    abs_gd = np.arctan2(gdy, gdx) # np.arctan2(abs_gdy, abs_gdx) # [-pi, pi]
    # Get mean direction
    mean_dir = np.rad2deg(spstats.circmean([np.arctan2(gy, gx) 
                         for gy, gx in zip(gdy.ravel(), gdx.ravel()) \
                                 if ((not np.isnan(gy)) and (not np.isnan(gx)))],
                         low=-np.pi, high=np.pi))
    # Get unit vector
    avg_gradient = spstats.circmean(abs_gd[~np.isnan(abs_gd)], low=-np.pi, high=np.pi) 
    dirvec = (np.cos(avg_gradient), np.sin(avg_gradient))
    vhat = dirvec / np.linalg.norm(dirvec)

    grad_ = {'image': img,
             'magnitude': gradmag,
             'gradient_x': gdx,
             'gradient_y': gdy,
             'direction': abs_gd,
             'mean_deg': mean_dir, # DEG
             'mean_direction': avg_gradient, # RADIANS
             'vhat': vhat}
    
    return grad_


def calculate_gradients(curr_segmented_mask, img_az, img_el):

    thr_img_az = img_az.copy()
    thr_img_az[curr_segmented_mask==0] = np.nan
    grad_az = image_gradient(thr_img_az)

    thr_img_el = img_el.copy()
    thr_img_el[curr_segmented_mask==0] = np.nan
    grad_el = image_gradient(thr_img_el)

    return grad_az, grad_el


def plot_gradients_in_area(labeled_image, img_az, img_el, grad_az, grad_el, 
                           cmap_phase='nipy_Spectral', contour_lc='r', contour_lw=1,
                           spacing=200, scale=None, width=0.01, 
                            headwidth=5, vmin=-59, vmax=59):
    '''
    Retinomaps overlaid w/ gradient field, plus average gradient dir.
    '''
    fig, axn = pl.subplots(2,2, figsize=(5,6))

    # Maps ------------
    ax=axn[0, 0]
    im = ax.imshow(img_az,cmap=cmap_phase) #, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.7, label='azimuth (deg.)')
    #ax.set_title('azimuth')
    ax = overlay_all_contours(labeled_image, ax=ax, lw=contour_lw, lc=contour_lc)

    ax=axn[1, 0]
    im = ax.imshow(img_el, cmap=cmap_phase) #, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.7, label='elevation (deg.)')
    #ax.set_title('elevation')
    ax = overlay_all_contours(labeled_image, ax=ax, lw=contour_lw, lc=contour_lc)

    # Gradients ------------   
    ax=axn[0,0]
    #ax.imshow(thr_img_az, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    plot_gradients(grad_az, ax=ax, draw_interval=spacing, scale=scale, width=width,
                  headwidth=headwidth)
    ax=axn[1, 0]
    #ax.imshow(thr_img_el, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    plot_gradients(grad_el, ax=ax, draw_interval=spacing, scale=scale, width=width,
                  headwidth=headwidth)
    # Unit vectors ------------
    # Get average unit vector
    avg_dir_el = np.rad2deg(grad_el['mean_direction'])
    #print('[EL] avg dir: %.2f deg' % avg_dir_el)
    vhat_el = grad_el['vhat']
    avg_dir_az = np.rad2deg(grad_az['mean_direction'])
    #print('[AZ] avg dir: %.2f deg' % avg_dir_az)
    vhat_az = grad_az['vhat']

    ax= axn[0,1]
    ax.grid(True)
    vh = grad_az['vhat'].copy()
    edir_str = "u=(%.2f, %.2f), %.2f deg" % (vhat_az[0], vhat_az[1], avg_dir_az)
    ax.set_title('azimuth\n%s' % edir_str)
    ax.quiver(0,0, vhat_az[0], vhat_az[1],  scale=1, scale_units='xy', 
              units='xy', angles='xy', width=.05, pivot='tail')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.invert_yaxis()

    ax = axn[1,1]
    ax.grid(True)
    edir_str = "u=(%.2f, %.2f), %.2f deg" % (vhat_el[0], vhat_el[1], avg_dir_el)
    ax.set_title('elevation\n%s' % edir_str)
    ax.quiver(0,0, vhat_el[0], vhat_el[1],  scale=1, scale_units='xy', 
              units='xy', angles='xy', width=.05, pivot='tail')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    pl.subplots_adjust(wspace=0.5, hspace=0.5, right=0.8)

    return fig


def plot_gradients(grad_, ax=None, draw_interval=3, 
                   scale=1, width=0.005, toy=False, headwidth=5):
    '''
    Simple sub function to plot a given gradient, using info provided in dict
    grad_: (dict)
        Output of calculate_gradients()
    scale: # of data units per arrow length unit (smaller=longer arrow)
    weight = width of plot
    
    Note: Arrows should point TOWARD larger numbers
    angles='xy' (i.e., arrows point from (x,y) to (x+u, y+v))
    '''
    if ax is None:
        fig, ax = pl.subplots()
        
    gradimg = grad_['image']
    mean_dir = grad_['mean_deg']
    gdx = grad_['gradient_x']
    gdy = grad_['gradient_y']
    
    # Set limits and number of points in grid
    y, x = np.mgrid[0:gradimg.shape[0], 0:gradimg.shape[1]]

    # Every 3rd point in each direction.
    skip = (slice(None, None, draw_interval), slice(None, None, draw_interval))
    
    # plot
    ax.quiver(x[skip], y[skip], gdx[skip], gdy[skip], color='k',
              scale=scale, width=width,
              scale_units='xy', angles='xy', pivot='mid', units='width',
              headwidth=headwidth)

    gdir_ = grad_['direction'].copy()
    gmean = spstats.circmean(gdir_[~np.isnan(gdir_)], low=-np.pi, high=np.pi)
    avg_dir_grad = np.rad2deg(gmean) #np.rad2deg(grad_['direction'].mean())
    ax.set(aspect=1, title="Mean: %.2f\n(dir: %.2f)" % (mean_dir, avg_dir_grad))

    return ax


def label_roi_masks(seg_results, roi_masks):
    '''
    Expects roi_masks in shape (d1, d2, nrois)
    See identify_area_boundaries_2p.ipynb
    '''

    d1, d2, nrois = roi_masks.shape
    print("Roi masks:", d1, d2, nrois)
    roi_assignments={}
    for area_name, seg in seg_results['areas'].items():
        #seg_mask = cv2.resize(seg['mask'], (d2, d1))
        id_mask = seg['id'] * seg['mask'].astype(int)

        multi = roi_masks*id_mask[:,:,np.newaxis]
        curr_rois = np.where(multi.max(axis=0).max(axis=0)==seg['id'])[0]
        print(area_name, len(curr_rois))
        roi_assignments[area_name] = [int(r) for r in curr_rois]

    return roi_assignments #, roi_masks


def plot_labeled_rois(labeled_image, roi_assignments, roi_masks, cmap='colorblind', 
                        surface=None, ax=None, contour_lw=1, contour_lc='w'):
    
    d1, d2, nr = roi_masks.shape
    
    defined_names = [k for k in roi_assignments.keys() if not(hutils.isnumber(k))]

    color_list = sns.color_palette('colorblind', n_colors=len(defined_names))
    color_dict = dict((k, v) for k, v in zip(defined_names, color_list))

    # Plot rois on visual areas
    roi_int_img = np.zeros((d1, d2,4), dtype=float) #*np.nan

    for area_name, roi_list in roi_assignments.items():
        rc = color_dict[area_name] if area_name in defined_names else (0.5, 0.5, 0.5, 1)
        for ri in roi_list:
            curr_msk = roi_masks[:, :, ri].copy() #* color_list[0]
            roi_int_img[curr_msk>0, :] = [rc[0], rc[1], rc[2], 1] 

    if ax is None:
        fig, ax = pl.subplots(figsize=(3,4))
    
    if surface is not None:
        ax.imshow(surface, cmap='gray')
    ax.imshow(roi_int_img)
    ax = overlay_all_contours(labeled_image, ax=ax, lw=contour_lw, lc=contour_lc)

    lhandles = pplot.custom_legend_markers(colors=color_list, labels=defined_names)
    ax.legend(handles=lhandles, bbox_to_anchor=(1,1), loc='upper left')
    
    return ax



