#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 14:14:07 2021

@author: julianarhee
"""
import glob
import os
import sys
import optparse
import cv2
import glob
import importlib
import h5py
import json
import copy
import traceback
import _pickle as pkl
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import pylab as pl
import matplotlib as mpl
import scipy.stats as spstats
import analyze2p.receptive_fields.utils as rfutils
import analyze2p.extraction.rois as roiutils

import analyze2p.utils as hutils
import analyze2p.aggregate_datasets as aggr
import analyze2p.retinotopy.utils as retutils
import analyze2p.retinotopy.segment as seg

# In[3]:

warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------------------------------------------------
# Background retino maps
# --------------------------------------------------------------------
def get_background_maps(dk, experiment='rfs', traceid='traces001',
                        response_type='dff', is_neuropil=True, 
                        do_spherical_correction=False, 
                        create_new=False, redo_smooth=False, 
                        desired_radius_um=20,
                        target_sigma_um=20, smooth_spline_x=1, smooth_spline_y=1,
                        ds_factor=1, fit_thr=0.5):
    '''
    Load RF fitting results from TILES protocol. 
    Create smoothed background maps for AZ and EL (like done w BAR).

    Returns:
        res: (dict)
            Keys: 'azim_orig', 'azim_final', 'elev_orig', 'elev_final', 'fitdf'
    '''
    res=None
    try:
        fit_results, fit_params = rfutils.load_fit_results(dk, experiment=experiment,
                                traceid=traceid, response_type=response_type,
                                is_neuropil=is_neuropil,
                                do_spherical_correction=do_spherical_correction)
    except FileNotFoundError as e:
        print(" skipping %s" % dk)
        return None
    maps_outfile = os.path.join(fit_params['rfdir'], 'neuropil_maps.pkl')
    redo = create_new is True
    if not create_new:
        try:
            with open(maps_outfile, 'rb') as f:
                res = pkl.load(f)
            azim_np = res['azim_orig']
            elev_np = res['elev_orig']
        except Exception as e:
            redo=True
    if redo:
        redo_smooth=True
        fitdf_all = rfutils.rfits_to_df(fit_results, fit_params, 
                               convert_coords=True, scale_sigma=True)
        fitdf = fitdf_all[fitdf_all['r2']>fit_thr].copy()
        roi_list = fitdf.index.tolist()
        # Get masks
        zproj, dilated_masks, centroids = retutils.dilate_centroids(dk,
                                            desired_radius_um=desired_radius_um,
                                            traceid=traceid)
        ixs = np.sum(dilated_masks, axis=0)
        # Get maps
        azim_ = np.array([dilated_masks[i]*v for i, v \
                                    in enumerate(fitdf['x0'].values)])
        azim_np = np.true_divide(np.nansum(azim_, axis=0), ixs)
        elev_ = np.array([dilated_masks[i]*v for i, v \
                                    in enumerate(fitdf['y0'].values)])
        elev_np = np.true_divide(np.nansum(elev_, axis=0), ixs)
    
    if redo_smooth:
        # Smmooth
        pixel_size = hutils.get_pixel_size()
        sm_azim, sm_elev = seg.smooth_maps(azim_np, elev_np, 
                                target_sigma_um=target_sigma_um,  
                                smooth_spline=(smooth_spline_x, smooth_spline_y),
                                fill_nans=True,
                                start_with_transformed=False, 
                                use_phase_smooth=False, ds_factor=ds_factor)
        sm_azim.update({'input': azim_np})
        sm_elev.update({'input': elev_np})
        fig, axn = seg.plot_smoothing_results(sm_azim, sm_elev)
        fig.text(0.01, 0.9, dk)
        pl.savefig(os.path.join(fit_params['rfdir'], 'neuropil_maps.svg'))
        pl.close()
        res = {'azim_orig': azim_np, 
               'azim_final': sm_azim['final'],
               'elev_orig': elev_np, 
               'elev_final': sm_elev['final'],
               'fitdf': fitdf,
               'zproj': zproj}
        with open(maps_outfile, 'wb') as f:
            pkl.dump(res, f, protocol=2)

    return res

def cycle_and_load_maps(dk_list, experiment='rfs', traceid='traces001',
                        response_type='dff', do_spherical_correction=False,
                        is_neuropil=True,
                        target_sigma_um=20, desired_radius_um=20,
                        smooth_spline_x=1, smooth_spline_y=1, ds_factor=1,
                        create_new=False, redo_smooth=False,
                        verbose=False,
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Quick func to cycle dsets to create neuropil maps using TILE experiments.
    '''
#    target_sigma_um=40
#    desired_radius_um=20
#    smooth_spline_x=1
#    smooth_spline_y=1
#    create_new=False
#    redo_smooth=True
#    is_neuropil=True

    basedir = os.path.join(aggregate_dir, 'receptive-fields', 'neuropil')
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    tmp_masks_fpath = os.path.join(basedir, 'np_maps.pkl')

    try_loading = (create_new is False) and (redo_smooth is False)
    if try_loading:
        try:
            with open(tmp_masks_fpath, 'rb') as f:
                MAPS = pkl.load(f)
        except Exception as e:
            if verbose:
                traceback.print_exc()
            print("Cycling thru to get background maps.")
            create_new=True

    if create_new or redo_smooth:
        MAPS = dict() #dict((k, ) for k in dk_list)
        for dk in dk_list:
            res = get_background_maps(dk, experiment=experiment, traceid=traceid,
                            response_type=response_type, is_neuropil=is_neuropil,  
                            do_spherical_correction=do_spherical_correction,    
                            create_new=create_new, redo_smooth=redo_smooth, 
                            target_sigma_um=target_sigma_um, 
                            smooth_spline_x=smooth_spline_x, 
                            smooth_spline_y=smooth_spline_y, ds_factor=ds_factor)
            MAPS[dk] = res

        with open(tmp_masks_fpath, 'wb') as f:
            pkl.dump(MAPS, f, protocol=2)
            

    return MAPS

#---------------------------------------------------------------------
# Functions to load NP from MOVING BAR
# --------------------------------------------------------------------
def get_best_retinorun(datakey):
    all_retinos = retutils.get_average_mag_across_pixels(datakey)     
    retinorun = all_retinos.loc[all_retinos[1].idxmax()][0] 
    return retinorun

def load_movingbar_results(dk, retinorun, traceid='traces001',
                        rootdir='/n/coxfs01/2p-data'):
    # load retinodata
    retinoid, RETID = retutils.load_retino_analysis_info(
                        dk, run=retinorun, use_pixels=False)
    data_id = '_'.join([dk, retinorun, retinoid])
    print("DATA ID: %s" % data_id)
    scaninfo = retutils.get_protocol_info(dk, run=retinorun)

    # Image dimensions
    d2_orig = scaninfo['pixels_per_line']
    d1_orig = scaninfo['lines_per_frame']
    print("Original dims: [%i, %i]" % (d1_orig, d2_orig))
    ds_factor = int(RETID['PARAMS']['downsample_factor'])
    print('Data were downsampled by %i.' % ds_factor)
    # Get pixel size
    pixel_size = hutils.get_pixel_size()
    pixel_size_ds = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)
    d1 = int(d1_orig/ds_factor)
    d2 = int(d2_orig/ds_factor)
    #print(d1, d2)
    # Load fft 
    fft_results = retutils.load_fft_results(dk,
                                    retinorun=retinorun, traceid=traceid, 
                                    rootdir=rootdir, create_new=False,
                                    use_pixels=False)
    fft_soma = fft_results['fft_soma']
    fft_np = fft_results['fft_neuropil']
    # Create dataframe of magratios -- each column is a condition
    magratios_soma, phases_soma = retutils.extract_from_fft_results(fft_soma)
    magratios_np, phases_np = retutils.extract_from_fft_results(fft_np)
    dims = (d1_orig, d2_orig)
    
    return magratios_soma, phases_soma, magratios_np, phases_np, dims


def adjust_retinodf(mvb_np, mag_thr=0.02):
    '''
    Filter out cells that are not responsive (mag_thr),
    and convert phase to centered linear coords of screen.
    '''
    # Filter
    pass_mag_rois = mvb_np[(mvb_np.mag_az>mag_thr) 
                          & (mvb_np.mag_el>mag_thr)].index.tolist()
    retinodf_np = mvb_np.loc[pass_mag_rois]
    # Get screen info
    screen = hutils.get_screen_dims()
    screen2p_x = screen['azimuth_deg'] # 119.5564
    screen2p_y = screen['altitude_deg'] #67.323
    resolution2p = screen['resolution'] #[1920, 1080] #[1024, 768]
    # Convert to screen coords
    abs_vmin, abs_vmax = (-np.pi, np.pi)
    lmax_az_2p = screen2p_x #/2.
    lmin_az_2p = 0 #-screen2p_x #-lmax_az_2p
    lmax_el_2p = screen2p_y #/2.
    lmin_el_2p = 0 #-screen2p_y# 0 #-lmax_el_2p
    retinodf_np['az_lin'] = hutils.convert_range(retinodf_np['phase_az'], 
                                       newmin=lmin_az_2p, newmax=lmax_az_2p, 
                                       oldmin=abs_vmin, oldmax=abs_vmax)
    retinodf_np['el_lin'] = hutils.convert_range(retinodf_np['phase_el'], 
                                       newmin=lmin_az_2p, newmax=lmax_az_2p, 
                                       oldmin=abs_vmin, oldmax=abs_vmax)
    retinodf_np['x0'] = retinodf_np['az_lin'] - (lmax_az_2p/2.)
    retinodf_np['y0'] = retinodf_np['el_lin'] - (lmax_az_2p/2.)
    
    return retinodf_np


def get_neuropil_data(dk, retinorun, mag_thr=0.001, delay_map_thr=1.0, ds_factor=2,
                    visual_areas=['V1', 'Lm', 'Li']):
    '''
    Wrapper for loading neuropil data from movingbar.
    Loads FFT results and calculates final retino pref. estimates for each NP mask.
    Converts phase to linear (screen) coords in DEG visual angle.
    Adds CTX position info for each centroid (Assumes single VA for all cells,
    so should filter with seg.load_roi_assignments() after).
    Filters out poorly responding cells.

    Returns:  retinodf_np (pd.DataFrame)
    '''
    df = None
    # Load FFT results
    mags_soma, phases_soma, mags_np, phases_np, dims = load_movingbar_results(dk, 
                                                                              retinorun)
    # Get maps:  abs_vmin, abs_vmax = (-np.pi, np.pi)
    mvb_np = retutils.get_final_maps(mags_np, phases_np, 
                        trials_by_cond=None,
                        mag_thr=None, dims=dims,
                        ds_factor=ds_factor, use_pixels=False)
    # Filter bad responses
    df = adjust_retinodf(mvb_np.dropna(), mag_thr=mag_thr)
    # Add cell position info
    df = add_position_info(df, dk, 'retino', retinorun=retinorun)

    return df

# --------------------------------------------------------------------
# Gradient functions
# --------------------------------------------------------------------
def load_neuropil_background(datakey, retinorun, map_type='final', protocol='BAR',
                        rootdir='/n/coxfs01/2p-data'):
    '''
    Load NP background (MOVINGBAR). Saved by notebook
    >> ./retinotopy/identify_visual_areas.ipynb.
    TODO:  Clean up, so all this stuff in same place 
    (retino_structure - has projections_, vectors_.pkl
    vs. segmentation - has smoothed/processed maps)
    '''
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    maps_fpaths = glob.glob(os.path.join(rootdir, animalid, 
                        session, 'FOV%i_*' % fovnum, '%s*' % retinorun, 
                        'retino_analysis/segmentation/smoothed_maps.npz'))[0]
    maps = np.load(maps_fpaths)
    if map_type=='final':
        az_ = maps['azimuth']
        el_ = maps['elevation']
    else:
        az_ = maps['start_az']
        el_ = maps['start_el']
        
    # screen info
    screen = hutils.get_screen_dims()
    screen_max = screen['azimuth_deg']/2.
    screen_min = -screen_max
    #### Convert to screen units
    vmin, vmax = (-np.pi, np.pi)
    az_map = hutils.convert_range(az_, oldmin=vmin, oldmax=vmax, 
                                  newmin=screen_min, newmax=screen_max)
    el_map = hutils.convert_range(el_, oldmin=vmin, oldmax=vmax, 
                                  newmin=screen_min, newmax=screen_max)

    return az_map, el_map


def load_gradients(dk, va, retinorun='retino_run1', create_new=False,
                    rootdir='/n/coxfs01/2p-data'):
    '''
    Load gradient results for specified visual area and run.
    Results are in: <FOVDIR>/segmentation/results_VAREA.pkl

    Returns:
    
    results: (dict)
        'az/el_gradients': {
            image: gradient image (az)
            magnitude: np.sqrt(gdx**2 + gdy**2)
            gradient_x: gdx values
            gradient_y: gdy values
            direction: direction of gradient at each point
            mean_deg/_direction:  mean direction in DEG or RAD
            vhat: unit vector 
        }
        area_mask:  binary mask with 1's corresponding to specified visual area (va)
        retinorun:  best run (used to calculate grads)
        visual_area: area specified in fov
    '''
    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
    gradients_basedir = os.path.join(fovdir, 'segmentation')
    if not os.path.exists(gradients_basedir):
        os.makedirs(gradients_basedir) 
    gradients_fpath = os.path.join(gradients_basedir, 'gradient_results.pkl')
    if not create_new:
        try: 
            with open(gradients_fpath, 'rb') as f:
                results = pkl.load(f)
            assert va in results.keys(), "Area <%s> not in results. creating now." % va
        except Exception as e:
            create_new=True
        
    if create_new:
        # Load area segmentation results 
        seg_results, seg_params = seg.load_segmentation_results(dk, retinorun=retinorun)
        segmented_areas = seg_results['areas']
        region_props = seg_results['region_props']
        assert va in segmented_areas.keys(), \
            "Visual area <%s> not in region. Found: %s" % (va, str(segmented_areas.keys()))
        curr_area_mask = segmented_areas[va]['mask'].copy()
        # Load NP masks
        AZMAP_NP, ELMAP_NP = load_neuropil_background(dk, retinorun,
                                            map_type='final', protocol='BAR')
        # Calculate gradients
        grad_az, grad_el = seg.calculate_gradients(curr_area_mask, AZMAP_NP, ELMAP_NP)

        # Save
        results={}
        if os.path.exists(gradients_fpath):
            with open(gradients_fpath, 'rb') as f:
                results = pkl.load(f)
       
        curr_results = {'az_gradients': grad_az,
                        'el_gradients': grad_el,
                        'area_mask': curr_area_mask,
                        'retinorun': retinorun}
        results.update({va: curr_results})
        with open(gradients_fpath, 'wb') as f:
            pkl.dump(results, f, protocol=2)
            
    return results[va]

def plot_gradients(az_map, el_map, grad_az, grad_el,
                  spacing=200, scale=0.0001, width=0.01, headwidth=20):
    '''plot az and el maps + gradient vectors as unit'''
    fig, axn = pl.subplots(2,2, figsize=(5,6))
    for ai, cond in enumerate(['az', 'el']):
        ax=axn[ai,0]
        npmap = az_map.copy() if cond=='az' else el_map.copy()
        im = ax.imshow(npmap, cmap='Spectral')
        grad_ = grad_az.copy() if cond=='az' else grad_el.copy()
        seg.plot_gradients(grad_, ax=ax, draw_interval=spacing, 
                           scale=scale, width=width, headwidth=headwidth)
        ax= axn[ai,1]
        ax.grid(True)
        vhat_ = grad_['vhat'].copy()
        ax.quiver(0,0, vhat_[0], vhat_[1],  scale=1, scale_units='xy',
                  units='xy', angles='xy', width=.05, pivot='tail')
        ax.set_aspect('equal')
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])
        ax.set_title("%s: [%.2f, %.2f]" % (cond, vhat_[0], vhat_[1]))
        ax.invert_yaxis()
    
    return fig



# --------------------------------------------------------------------
# ALIGNMENT functions.
# --------------------------------------------------------------------
def project_onto_gradient_vectors(df_, u1, u2, xlabel='ml_proj', ylabel='ap_proj'):
    '''Project data position (ml_pos, al_pos) onto direction of gradients,
    specified by u1 (azimuth) and u2 (elevation)'''
    M = np.array([[u1[0], u1[1]],
                  [u2[0], u2[1]]])
    transf_vs = [M.dot(np.array([x, y])) for (x, y) \
                     in df_[['ml_pos', 'ap_pos']].values]
    t_df = pd.DataFrame(transf_vs, columns=[xlabel, ylabel], index=df_.index)

    return t_df, M


def align_cortex_to_gradient(df, gvectors, xlabel='ml_pos', ylabel='ap_pos'):
    '''
    Align FOV to gradient vector direction w transformation matrix.
    Use gvectors to align coordinates specified in df.
    Note: calculate separate for each axis.
    
    gvectors: dict()
        keys/vals: 'az': [v1, v2], 'el': [w1, w2]
    df: pd.DataFrame()
        coordi
    '''
    # Transform predicted-ctx pos back to FOV coords
    u1 = (gvectors['az'])
    u2 = (gvectors['el'])

    # Transform FOV coords to lie alone gradient axis
    transf_df, M = project_onto_gradient_vectors(df, u1, u2, 
                                    xlabel='ml_proj', ylabel='ap_proj')
    # rename

    return transf_df, M

def regress_cortex_and_retino_pos(df, xvar='pos', model='ridge'):
    '''
    Linear regression for each condition (az, el). 
    Return as dataframe.
    '''
    r_=[]
    for ai, cond in enumerate(['az', 'el']):
        ctx_label = 'ml' if cond=='az' else 'ap'
        ret_label = 'x0' if cond=='az' else 'y0'

        xvs = df['%s_%s' % (ctx_label, xvar)].values
        yvs = df['%s' % ret_label].values
        regr_, linmodel = do_linear_fit(xvs, yvs, model=model)
        regr_['cond'] = cond
        r_.append(regr_)
    regr_tiles = pd.concat(r_).reset_index(drop=True)

    return regr_tiles

# ALIGNMENT:  plotting -----------------------
def scatter_ctx_vs_retino_by_cond(df_, 
                             az_x='ml_pos', az_y='x0', el_x='ap_pos', el_y='y0',
                             xlabel='meas. CTX pos (um)', 
                             ylabel='meas. RET pos (deg)', scatter_kws={'s': 5}):
    '''Plot regression for azimuth (left) and elevation (right) for Y on X
    az_x:  x-axis for Azimuth plot.
    az_y:  y-axis for Azimuth plot.
    el_x = x-axis for Elevation plot
    el_y:  y-axis for Elevation plot.
    (Called: plot_regression_az_and_el in Nb)
    '''
    fig, axn = pl.subplots(1,2, figsize=(6.5,3))
    ax=axn[0]; ax.set_title('Azimuth');
    sns.regplot(az_x, az_y, df_, ax=ax, scatter_kws=scatter_kws)

    ax=axn[1]; ax.set_title('Elevation');
    sns.regplot(el_x, el_y, df_, ax=ax, scatter_kws=scatter_kws)
    for ax in axn:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    pl.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.8, wspace=0.5)
    return fig

def plot_regression_ctx_vs_retino_pos(df, regr):
    '''Plot AZ and EL results for ctx vs. retino pos'''
    fig, axn = pl.subplots(1,2, figsize=(8,4))
    for ai, (cond, regr_) in enumerate(regr.groupby(['cond'])):
        ax=axn[ai]
        ctx_label = 'ml' if cond=='az' else 'ap'
        ret_label = 'x0' if cond=='az' else 'y0'
        xvs = df['%s_proj' % ctx_label].values
        yvs = df['%s' % ret_label].values

        sns.regplot(xvs, yvs, ax=ax)
        fit_str = '(R2=%.2f) y=%.1fx + %.1f' \
                % (regr_['R2'], regr_['coefficient'], regr_['intercept'])
        ax.set_title('%s\n%s' %(cond, fit_str), loc='left', fontsize=12)
        ax.set_ylabel('retino pos')
        ax.set_xlabel('ctx pos')
    pl.subplots_adjust(left=0.1, right=0.8, bottom=0.3, hspace=0.5, top=0.8,
                      wspace=0.5)
    return fig

def plot_measured_and_aligned(aligned_np, REGR_NP, REGR_MEAS):
    fig, axn = pl.subplots(2,2, figsize=(6.5,6))
    for ai, cond in enumerate(['az', 'el']):
        regr_ = REGR_NP[REGR_NP.cond==cond].copy()
        ctx_ = 'ml' if cond=='az' else 'ap'
        ret_ = 'x' if cond=='az' else 'y'
        ax=axn[0, ai]
        sns.regplot('%s_pos' % ctx_, '%s0' % ret_, data=aligned_np, 
                    ax=ax, color='k', scatter_kws={'s':2}, label='measured')
        regr_meas_ = REGR_MEAS[REGR_MEAS.cond==cond].copy()
        regr_meas_str = 'y=%.2fx+%.2f (R2=%.2f)\npearson r=%.2f, p=%.2f'\
                        % (regr_meas_['coefficient'], regr_meas_['intercept'], 
                           regr_meas_['R2'], 
                           regr_meas_['pearson_r'], regr_meas_['pearson_p'])
        ax.set_title(regr_meas_str, loc='left', fontsize=8)
        ax=axn[1, ai]
        sns.regplot('%s_proj' % ctx_, '%s0' % ret_, data=aligned_np, 
                    ax=ax, color='m', scatter_kws={'s':2}, label='aligned')
        # show linear fit
        (slope, intercept), = regr_[['coefficient', 'intercept']].values
        xvs = aligned_np['%s_pos' % ctx_].values
        yvs = xvs*slope + intercept
        regr_str = 'y=%.2fx+%.2f (R2=%.2f)\npearson r=%.2f, p=%.2f'\
                        % (slope, intercept, regr_['R2'], 
                           regr_['pearson_r'], regr_['pearson_p'])
        ax.plot(xvs, yvs, 'r:', label='regression')
        ax.set_title(regr_str, loc='left', fontsize=8)
    pl.subplots_adjust(bottom=0.2, wspace=0.5, hspace=0.8, right=0.75, top=0.85)
    pl.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=6)
    return fig

# --------------------------------------------------------------------
# Model CTX vs RETINOTOPIC POS.
# --------------------------------------------------------------------
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.model_selection import train_test_split
import scipy.stats as spstats
import sklearn.metrics as skmetrics #import mean_squared_error

def fit_linear_regr(xvals, yvals, return_regr=False, model='ridge'):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    if model=='ridge':
        regr = Ridge()
    elif model=='Lasso':
        regr = Lasso()
    else:
        model = 'ols'
        regr = LinearRegression()
    if len(xvals.shape) == 1:
        xvals = np.array(xvals).reshape(-1, 1)
        yvals = np.array(yvals).reshape(-1, 1)
    else:
        xvals = np.array(xvals)
        yvals = np.array(yvals)
    if any([np.isnan(x) for x in xvals]) or any([np.isnan(y) for y in yvals]):
        print("NAN")
    regr.fit(xvals, yvals)
    fitv = regr.predict(xvals)
    if return_regr:
        return fitv.reshape(-1), regr
    else:
        return fitv.reshape(-1)

def do_linear_fit(xvs, yvs, model='ridge', di=0, verbose=False):
    fitv, regr = fit_linear_regr(xvs, yvs,
                            return_regr=True, model=model)
     
    rmse = np.sqrt(skmetrics.mean_squared_error(yvs, fitv))
    r2 = skmetrics.r2_score(yvs, fitv)
    pearson_r, pearson_p = spstats.pearsonr(xvs, yvs) 
    slope = float(regr.coef_)
    intercept = float(regr.intercept_)

    model_results = {'fitv': fitv, 
                     'regr': regr}

    regr_df = pd.DataFrame({ 
                      'R2': r2,
                      'RMSE': rmse,
                      'pearson_p': pearson_p,
                      'pearson_r': pearson_r,
                      'coefficient': slope, # float(regr.coef_), 
                      'intercept': intercept, #float(regr.intercept_)
                      }, index=[di])
    if verbose:
        print("~~~regr results: y = %.2f + %.2f (R2=%.2f)" % (slope, intercept, r2))

    return regr_df, model_results

def load_models(dk, va, rootdir='/n/coxfs01/2p-data'):
    REGR_NP=None
    try:
        session, animalid, fovn = hutils.split_datakey_str(dk)
        fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
        curr_dst_dir = os.path.join(fovdir, 'segmentation')
        alignment_fpath = os.path.join(curr_dst_dir, 'alignment.pkl')
        with open(alignment_fpath, 'rb') as f:
            regr_results = pkl.load(f)
        assert va in regr_results, "Visual area not found"
        REGR_NP = regr_results[va].copy()
    except Exception as e:
        return None
    
    return REGR_NP

def update_models(dk, va, REGR_NP, create_new=False, 
                    rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
    curr_dst_dir = os.path.join(fovdir, 'segmentation')
    alignment_fpath = os.path.join(curr_dst_dir, 'alignment.pkl')
    results={}
    if os.path.exists(alignment_fpath) and (create_new is False):
        with open(alignment_fpath, 'rb') as f:
            results = pkl.load(f)
        if not isinstance(results, dict):
            results={}

    results.update({va: REGR_NP})
    with open(alignment_fpath, 'wb') as f:
        pkl.dump(results, f, protocol=2)

    return

def add_position_info(df, dk, experiment, retinorun='retino_run1'):
    '''
    Correctly assign visual area to each cell based on segmentation results,
    specify RETINORUN for loading roi assignemnts.
    
    '''
    # Add pos info to NP masks
    df['cell'] = df.index.tolist()
    # Assign va to each cell
    roi_assignments = seg.load_roi_assignments(dk, retinorun=retinorun)
    df['visual_area'] = None
    for va, rois_ in roi_assignments.items():
        df.loc[df['cell'].isin(rois_), 'visual_area'] = str(va)
    # Add other meta info
    df = hutils.add_meta_to_df(df, {'datakey': dk,
                                    'experiment': experiment})
    df = aggr.add_roi_positions(df)

    return df

# --------------------------------------------------------------------
# Scatter analysis functions
# --------------------------------------------------------------------
def get_soma_data(dk, experiment='rfs', retinorun='retino_run1', 
                        protocol='TILE', traceid='traces001',
                        response_type='dff', do_spherical_correction=False, fit_thr=0.5,
                        mag_thr=0.01):
    '''
    Load SOMA data. **Specify RETINORUN for correct roi assignments 
    (even if protocol is TILE).
    ''' 
    df=None
    if protocol=='BAR':
        magthr_2p=0.001
        delay_map_thr=1.0
        ds_factor=2
        mags_soma, phases_soma, mags_np, phases_np, dims = load_movingbar_results(dk, 
                                                                    retinorun)
        # #### Get maps:  abs_vmin, abs_vmax = (-np.pi, np.pi)
        mvb_soma = retutils.get_final_maps(mags_soma, phases_soma, 
                            trials_by_cond=None,
                            mag_thr=magthr_2p, dims=dims,
                            ds_factor=ds_factor, use_pixels=False)
        # add pos
        mvb_soma['cell'] = mvb_soma.index.tolist()
        # adjust to LIN coords
        mvb_soma = adjust_retinodf(mvb_soma.dropna(), mag_thr=mag_thr)
        df = mvb_soma.copy()
    else:
        fit_results, fit_params = rfutils.load_fit_results(dk, experiment=experiment,
                        traceid=traceid, response_type=response_type,
                        is_neuropil=False,
                        do_spherical_correction=do_spherical_correction)
        if fit_results is None:
            return None

        fitdf_all = rfutils.rfits_to_df(fit_results, fit_params, 
                                convert_coords=True, scale_sigma=True)
        fitdf_soma = fitdf_all[fitdf_all['r2']>fit_thr].copy()
        # Add position info
        fitdf_soma['cell'] = fitdf_soma.index.tolist()
        df = fitdf_soma.copy()

    # Add pos info to NP masks
    df = add_position_info(df, dk, experiment, retinorun=retinorun)

    return df.reset_index(drop=True)

# predicted_rf_locs = slope*proj_locs + intercept
# predicted_ctx_locs = (actual_rf_locs - intercept) / slope
def predict_cortex_position(regr, cond='az', points=None):
    g_intercept = float(regr[regr.cond==cond]['intercept'])
    g_slope = float(regr[regr.cond==cond]['coefficient'])
    predicted_ctx_x = (points - g_intercept) / g_slope

    return predicted_ctx_x

def predict_retino_position(regr, cond='az', points=None):
    g_intercept = float(regr[regr.cond==cond]['intercept'])
    g_slope = float(regr[regr.cond==cond]['coefficient'])
    predicted_ret_x = (points * g_slope) + g_intercept

    return predicted_ret_x

def get_deviations(df):
    df['deg_scatter_az'] = abs(df['x0']-df['predicted_x0'])
    df['deg_scatter_el'] = abs(df['y0']-df['predicted_y0'])
    df['dist_scatter_az'] = abs(df['ml_proj']-df['predicted_ml_proj'])
    df['dist_scatter_el'] = abs(df['ap_proj']-df['predicted_ap_proj'])
    deviations = df[['cell', 'deg_scatter_az', 'dist_scatter_az']].copy()\
                    .rename(columns={'deg_scatter_az': 'deg_scatter', 
                                     'dist_scatter_az': 'dist_scatter'})
    deviations['axis'] = 'az'
    devE = df[['cell', 'deg_scatter_el', 'dist_scatter_el']].copy()\
                    .rename(columns={'deg_scatter_el': 'deg_scatter', 
                                     'dist_scatter_el': 'dist_scatter'})
    devE['axis'] = 'el'
    deviations = pd.concat([deviations, devE], axis=0).reset_index(drop=True)
    return deviations


#---------------------------------------------------------------------
# MAIN STEPS FOR SCATTER ANALYSIS 
#---------------------------------------------------------------------

def load_vectors(dk, va, create_new=False):
    retinorun = get_best_retinorun(dk)
    gresults = load_gradients(dk, va, retinorun, create_new=create_new)
    
    AZMAP_NP = gresults['az_gradients']['image']
    ELMAP_NP = gresults['el_gradients']['image']
    GVECTORS = {'az': gresults['az_gradients']['vhat'], 
                'el': gresults['el_gradients']['vhat']}

    return retinorun, AZMAP_NP, ELMAP_NP, GVECTORS

def plot_gradients(dk, va, retinorun, cmap='Spectral'):
    # Gradient plot
    spacing =200
    scale = 0.0001 #0.0001
    width = 0.01 #1 #0.01
    headwidth=20
    contour_lc='w'
    contour_lw=1

    # load
    gresults = load_gradients(dk, va, retinorun, create_new=False)
    grad_az = gresults['az_gradients']
    grad_el = gresults['el_gradients']
    AZMAP_NP = grad_az['image']
    ELMAP_NP = grad_el['image']
    area_mask = gresults['area_mask']

    #### Plot gradients
    fig = seg.plot_gradients_in_area(area_mask, AZMAP_NP, ELMAP_NP, 
                         grad_az, grad_el, cmap_phase=cmap,
                         contour_lc=contour_lc, contour_lw=contour_lw, 
                         spacing=spacing, 
                         scale=scale, width=width, headwidth=headwidth)
    pl.subplots_adjust(left=0.1, right=0.9, bottom=0.2, hspace=0.8, top=0.8)

    return fig

def visualize_scatter_on_fov(df_, AZMAP_NP, ELMAP_NP, zimg_r=None,
                cmap='Spectral', markersize=50, lw=0.6, alpha=1., 
                plot_true=True, plot_predicted=True, plot_lines=True):

    # Make sure we are in bounds of FOV
    max_ypos, max_xpos = AZMAP_NP.shape
    incl_plotdf = df_[(df_['predicted_ml_pos']>=0) \
                    & (df_['predicted_ml_pos']<=max_xpos)\
                    & (df_['predicted_ap_proj']>=0) \
                    & (df_['predicted_ap_proj']<=max_ypos)].copy()
    excl_ixs = [i for i in df_.index.tolist() if i not in incl_plotdf.index]
    #plotdf = df_.loc[excl_ixs].copy()
    #plotdf = incl_plotdf.iloc[0::].copy()
    plotdf=df_.copy()

    fig, axn = pl.subplots(1, 2, figsize=(7,5))
    for ax, cond in zip(axn, ['azimuth', 'elevation']):
        neuropil_map = AZMAP_NP.copy() if cond=='azimuth' else ELMAP_NP.copy()
        retino_label='x0' if cond=='azimuth' else 'y0'
        # Set color limits
        vmin = min([np.nanmin(neuropil_map), plotdf[retino_label].min()])
        vmax = max([np.nanmax(neuropil_map), plotdf[retino_label].max()])
        normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        ax.set_title(cond)
        if zimg_r is not None:
            ax.imshow(zimg_r, cmap='gray') #, vmin=abs_vmin, vmax=abs_vmax)
        ax.imshow(neuropil_map, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        if plot_true:
            # Plot soma
            sns.scatterplot(x='ml_pos', y='ap_pos', data=plotdf, ax=ax,
                    alpha=alpha, hue=retino_label, hue_norm=normalize, 
                    palette=cmap,
                    s=markersize, linewidth=lw, edgecolor='k', zorder=1000) 
        if plot_predicted:
            # Plot soma
            sns.scatterplot(x='predicted_ml_pos', y='predicted_ap_pos', 
                    data=plotdf, ax=ax,
                    alpha=alpha, hue=retino_label, hue_norm=normalize, 
                    palette=cmap,
                    s=markersize, linewidth=lw, edgecolor='w', zorder=1000) 
        if plot_lines:
            # Plot connecting line
            for (x1, y1), (x2, y2) in zip(plotdf[['predicted_ml_pos', \
                                                  'predicted_ap_pos']].values,
                           plotdf[['ml_pos', 'ap_pos']].values):
                ax.plot([x1, x2], [y1, y2], lw=0.5, markersize=0, color='k')
    for ax in axn:
        ax.legend_.remove()
        ax.axis('off')

    return fig


def get_gradient_results(dk, va, do_gradients=False, do_model=False, plot=True,
                    np_mag_thr=0.001, np_delay_map_thr=1., np_ds_factor=2,
                    cmap='Spectral', plot_dst_dir='/tmp', verbose=False,
                    create_new=False, rootdir='/n/coxfs01/2p-data'):  
    '''
    create_new to completely overwrite ALL found gradient results
    '''
    if plot:
        if not os.path.exists(plot_dst_dir):
            os.makedirs(plot_dst_dir)

    #### Load NEUROPIL BACKGROUND and GRADIENTS
    retinorun, AZMAP_NP, ELMAP_NP, GVECTORS = load_vectors(dk, va, create_new=do_gradients)
    if plot:
        fig = plot_gradients(dk, va, retinorun, cmap=cmap)
        fig.text(0.05, 0.95, 'Gradients, est. from MOVINGBAR (%s)' % dk)
        pl.savefig(os.path.join(plot_dst_dir, 'np_gradients.svg'))
        pl.close()

    #### Use NEUROPIL to estimate linear model
    try:
        REGR_NP = load_models(dk, va, rootdir=rootdir)
        assert REGR_NP is not None
    except Exception as e:
        plot=True
        do_model=True

    if do_model:
        # 1. Get retino data for NEUROPIL (background)
        retinodf_np = get_neuropil_data(dk, retinorun, mag_thr=np_mag_thr, 
                                            delay_map_thr=np_delay_map_thr, 
                                            ds_factor=np_ds_factor)
        assert retinodf_np is not None, 'ERROR: %s, %s - no df' % (dk, retinorun)

        # 2. Align FOV to gradient vector direction 
        aligned_, M = align_cortex_to_gradient(retinodf_np, GVECTORS,
                                          xlabel='ml_pos', ylabel='ap_pos')
        aligned_np = pd.concat([retinodf_np, aligned_], axis=1).dropna()
        # 3. Fit model
        REGR_NP = regress_cortex_and_retino_pos(aligned_np, \
                        xvar='proj', model='ridge')
        regr_np_meas = regress_cortex_and_retino_pos(aligned_np, \
                        xvar='pos', model='ridge')
        # Save
        update_models(dk, va, REGR_NP, create_new=create_new)
        if verbose:
            print("NEUROPIL, MEASURED:")
            print(regr_np_meas.to_markdown())
            print("NEUROPIL, ALIGNED:")
            print(REGR_NP.to_markdown())
        fig = plot_measured_and_aligned(aligned_np, REGR_NP, regr_np_meas)
        fig.text(0.01, 0.95, 'Aligned CTX. vs retino\n(BAR, Neuropil, %s)' % dk)
        figname = 'measured_vs_aligned_NEUROPIL'
        pl.savefig(os.path.join(plot_dst_dir, '%s.svg' % figname))
        pl.close()

    return retinorun, AZMAP_NP, ELMAP_NP, GVECTORS, REGR_NP

def get_aligned_soma(dk, retinorun, GVECTORS, REGR_NP, experiment='rfs',
                     traceid='traces001', protocol='TILE',
                    response_type='dff', do_spherical_correction=False,
                    verbose=False, plot=False, plot_dst_dir='/tmp'):
    #### Load soma
    df_soma = get_soma_data(dk, experiment=experiment, retinorun=retinorun, 
                                protocol=protocol, traceid=traceid,
                                response_type=response_type,
                                do_spherical_correction=do_spherical_correction)
    #### Align soma coords to gradient
    aligned_, M = align_cortex_to_gradient(df_soma, GVECTORS,
                                      xlabel='ml_pos', ylabel='ap_pos')
    aligned_soma = pd.concat([df_soma, aligned_], axis=1).dropna()\
                        .reset_index(drop=True)
    #### Align SOMA coords
    regr_soma_meas = regress_cortex_and_retino_pos(aligned_soma, 
                                                       xvar='pos', model='ridge')
    regr_soma_proj = regress_cortex_and_retino_pos(aligned_soma, 
                                                        xvar='proj', model='ridge')
    if verbose:
        print("SOMA, MEASURED:")
        print(regr_soma_meas.to_markdown())
        print("SOMA, ALIGNED:")
        print(regr_soma_proj.to_markdown())
    if plot:
        # PLOT, soma
        fig = plot_measured_and_aligned(aligned_soma, 
                            regr_soma_proj, regr_soma_meas)
        fig.text(0.01, 0.95, 'Measured vs Aligned CTX to RETINO pos (%s)' % dk)
        figname = 'measured_vs_aligned_SOMA'
        pl.savefig(os.path.join(plot_dst_dir, '%s.svg' % figname))
        pl.close()

    #### Predict CORTICAL position (from retino position)
    p_x = predict_cortex_position(REGR_NP, cond='az', 
                              points=aligned_soma['x0'].values)
    p_y = predict_cortex_position(REGR_NP, cond='el', 
                              points=aligned_soma['y0'].values)
    aligned_soma['predicted_ml_proj'] = p_x
    aligned_soma['predicted_ap_proj'] = p_y

    #### Predict RETINO position (from cortical position)
    r_x = predict_retino_position(REGR_NP, cond='az', 
                              points=aligned_soma['ml_proj'].values)
    r_y = predict_retino_position(REGR_NP, cond='el', 
                              points=aligned_soma['ap_proj'].values)
    aligned_soma['predicted_x0'] = r_x
    aligned_soma['predicted_y0'] = r_y

    #### Calculate inverse for visualizing on FOV
    pred_INV = [np.linalg.inv(M).dot(np.array([x, y])) for (x, y) \
                in aligned_soma[['predicted_ml_proj', 'predicted_ap_proj']].values]
    pred_inv_df = pd.DataFrame(pred_INV, columns=['pred_inv_x', 'pred_inv_y'], 
                          index=aligned_soma.index)
    aligned_soma['predicted_ml_pos'] = pred_inv_df['pred_inv_x']
    aligned_soma['predicted_ap_pos'] = pred_inv_df['pred_inv_y']
    
    return aligned_soma.reset_index(drop=True)

def do_visualization(dk, df_, AZMAP_NP, ELMAP_NP, traceid='traces001', 
                    markersize=50, lw=0.5, alpha=1, cmap='Spectral', 
                    plot_true=True, plot_predicted=True, plot_lines=True,
                    plot_dst_dir='/tmp'):
    # # Visualization
    zimg, masks, ctrs = roiutils.get_masks_and_centroids(dk, traceid=traceid)
    pixel_size = hutils.get_pixel_size()
    zimg_r = retutils.transform_2p_fov(zimg, pixel_size)
    fig = visualize_scatter_on_fov(df_, AZMAP_NP, ELMAP_NP, zimg_r=zimg_r,
                    cmap=cmap, markersize=markersize, lw=lw, alpha=alpha,
                    plot_true=plot_true, plot_predicted=plot_predicted, 
                    plot_lines=plot_lines)
    fig.text(0.01, 0.95, 'CTX vs RETINO positions - MEASURED (%s)' % dk)
    pl.savefig(os.path.join(plot_dst_dir, 'fov_true_v_predicted_scatter.svg'))
    pl.close()
    return



def do_scatter_analysis(dk, va, do_gradients=False, do_model=False,
                        np_mag_thr=0.001, np_delay_map_thr=1.0, 
                        np_ds_factor=2., 
                        response_type='dff', do_spherical_correction=False, 
                        experiment='rfs', traceid='traces001',
                        cmap='Spectral', plot=True,
                        rootdir='/n/coxfs01/2p-data', verbose=False,
                        create_new=False):
    '''
    create_new to completely overwrite scatter analysis results
    '''
    deviations=None

    scatter_kws={'s':2}

    #### Select output dirs
    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, \
                    'FOV%i_*' % fovn))[0]
    curr_dst_dir = os.path.join(fovdir, 'segmentation')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)

    try:
        retinorun, AZMAP_NP, ELMAP_NP, GVECTORS, REGR_NP = get_gradient_results(dk, va, 
                                do_gradients=do_gradients, do_model=do_model, 
                                np_mag_thr=np_mag_thr, 
                                np_delay_map_thr=np_delay_map_thr, 
                                np_ds_factor=np_ds_factor,
                                plot=plot, cmap=cmap, plot_dst_dir=curr_dst_dir,
                                verbose=verbose)  
        protocol = 'TILE' if 'rfs' in experiment else 'BAR'
        soma_dst_dir = os.path.join(curr_dst_dir, 'scatter_%s' % experiment)
        if not os.path.exists(soma_dst_dir):
            os.makedirs(soma_dst_dir)
        aligned_soma = get_aligned_soma(dk, retinorun, GVECTORS, REGR_NP, 
                                experiment=experiment,
                                traceid=traceid, protocol=protocol, 
                                response_type=response_type,
                                do_spherical_correction=do_spherical_correction,
                                verbose=verbose, plot=plot, plot_dst_dir=soma_dst_dir)

        if plot:
            # FOV scatter coords 
            markersize=50
            lw=0.6
            alpha=1
            plot_true=True
            plot_predicted=True
            plot_lines=True
            do_visualization(dk, aligned_soma, AZMAP_NP, ELMAP_NP, traceid=traceid, 
                            markersize=50, lw=0.5, alpha=1, cmap=cmap,
                            plot_true=True, plot_predicted=True, plot_lines=True,
                            plot_dst_dir=soma_dst_dir)


        # # Calculate scatter
        deviations = get_deviations(aligned_soma)
        if plot:
            # Plot
            fig, axn = pl.subplots(1,2, figsize=(6.5, 3))
            ax=axn[0]
            sns.histplot(deviations, x='deg_scatter', hue='axis', ax=ax,
                        stat='probability', cumulative=False )
            ax.set_title('Retino scatter (deg)')
            ax=axn[1]
            sns.histplot(deviations, x='dist_scatter', hue='axis', ax=ax,
                        stat='probability', cumulative=False)
            ax.set_title('Cortical scatter (um)')
            pl.subplots_adjust(left=0.1, right=0.8, bottom=0.25, top=0.85, 
                                wspace=0.5, hspace=0.5)
            pl.savefig(os.path.join(soma_dst_dir, 'deviations.svg'))
            pl.close()

        update_results(dk, va, deviations, soma_dst_dir, create_new=create_new)

    except Exception as e:
        print("ERROR in %s, %s" % (dk, retinorun))

        traceback.print_exc()
        
    return deviations

def update_results(dk, va, soma_results, soma_dst_dir, create_new=False):
    scatter_fpath = os.path.join(soma_dst_dir, 'scatter_results.pkl')

    results={}
    if os.path.exists(scatter_fpath) and (create_new is False):
        with open(scatter_fpath, 'rb') as f:
            results = pkl.load(f)
        if not isinstance(results, dict):
            results = {}

    results.update({va: soma_results})
    with open(scatter_fpath, 'wb') as f:
        pkl.dump(results, f, protocol=2)

    return

def load_results(dk, va, rootdir='/n/coxfs01/2p-data'):
    currdf=None
    try:
        session, animalid, fovn = hutils.split_datakey_str(dk)
        fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
        curr_dst_dir = os.path.join(fovdir, 'segmentation')
        results_fpath = os.path.join(curr_dst_dir, 'scatter_results.pkl')
        with open(results_fpath, 'rb') as f:
            results = pkl.load(f)
        assert va in results.keys(), 'Visual area not found in scatter analysis'
        currdf = results[va].copy()
    except Exception as e:
        return None
    
    return currdf


def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', 
                      default='/n/coxfs01/2p-data', 
                      help='root dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-G', '--aggr', action='store', dest='aggregate_dir', 
                      default='/n/coxfs01/julianarhee/aggregate-visual-areas', 
                      help='aggregate analysis dir [default: aggregate-visual-areas]')
    parser.add_option('--zoom', action='store', dest='fov_type', default='zoom2p0x', 
                      help="fov type (zoom2p0x)") 
    parser.add_option('--state', action='store', dest='state', default='awake', 
                      help="animal state (awake)") 
  
    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='rfs', 
                      help="experiment to calculate scatter (e.g,. rfs, rfs10, retino)") 
    parser.add_option('-i', '--datakey', action='store', dest='datakey', default='', 
                      help='datakey (YYYYMMDD_JCxx_fov1)')
    parser.add_option('-V', '--area', action='store', dest='visual_area', default=None, 
                      help='visual area to process (None, to do all)')

    parser.add_option('-t', '--traceid', action='store', dest='traceid', 
                      default='traces001', \
                      help="name of traces ID [default: traces001]")
       
    parser.add_option('-M', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")
    parser.add_option('--sphere', action='store_true', 
                        dest='do_spherical_correction', default=False, 
                        help="Flag to do fit on spherically-corrected response arrays")
    parser.add_option('--rf-thr', action='store', 
                        dest='rf_fit_thr', default=0.5, 
                        help="Fit thr for RF fits (default: 0.5)")
    # Neuropil 
    parser.add_option('--np-mag', action='store', 
                        dest='np_mag_thr', default=0.001,
                        help="Mag thr for neuropil retino (default: 0.001)")
    parser.add_option('--np-delay', action='store', 
                        dest='np_delay_map_thr', default=1., 
                        help="Delay map thr for neuropil retino (default: 1)")
    parser.add_option('--np-downsample', action='store', 
                        dest='np_ds_factor', default=2., 
                        help="Downsample factor for retino maps (default: 2)")

    parser.add_option('--gradients',  action='store_true', dest='do_gradients', default=False,
                      help="Recalculate gradients from NP image")
    parser.add_option('--model',  action='store_true', dest='do_model', default=False,
                      help="Refit model for retino-pos and ctx-pos on NP image")

    parser.add_option('--plot',  action='store_true', dest='plot', default=False,
                      help="plot and save figures")
    parser.add_option('--cmap',  action='store', dest='cmap', default='Spectral',
                      help="Colormap to use for background img (default: Spectral)")
    parser.add_option('-v', '--verbose',  action='store_true', dest='verbose', default=False,
                      help="verbose")


    parser.add_option('--all',  action='store_true', dest='cycle_all', default=False,
                      help="Set flag to cycle thru ALL dsets")

    (options, args) = parser.parse_args(options)

    return options



#### Select dataset and create output dirs
#dk = '20190617_JC097_fov1'
#va = 'V1'

#### RFs, Select parameters
#fit_thr=0.5
## NP background calc.
#np_mag_thr=0.001
#np_delay_map_thr=1.0
#np_ds_factor=2
#
#do_gradients=False
#do_model=False
#plot=True
#
#### Some plotting stuff
#cmap ='Spectral'

def main(options):

    optsE = extract_options(options)
    
    rootdir = optsE.rootdir
    datakey = optsE.datakey
    traceid = optsE.traceid
    
    dk = optsE.datakey
    va = None if optsE.visual_area in [None, 'None'] else optsE.visual_area
    experiment = optsE.experiment
    cmap= optsE.cmap

    # fit params
    do_gradients = optsE.do_gradients
    do_model = optsE.do_model
    plot = optsE.plot
    if do_gradients:
        do_model=True
        plot=True

    # RF params
    response_type = optsE.response_type
    do_spherical_correction = optsE.do_spherical_correction
   
    # NP background
    np_mag_thr = float(optsE.np_mag_thr) 
    np_delay_map_thr = float(optsE.np_delay_map_thr) 
    np_ds_factor = float(optsE.np_ds_factor) 

    verbose = optsE.verbose
    cycle_all = optsE.cycle_all

    if cycle_all:
        sdata = aggr.get_aggregate_info(visual_areas=['V1', 'Lm', 'Li'], 
                                        return_cells=False)
        meta = sdata[(sdata.experiment==experiment)]
        for (va, dk, experiment), g in meta.groupby(['visual_area', 'datakey', 'experiment']):
            print("Area: %s, all <%s> datasets" % (va, experiment))
            deviants = do_scatter_analysis(dk, va, experiment=experiment, 
                            do_gradients=do_gradients, do_model=do_model,
                            np_mag_thr=np_mag_thr, np_delay_map_thr=np_delay_map_thr, 
                            np_ds_factor=np_ds_factor,
                            response_type=response_type, 
                            do_spherical_correction=do_spherical_correction,
                            traceid=traceid,
                            cmap=cmap, plot=plot, verbose=verbose)
    else:
        assert dk is not None, "Must specify datakey" 
        if (va is None):
            sdata = aggr.get_aggregate_info(visual_areas=['V1', 'Lm', 'Li'], 
                                        return_cells=False)
            meta = sdata[(sdata.datakey==dk) & (sdata.experiment==experiment)]
            found_areas = meta['visual_area'].unique()
        else:
            found_areas = [va]
        for va in found_areas:
            print("Processing: %s, %s" % (va, dk))
            deviants = do_scatter_analysis(dk, va, 
                            do_gradients=do_gradients, do_model=do_model,
                            np_mag_thr=np_mag_thr, np_delay_map_thr=np_delay_map_thr, 
                            np_ds_factor=np_ds_factor,
                            response_type=response_type, 
                            do_spherical_correction=do_spherical_correction,
                            experiment=experiment, traceid=traceid,
                            cmap=cmap, plot=plot, verbose=verbose)


    print("Done.")

if __name__ == '__main__':

    main(sys.argv[1:])


#def get_transformation_matrix(u1, u2):
#    '''this is not the one to use'''
#    a, b = u1[0], u1[1]
#    c, d = u2[0], u2[1]
#
#    a_21 = 1./ ( (-a*d/b) + c)
#    a_11 = (-d*b) * a_21
#
#    a_22 = 1./ ((-c*b/a)+d)
#    a_12 = (-c*a) * a_22
#
#    M = np.array( [ [a_11, a_12], [a_21, a_22] ])
#    
#    return M
#
#def X_align_cortex_to_gradient(df, gvectors, xlabel='ml_pos', ylabel='ap_pos'):
#    '''
#    Align FOV to gradient vector direction w transformation matrix.
#    Use gvectors to align coordinates specified in df.
#    Note: calculate separate for each axis.
#    
#    gvectors: dict()
#        keys/vals: 'az': [v1, v2], 'el': [w1, w2]
#    df: pd.DataFrame()
#        coordi
#    '''
#    # Transform predicted-ctx pos back to FOV coords
#    u1 = (gvectors['az'])
#    u2 = (gvectors['el'])
#    # Cartesian normal (plus error)
#    o1 = np.array([1, 0]) + np.finfo(np.float32).eps
#    o2 = np.array([0, 1]) + np.finfo(np.float32).eps
#    # x-axis -- transform FOV coords to lie alone gradient axis
#    T1 = roiutils.get_transformation_matrix(u1, o2)
#    transf_x = [T1.dot(np.array([x, y])) for (x,y) \
#                        in df[[xlabel, ylabel]].values]
#    proj_ctx_x = np.array([p[0] for p in transf_x])
#    # y-axis -- same
#    T2 = roiutils.get_transformation_matrix(o1, u2)
#    transf_y = [T2.dot(np.array([x, y])) for (x,y) \
#                        in df[[xlabel, ylabel]].values]
#    proj_ctx_y = np.array([p[1] for p in transf_y])
#    # rename
#    new_xlabel = '%s_proj' % xlabel.split('_')[0]
#    new_ylabel = '%s_proj' % ylabel.split('_')[0]
#    df_ = pd.DataFrame({new_xlabel: proj_ctx_x, 
#                        new_ylabel: proj_ctx_y}, index=df.index)
#    
#    return df_, T1, T2
