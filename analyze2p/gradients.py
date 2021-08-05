#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 14:14:07 2021

@author: julianarhee
"""


import glob
import os
import cv2
import glob
import importlib
import h5py
import json

import seaborn as sns
import _pickle as pkl
import numpy as np
import pandas as pd
import pylab as pl

import analyze2p.utils as hutils
import analyze2p.plotting as pplot

import analyze2p.receptive_fields.utils as rfutils
import analyze2p.aggregate_datasets as aggr

import analyze2p.retinotopy.utils as retutils
import analyze2p.retinotopy.segment as seg
import analyze2p.extraction.rois as roiutils


# --------------------------------------------------------------------
# Creating background maps
# --------------------------------------------------------------------
def plot_results(sm_azim, sm_elev, fig=None, axn=None):
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


def get_background_maps(dk, experiment='rfs', traceid='traces001',
                        response_type='dff', is_neuropil=True, 
                        do_spherical_correction=False, 
                        create_new=False, redo_smooth=False, 
                        desired_radius_um=20,
                        target_sigma_um=20, smooth_spline_x=1, smooth_spline_y=1,
                        ds_factor=1, fit_thr=0.5):
    '''
    Load RF fitting results from tiles protocol. Create smoothed background maps
    for AZ and EL.

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
        return Non
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
        fig, axn = plot_results(sm_azim, sm_elev)
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


def cycle_and_load_maps(dk_list, target_sigma_um=20, desired_radius_um=20,
                        smooth_spline_x=1, smooth_spline_y=1, 
                        create_new=False, redo_smooth=False, is_neuropil=True,
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
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
            res = grd.get_background_maps(dk, experiment=experiment, traceid=traceid,
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


# --------------------------------------------------------------------
# MOVINGBAR - data loading and processing for global gradients
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
    print(d1, d2)
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


# --------------------------------------------------------------------
# Calculating gradients
# --------------------------------------------------------------------

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

#def change_of_basis(u1, u2):
#    B_2 = np.array([[u1[0], u2[0]],
#                    [u1[1], u2[1]]])
#    return B_2
#

def get_transformation_matrix(u1, u2):
    '''this is not the one to use'''
    a, b = u1[0], u1[1]
    c, d = u2[0], u2[1]

    a_21 = 1./ ( (-a*d/b) + c)
    a_11 = (-d*b) * a_21

    a_22 = 1./ ((-c*b/a)+d)
    a_12 = (-c*a) * a_22

    M = np.array( [ [a_11, a_12], [a_21, a_22] ])
    
    return M


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


def X_align_cortex_to_gradient(df, gvectors, xlabel='ml_pos', ylabel='ap_pos'):
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
    # Cartesian normal (plus error)
    o1 = np.array([1, 0]) + np.finfo(np.float32).eps
    o2 = np.array([0, 1]) + np.finfo(np.float32).eps
    # x-axis -- transform FOV coords to lie alone gradient axis
    T1 = roiutils.get_transformation_matrix(u1, o2)
    transf_x = [T1.dot(np.array([x, y])) for (x,y) \
                        in df[[xlabel, ylabel]].values]
    proj_ctx_x = np.array([p[0] for p in transf_x])
    # y-axis -- same
    T2 = roiutils.get_transformation_matrix(o1, u2)
    transf_y = [T2.dot(np.array([x, y])) for (x,y) \
                        in df[[xlabel, ylabel]].values]
    proj_ctx_y = np.array([p[1] for p in transf_y])
    # rename
    new_xlabel = '%s_proj' % xlabel.split('_')[0]
    new_ylabel = '%s_proj' % ylabel.split('_')[0]
    df_ = pd.DataFrame({new_xlabel: proj_ctx_x, 
                        new_ylabel: proj_ctx_y}, index=df.index)
    
    return df_, T1, T2


def get_projection_points(grad_az, grad_el):
    '''
    Use gradient info and FOV info of image (pixel locs) to  project pixel locations
    onto the direction of the normalized mean gradient vector.
    '''
    gimg_az = grad_az['image'].copy()
    gimg_el = grad_el['image'].copy()
    d1, d2 = grad_az['image'].shape

    vhat_az = grad_az['vhat'].copy() #[0], -0.04) #abs(grad_az['vhat'][1]))
    vhat_el = grad_el['vhat'].copy() #[0], -0.04) #abs(grad_az['vhat'][1]))

    proj_az = np.array([np.dot(np.array((xv, yv)), vhat_az) for yv in np.arange(0, d1) for xv in np.arange(0, d2)])
    ret_az = np.array([gimg_az[xv, yv] for xv in np.arange(0, d1) for yv in np.arange(0, d2)] )

    proj_el = np.array([np.dot(np.array((xv, yv)), vhat_el) for yv in np.arange(0, d1) for xv in np.arange(0, d2)])
    ret_el = np.array([gimg_el[xv, yv] for xv in np.arange(0, d1) for yv in np.arange(0, d2)] )

    pix = np.array([xv for yv in np.arange(0, d1) for xv in np.arange(0, d2) ])
    #coords = np.array([np.array((xv, yv)) for yv in np.arange(0, d1) for xv in np.arange(0, d2)])
    
    projections = {'proj_az': proj_az,
                   'proj_el': proj_el,
                   'retino_az': ret_az,
                   'retino_el': ret_el,
                   'pixel_ixs': pix}
    
    return projections 


# --------------------------------------------------------------------
# Model CTX vs RETINOTOPIC POS.
# --------------------------------------------------------------------
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
        regr_, linmodel = rfutils.do_linear_fit(xvs, yvs, model=model)
        regr_['cond'] = cond
        r_.append(regr_)
    regr_tiles = pd.concat(r_).reset_index(drop=True)

    return regr_tiles


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


#
# --------------------------------------------------------------------
# Predictions and whatnot
# --------------------------------------------------------------------

def load_soma_estimates(dk, experiment='rfs', retinorun='retino_run1', 
                        protocol='BAR', traceid='traces001',
                        response_type='dff', do_spherical_correction=False, fit_thr=0.5,
                        mag_thr=0.01):
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
        fitdf_all = rfutils.rfits_to_df(fit_results, fit_params, 
                                convert_coords=True, scale_sigma=True)
        fitdf_soma = fitdf_all[fitdf_all['r2']>fit_thr].copy()
        # Add position info
        fitdf_soma['cell'] = fitdf_soma.index.tolist()
        df = fitdf_soma.copy()
    return df



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




