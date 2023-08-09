#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:10:13 2019

@author: julianarhee
"""
#%%
import matplotlib as mpl
mpl.use('agg')
import os
import glob
import json
import copy
import sys
import optparse
import shutil
import traceback
import time
import inspect

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl


from pipeline.python.retinotopy import fit_2d_rfs as fitrf

from shapely.geometry.point import Point
from shapely import affinity
import multiprocessing as mp

#%%

# ############################################
# Functions for processing visual field coverage
# ############################################
           
def group_configs(group, response_type):
    '''
    Takes each trial's reponse for specified config, and puts into dataframe
    '''
    config = group['config'].unique()[0]
    group.index = np.arange(0, group.shape[0])

    return pd.DataFrame(data={'%s' % config: group[response_type]})
 
def bootstrap_rf_params(rdf, response_type='dff', fit_params={},
                        row_vals=[], col_vals=[], sigma_scale=2.35,
                        n_resamples=10, n_bootstrap_iters=1000,
                        do_spherical_correction=False):     

    do_spherical_correction=fit_params['do_spherical_correction']
    row_vals = fit_params['row_vals']
    col_vals = fit_params['col_vals']
    n_resamples = fit_params['evaluation']['n_resamples']
    n_bootstrap_iters = fit_params['evaluation']['n_bootstrap_iters']
 
    paramsdf = None
    try:
        if not terminating.is_set():
            time.sleep(1)
            
        #xres = np.unique(np.diff(row_vals))[0]
        #yres = np.unique(np.diff(col_vals))[0]
        xres=1 if do_spherical_correction else float(np.unique(np.diff(row_vals)))
        yres=1 if do_spherical_correction else float(np.unique(np.diff(col_vals)))
        sigma_scale=1 if do_spherical_correction else sigma_scale
        sigma_scale=1 if do_spherical_correction else sigma_scale
        min_sigma=2.5; max_sigma=50;

        if do_spherical_correction:
            grid_points, cart_values, sphr_values = fitrf.coordinates_for_transformation(fit_params)

        # Get all trials for each config (indices = trial reps, columns = conditions)
        grouplist = [group_configs(group, response_type) \
                        for config, group in rdf.groupby(['config'])]
        responses_df = pd.concat(grouplist, axis=1) 

        # Get mean response across re-sampled trials for each condition 
        # (i.e., each position). Do this n-bootstrap-iters times
        boot_ = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) 
                                for ni in range(n_bootstrap_iters)], axis=1)

        # Reshape array so that it matches for fitrf changs 
        # (should match .reshape(ny, nx))
        nx=len(col_vals)
        ny=len(row_vals)
        bootdf = boot_.apply(fitrf.reshape_array_for_nynx, args=(nx, ny))

        # Fit receptive field for each set of bootstrapped samples 
        bparams = []; #x0=[]; y0=[];
        for ii in bootdf.columns:
            response_vector = bootdf[ii].values
            # nx=len(col_vals), ny=len(row_vals)
            rfmap = fitrf.get_rf_map(response_vector, nx, ny) 
            fitr, fit_y = fitrf.do_2d_fit(rfmap, nx=nx, ny=ny) 
            if fitr['success']:
                amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
                if do_spherical_correction:
                    # Correct for spher correction, if nec
                    x0_f, y0_f, sigx_f, sigy_f = fitrf.get_scaled_sigmas(
                                                        grid_points, sphr_values,
                                                        x0_f, y0_f,
                                                        sigx_f, sigy_f, theta_f,
                                                        convert=True)
                    fitr['popt'] = (amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f) 
                if any(s<min_sigma for s \
                        in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale])\
                        or any(s > max_sigma for s \
                        in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale]):
                    fitr['success'] = False
                
            # If the fit for current bootstrap sample is good, 
            # add it to dataframe of bootstrapped rf params
            if fitr['success']:
                #amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
                curr_fit_results = list(fitr['popt'])
                curr_fit_results.append(fitr['r2'])
                bparams.append(tuple(curr_fit_results)) #(fitr['popt'])
        if len(bparams)==0:
            return None

        bparams = np.array(bparams)   
        paramsdf = pd.DataFrame(data=bparams, 
            columns=['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset', 'r2'])
        paramsdf['cell'] = [rdf.index[0] for _ in range(bparams.shape[0])]
   
    except KeyboardInterrupt:
        print("----exiting----")
        terminating.set()
        print("---set terminating---")

    return paramsdf

#%%
# --------------------------------------------------------
# Bootstrap (and corresponding pool/mp functions)
# --------------------------------------------------------

from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    pool.join()
  
def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_

def pool_bootstrap(rdf_list, params, do_spherical_correction, n_processes=1):   
    #try:
    results = []# None
    terminating = mp.Event()
        
    pool = mp.Pool(initializer=initializer, initargs=(terminating, ), 
                    processes=n_processes)
    try:
        results = pool.map_async(partial(bootstrap_rf_params, 
                            fit_params=params), 
                        rdf_list).get(99999999)
        #pool.close()
    except KeyboardInterrupt:
        print("**interupt")
        pool.terminate()
        print("***Terminating!")
    finally:
        pool.close()
        pool.join()
       
    return results
    

def evaluate_rfs(roidf_list, fit_params, n_processes=1):
    '''
    Evaluate receptive field fits for cells with R2 > fit_thr.

    Returns:
        eval_results = {'data': bootdata, 'params': bootparams, 'cis': bootcis}
        
        bootdata : dataframe containing results of param fits for bootstrap iterations
        bootparams: params used to do bootstrapping
        cis: confidence intervals of fit params

        If no fits, returns {}

    '''
    # Create output dir for bootstrap results

    # Get params        
    eval_results = {}
    scale_sigma = fit_params['scale_sigma']
    sigma_scale = fit_params['sigma_scale'] if scale_sigma else 1.0
    response_type = fit_params['response_type']
             
    #print("... doing bootstrap analysis for param fits.")
    start_t = time.time()
    bootstrap_results = pool_bootstrap(roidf_list, fit_params, n_processes=n_processes)
    end_t = time.time() - start_t
    print "Multiple processes: {0:.2f}sec".format(end_t)
    print "--- %i results" % len(bootstrap_results)

    if len(bootstrap_results)==0:
        return eval_results #None

    # Create dataframe of bootstrapped data
    bootdata = pd.concat(bootstrap_results)
   
    if do_spherical_correction is False: 
        xx, yy, sigx, sigy = fitrf.convert_fit_to_coords(bootdata, 
                                                         fit_params['row_vals'], 
                                                         fit_params['col_vals'])
        bootdata['x0'] = xx
        bootdata['y0'] = yy
        bootdata['sigma_x'] = sigx
        bootdata['sigma_y'] = sigy

    bootdata['sigma_x'] = bootdata['sigma_x'] * sigma_scale
    bootdata['sigma_y'] = bootdata['sigma_y'] * sigma_scale
    theta_vs = bootdata['theta'].values.copy()
    bootdata['theta'] = theta_vs % (2*np.pi)

    # Calculate confidence intervals
    bootdata = bootdata.dropna()
    bootcis = get_cis_for_params(bootdata, ci=ci)

    # Plot bootstrapped distn of x0 and y0 parameters for each roi (w/ CIs)
    counts = bootdata.groupby(['cell']).count()['x0']
    unreliable = counts[counts < n_bootstrap_iters*0.5].index.tolist()
    print("%i cells seem to have <50%% iters with fits" % len(unreliable))
    
    eval_results = {'data': bootdata, 
                    'params': bootparams, 
                    'cis': bootcis, 
                    'unreliable': unreliable}

    # Update params if re-did evaluation
    #eval_params_fpath = os.path.join(evaldir, 'evaluation_params.json')
    #save_params(eval_params_fpath, eval_params)
    #print("... updated eval params")
        
    # Save
    save_eval_results(eval_results, fit_params)

    #%% Identify reliable fits 
    if eval_results is not None:
        meas_df = estats.fits.copy()
        pass_cis = check_reliable_fits(meas_df, eval_results['cis']) 
        eval_results.update({'pass_cis': pass_cis})
        
        # Save results
        with open(rf_eval_fpath, 'wb') as f:
            pkl.dump(eval_results, f, protocol=pkl.HIGHEST_PROTOCOL)
   
    return eval_results

def save_eval_results(eval_results, fit_params):
    rfdir = fit_params['rfdir']
    evaldir = os.path.join(rfdir, 'evaluation')
    rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    return
 

#%%
# ############################################################################
# Functions for receptive field fitting and evaluation
# ############################################################################
def plot_bootstrapped_position_estimates(x0, y0, true_x, true_y, ci=0.95):
    lower_x0, upper_x0 = hutils.get_empirical_ci(x0, ci=ci)
    lower_y0, upper_y0 = hutils.get_empirical_ci(y0, ci=ci)

    fig, axes = pl.subplots(1, 2, figsize=(5,3))
    ax=axes[0]
    ax.hist(x0, color='k', alpha=0.5)
    ax.axvline(x=lower_x0, color='k', linestyle=':')
    ax.axvline(x=upper_x0, color='k', linestyle=':')
    ax.axvline(x=true_x, color='r', linestyle='-')
    ax.set_title('x0 (n=%i)' % len(x0))
    
    ax=axes[1]
    ax.hist(y0, color='k', alpha=0.5)
    ax.axvline(x=lower_y0, color='k', linestyle=':')
    ax.axvline(x=upper_y0, color='k', linestyle=':')
    ax.axvline(x=true_y, color='r', linestyle='-')
    lower_y0, upper_y0 = hutils.get_empirical_ci(y0, ci=ci)

    fig, axes = pl.subplots(1, 2, figsize=(5,3))
    ax=axes[0]
    ax.hist(x0, color='k', alpha=0.5)
    ax.axvline(x=lower_x0, color='k', linestyle=':')
    ax.axvline(x=upper_x0, color='k', linestyle=':')
    ax.axvline(x=true_x, color='r', linestyle='-')
    ax.set_title('x0 (n=%i)' % len(x0))
    
    ax=axes[1]
    ax.hist(y0, color='k', alpha=0.5)
    ax.axvline(x=lower_y0, color='k', linestyle=':')
    ax.axvline(x=upper_y0, color='k', linestyle=':')
    ax.axvline(x=true_y, color='r', linestyle='-')
    ax=axes[0]
    ax.hist(x0, color='k', alpha=0.5)
    ax.axvline(x=lower_x0, color='k', linestyle=':')
    ax.axvline(x=upper_x0, color='k', linestyle=':')
    ax.axvline(x=true_x, color='r', linestyle='-')
    ax.set_title('x0 (n=%i)' % len(x0))
    
    ax=axes[1]
    ax.hist(y0, color='k', alpha=0.5)
    ax.axvline(x=lower_y0, color='k', linestyle=':')
    ax.axvline(x=upper_y0, color='k', linestyle=':')
    ax.axvline(x=true_y, color='r', linestyle='-')
    ax.set_title('y0 (n=%i)' % len(y0))
    pl.subplots_adjust(wspace=0.5, top=0.8)
    
    return fig

def plot_bootstrapped_distribution(boot_values, true_x, ci=0.95, ax=None, param_name=''):
    lower_x0, upper_x0 = hutils.get_empirical_ci(boot_values, ci=ci)

    if ax is None:
        fig, ax = pl.subplots()
    ax.hist(boot_values, color='k', alpha=0.5)
    ax.axvline(x=lower_x0, color='k', linestyle=':')
    ax.axvline(x=upper_x0, color='k', linestyle=':')
    ax.axvline(x=true_x, color='r', linestyle='-')
    ax.set_title('%s (n=%i)' % (param_name, len(boot_values)))
   
    return ax

def plot_roi_evaluation(rid, meas_df, _fitr, _bootdata, ci=0.95, 
                             scale_sigma=True, sigma_scale=2.35):
    
    fig, axn = pl.subplots(2,3, figsize=(10,6))
    ax = axn.flat[0]
    ax = fitrf.plot_rf_map(_fitr['data'], cmap='inferno', ax=ax)
    ax = fitrf.plot_rf_ellipse(_fitr['fit_r'], ax=ax, scale_sigma=scale_sigma)
    params = ['sigma_x', 'sigma_y', 'theta', 'x0', 'y0']
    ai=0
    for param in params:
        ai += 1
        try:
            ax = axn.flat[ai]
            ax = plot_bootstrapped_distribution(_bootdata[param], meas_df[param][rid], 
                                                    ci=ci, ax=ax, param_name=param)
            pl.subplots_adjust(wspace=0.7, hspace=0.5, top=0.8)
            fig.suptitle('rid %i' % rid)
        except Exception as e:
            print("!! eval error (plot_boot_distn): rid %i, param %s" % (rid, param))
            #traceback.print_exc()
            
    return fig


def get_reliable_fits(pass_cis, pass_criterion='all', single=False):
    '''
    single: (bool)
        Set flag to only check whether 1 criterion is good, specfied
        with pass_criterion

    pass_criterion: (str)
        all - only pass if ALL fit params within CIs (this can be weird for theta)
        any - just 1 param needs to be within CIs
        position - just check for position (x0, y0)
        size - only check for size estimates (sigma_x, sigma_y)
        most - just check that majority (>50%) of the parameters pass

    '''
    if single is True:
        keep_rids = [i for i in pass_cis.index.tolist() \
                        if pass_cis[pass_criterion][i]==True]
    else:       
        param_cols = [p for p in pass_cis.columns if p!='cell']
        if pass_criterion=='all':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() if all(tmp_ci.loc[i])]
        elif pass_criterion=='any':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() if any(tmp_ci.loc[i])]
        elif pass_criterion=='size':
            keep_rids = [i for i in pass_cis.index.tolist() 
                        if (pass_cis['sigma_x'][i]==True \
                        and pass_cis['sigma_y'][i]==True)]
        elif pass_criterion=='position':
            keep_rids = [i for i in pass_cis.index.tolist() 
                        if (pass_cis['x0'][i]==True and pass_cis['y0'][i]==True)]
        elif pass_criterion=='most':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() 
                        if sum([pv==True 
                        for pv in tmp_ci.loc[rid]])/float(len(param_cols))>0.5]
        else:   
            keep_rids = [i for i in pass_cis.index.tolist() if any(pass_cis.loc[i])]
       
    pass_df = pass_cis.loc[keep_rids]
 
    reliable_rois = sorted(pass_df.index.tolist())

    return reliable_rois

 
def check_reliable_fits(meas_df, boot_cis): 
    # Test which params lie within 95% CI
    params = [p for p in meas_df.columns.tolist() if p!='r2']
    pass_cis = pd.concat([pd.DataFrame(
            [boot_cis['%s_lower' % p][ri]<=meas_df[p][ri]<=boot_cis['%s_upper' % p][ri] \
            for p in params], columns=[ri], index=params) \
            for ri in meas_df.index.tolist()], axis=1).T
       
    return pass_cis


def get_cis_for_params(bdata, ci=0.95):
    roi_list = [roi for roi, bdf in bdata.groupby(['cell'])]
    param_names = [p for p in bdata.columns if p != 'cell']
    CI = {}
    for p in param_names:
        CI[p] = dict((roi, hutils.get_empirical_ci(bdf[p].values, ci=ci)) \
                        for roi, bdf in bdata.groupby(['cell']))
    
    cis = {}
    for p in param_names:
        cvals = np.array([hutils.get_empirical_ci(bdf[p].values, ci=ci) \
                        for roi, bdf in bdata.groupby(['cell'])])
        cis['%s_lower' % p] = cvals[:, 0]
        cis['%s_upper' % p] = cvals[:, 1]
    cis = pd.DataFrame(cis, index=[roi_list])
    
    return cis
    
def visualize_bootstrapped_params(bdata, sorted_rois=[], sorted_values=[], 
                                    nplot=20, rank_type='R2'):
    if sorted_rois is None:
        sorted_rois = bdata['cell'].unique()[0:nplot]
        rank_type = 'no rank'
    
    nplot = 20
    dflist = []
    for roi, d in bdata.groupby(['cell']): #.items():
        if roi not in sorted_rois[0:nplot]:
            continue
        tmpd = d.copy()
        tmpd['cell'] = [roi for _ in range(len(tmpd))]
        tmpd['rank'] = [sorted_values[roi] for _ in range(len(tmpd))]
        dflist.append(tmpd)
    df = pd.concat(dflist, axis=0)
    df['theta'] = [np.rad2deg(theta) % 360. for theta in df['theta'].values]
        
    fig, axes = pl.subplots(2,3, figsize=(15, 5))
    sns.boxplot(x='rank', y='amp', data=df, ax=axes[0,0])
    sns.boxplot(x='rank', y='x0', data=df, ax=axes[0,1])
    sns.boxplot(x='rank', y='y0', data=df, ax=axes[0,2])
    sns.boxplot(x='rank', y='theta', data=df, ax=axes[1,0])
    sns.boxplot(x='rank', y='sigma_x', data=df, ax=axes[1,1])
    sns.boxplot(x='rank', y='sigma_y', data=df, ax=axes[1,2])
    for ax in axes.flat:
        ax.set_xticks([]) 
        ax.set_xlabel('')
        sns.despine(ax=ax, trim=True, offset=2)
    fig.suptitle('bootstrapped param distNs (top 20 cells by %s)' % rank_type)
    
    return fig


# ----------------------------------------------------------------------     
#%% FITTING FUNCTIONS
# ----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import scipy.stats as spstats
import sklearn.metrics as skmetrics #import mean_squared_error

def regplot(x, y, data=None, x_estimator=None, x_bins=None, x_ci="ci",
            scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None,
            order=1, logistic=False, lowess=False, robust=False,
            logx=False, x_partial=None, y_partial=None,
            truncate=False, dropna=True, x_jitter=None, y_jitter=None,
            label=None, color=None, marker="o",
            scatter_kws=None, line_kws=None, ax=None):
    '''
    Adjust regplot from Seaborn to return data (to access CIs)
    '''
    plotter = sns.regression._RegressionPlotter(x, y, data, x_estimator, x_bins, x_ci,
                                 scatter, fit_reg, ci, n_boot, units,
                                 order, logistic, lowess, robust, logx,
                                 x_partial, y_partial, truncate, dropna,
                                 x_jitter, y_jitter, color, label)
    if ax is None:
        ax = pl.gca()
        
    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax, plotter

def do_regr_on_fov_cis(bootdata, bootcis, posdf, reliable_rois=[],
                    cond='azimuth', 
                    xaxis_lim=None, ci=.95,  model='ridge', 
                   deviant_color='dodgerblue', marker='o', marker_size=20,
                   plot_boot_med=False, fill_marker=True, plot_all_cis=False):

   
    '''
    Identify RELIABLE cells based on pass criterion for x0, y0 (position test).

    Plot "scatter":
    
    1. Mark all ROIs with fit (R2>0.5)
    2. Linear regression + CI (based off of seaborn's function)
    3. Mark cells with reliable fits (R2>0.5 + measured value w/in 95% CI)
    4. Mark reliable cells w/ CI outside of linear fit (1).
    
    ''' 
    axname = 'xpos' if cond=='azimuth' else 'ypos'
    parname = 'x0' if cond=='azimuth' else 'y0'

    fig, ax = pl.subplots(figsize=(10,8)); ax.set_title(cond);
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    if xaxis_lim is not None:
        ax.set_xlim([0, xaxis_lim])
    else:
        ax.set_xlim([0, 1200])
       
    # 1. Identify which cells fail bootstrap fits - do not include in fit.
    fail_rois = [r for r in posdf.index.tolist() if r not in reliable_rois] 
    fail_df = posdf.loc[fail_rois].copy()
    sns.regplot('%s_fov' % axname, '%s_rf' % axname, data=fail_df, ax=ax, 
                label='unreliable (R2>0.5)', color='gray', marker='x', fit_reg=False,
                scatter_kws=dict(s=marker_size, alpha=0.5))

    # 2a. Linear regression, include cells with reliable fits (R2 + 95% CI) 
    scatter_kws = dict(s=marker_size, alpha=1.0, facecolors='k')
    if not fill_marker:
        scatter_kws.update({'facecolors':'none', 'edgecolors':'k'})
    ax, plotter = regplot('%s_fov' % axname, '%s_rf' % axname,  ax=ax,
                          data=posdf.loc[reliable_rois], ci=ci*100, 
                          color='k', marker=marker, 
                          scatter_kws=scatter_kws, 
                          label='reliable (%i%% CI)' % int(ci*100)) 

    # 2b. Get CIs from linear fit (fit w/ reliable rois only)
    grid, yhat, err_bands = plotter.fit_regression(grid=plotter.x)
    e1 = err_bands[0, :] # err_bands[0, np.argsort(xvals)] <- sort by xpos to plot
    e2 = err_bands[1, :] #err_bands[1, np.argsort(xvals)]
    regr_cis = np.array([(ex, ey) for ex, ey in zip(e1, e2)])

    
    # Get mean and upper/lower CI bounds of bootstrapped distn for each cell
    boot_rois = [k for k, g in bootdata.groupby(['cell'])] 
    roi_ixs = [boot_rois.index(ri) for ri in reliable_rois]
 
    fov_pos = posdf['%s_fov' % axname][reliable_rois].values
    rf_pos = posdf['%s_rf' % axname][reliable_rois].values 
    fitv, regr = fit_linear_regr(fov_pos, rf_pos, return_regr=True, model=model)
    regr_info = {'regr': regr, 'fitv': fitv, 'xv': fov_pos, 'yv': rf_pos}

    #ax.plot(fov_pos, fitv, 'r:')
    eq_str = 'y=%.2fx + %.2f' % (regr.coef_[0], regr.intercept_[0])
    ax.set_title(eq_str, loc='left', fontsize=12)
     
#%
    boot_meds = np.array([g[parname].mean() for k, g \
                in bootdata[bootdata['cell'].isin(reliable_rois)].groupby(['cell'])])
    bootc = [(lo, up) for lo, up in \
                zip(bootcis['%s_lower' % parname][reliable_rois].values, 
                    bootcis['%s_upper' % parname][reliable_rois].values)]
   
    # Get YERR for plotting, (2, N), where 1st row=lower errors, 2nd row=upper errors
    boot_errs = np.array(zip(boot_meds-bootcis['%s_lower' % parname]\
                                    .loc[reliable_rois].values, 
                             bootcis['%s_upper' % parname]\
                                    .loc[reliable_rois].values-boot_meds)).T
    if plot_all_cis:
        # Plot bootstrap results for all RELIABLE cells 
        ax.scatter(fov_pos, boot_meds, c='k', marker='_', alpha=1.0, 
                   label='bootstrapped (%i%% CI)' % int(ci*100) )
        ax.errorbar(fov_pos, boot_meds, yerr=boot_errs, 
                    fmt='none', color='k', alpha=0.7, lw=1)
    sns.despine(offset=4, trim=True, ax=ax)

    # Check that values make sense and mark deviants
    vals = [(ri, roi, posdf['%s_fov' % axname][roi], posdf['%s_rf' % axname][roi]) \
            for ri, (roi, (bootL, bootU), (regL, regU), measured)
                in enumerate(zip(reliable_rois, bootc, regr_cis, rf_pos)) \
                if (bootL<=measured<=bootU) and ( (regL>bootU) or (regU<bootL) )]
     
    deviants = [v[1] for v in vals]
    xv = np.array([v[2] for v in vals])
    yv = np.array([v[3] for v in vals])
    dev_ixs = np.array([v[0] for v in vals])
    # Color/mark reliable fits that are also deviants
    if len(dev_ixs) > 0:
        yerrs = boot_errs[:, dev_ixs]
        ax.scatter(xv, yv, label='scattered', marker=marker,
                   s=marker_size, facecolors=deviant_color if fill_marker else 'none', 
                   edgecolors=deviant_color, alpha=1.0)
        if plot_boot_med:
            ax.scatter(xv, boot_meds[dev_ixs], c=deviant_color, marker='_', alpha=1.0) 
        ax.errorbar(xv, boot_meds[dev_ixs], yerr=yerrs, 
                        fmt='none', color=deviant_color, alpha=0.7, lw=1)
    ax.legend()

    bad_fits = [roi for rix, (roi, (lo, up), med) \
                in enumerate(zip(reliable_rois, bootc, rf_pos)) if not (lo<=med<= up) ]

    print("[%s] N deviants: %i (of %i reliable fits) | %i bad fits" \
            % (cond, len(deviants), len(reliable_rois), len(bad_fits)))
 
    return fig, regr_info, regr_cis, reliable_rois, deviants, bad_fits

#%%


#%%
def plot_regr_and_cis(eval_results, posdf, cond='azimuth', ci=.95, xaxis_lim=1200, ax=None):
    
    bootdata = eval_results['data']
    bootcis = eval_results['cis']
    roi_list = [k for k, g in bootdata.groupby(['cell'])]    
    
    if ax is None:
        fig, ax = pl.subplots(figsize=(20,10))
    
    ax.set_title(cond)
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    ax.set_xlim([0, xaxis_lim])
    
    axname = 'xpos' if cond=='azimuth' else 'ypos'
    parname = 'x0' if cond=='azimuth' else 'y0'
    
    g = sns.regplot('%s_fov' % axname, '%s_rf' % axname, \
                data=posdf.loc[roi_list], ci=ci*100, color='k', marker='o',
                scatter_kws=dict(s=50, alpha=0.5), ax=ax, 
                label='measured (regr: %i%% CI)' % int(ci*100) )

    xvals = posdf['%s_fov' % axname][roi_list].values
    yvals = posdf['%s_rf' % axname][roi_list].values
    
    # Get rois sorted by position:
    x0_meds = np.array([g[parname].mean() for k, g in bootdata.groupby(['cell'])])
    x0_lower = bootcis['%s_lower' % parname][roi_list]
    x0_upper = bootcis['%s_upper' % parname][roi_list]

    ci_intervals = bootcis['x0_upper'] - bootcis['x0_lower']
    weird = [i for i in ci_intervals.index.tolist() if ci_intervals[i] > 10]
    #weird = [i for ii, i in enumerate(bootcis.index.tolist()) if ((bootcis['%s_upper' % parname][i]) - (bootcis['%s_lower' % parname][i])) > 40]
    rlist = [i for i in roi_list if i not in weird]
    roi_ixs = np.array([roi_list.index(i) for i in rlist])
    roi_list = np.array([i for i in roi_list if i not in weird])
   
    if len(roi_ixs)==0:
        return ax

    # Plot bootstrap results
    xvals = posdf['%s_fov' % axname][roi_list].values
    yvals = posdf['%s_rf' % axname][roi_list].values
    
    ax.scatter(xvals, x0_meds[roi_ixs], c='k', marker='_', \
                            label='bootstrapped (%i%% CI)' % int(ci*100) )
    ax.errorbar(xvals, x0_meds[roi_ixs], \
                yerr=np.array(zip(x0_meds[roi_ixs]-x0_lower.iloc[roi_ixs], 
                                  x0_upper.iloc[roi_ixs]-x0_meds[roi_ixs])).T, 
                fmt='none', color='k', alpha=0.5) 
#
#    ax.scatter(xvals, x0_meds, c='k', marker='_', label='bootstrapped (%i%% CI)' % int(ci*100) )
#    ax.errorbar(xvals, x0_meds, yerr=np.array(zip(x0_meds-x0_lower, x0_upper-x0_meds)).T, 
#            fmt='none', color='k', alpha=0.5)
    ax.set_xticks(np.arange(0, xaxis_lim, 100))
    #sns.despine(offset=1, trim=True, ax=ax)
    
    ax.legend()
            
    return ax


def fit_linear_regr(xvals, yvals, return_regr=False, model='ridge'):
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
    if any(np.isnan(xvals)) or any(np.isnan(yvals)):
        print("NAN")
        #print(np.where(np.isnan(xvals)))
        #print(np.where(np.isnan(yvals)))
    regr.fit(xvals, yvals)
    fitv = regr.predict(xvals)
    if return_regr:
        return fitv.reshape(-1), regr
    else:
        return fitv.reshape(-1)


def plot_linear_regr(xv, yv, ax=None,  model='ridge', 
                     marker='o', marker_size=30, alpha=1.0, marker_color='k',
                     linestyle='_', linecolor='r'):
    try:
        fitv, regr = fit_linear_regr(xv, yv, return_regr=True, model=model)
    except Exception as e:
        traceback.print_exc()
        print("... no lin fit")
        return None

    if ax is none:
        fig, ax = pl.subplots()
        
    rmse = np.sqrt(skmetrics.mean_squared_error(yv, fitv))
    r2 = float(skmetrics.r2_score(yv, fitv))
    print("[%s] Mean squared error: %.2f" % (cond, rmse))
    print('[%s] Variance score: %.2f' % (cond, r2))
    
    ax.scatter(xv, yv, c=marker_color, marker=marker, s=marker_size, alpha=alpha)
    ax.plot(xv, fitv, linestyle, color=linecolor, label=model)
    ax.set_xlim([0, 1200])
    #ax.set_ylim()    
    eq_str = 'y=%.2fx + %.2f' % (regr.coef_[0], regr.intercept_[0])
    ax.set_title(eq_str, loc='left', fontsize=12)
 
    r, p = spstats.pearsonr(xv, yv) #.abs())
    corr_str = 'pearson=%.2f (p=%.2f)' % (r, p)
    ax.plot(ax.get_xlim()[0], ax.get_ylim()[0], alpha=0, label=corr_str)
    ax.legend(loc='upper right', fontsize=8)

    return regr


def plot_linear_regr_by_condition(posdf, model='ridge'):
    
    fig, axes = pl.subplots(2, 3, figsize=(10, 6))
    for ri, cond in enumerate(['azimuth', 'elevation']):
        # Do fit
        axname = 'xpos' if cond=='azimuth' else 'ypos' 
        yv = posdf['%s_rf' % axname].values
        xv = posdf['%s_fov' % axname].values    
        try:
            fitv, regr = fit_linear_regr(xv, yv, return_regr=True, model=model)
        except Exception as e:
            traceback.print_exc()
            print("Error fitting cond %s" % cond)
            continue
        # Evaluate fit
        rmse = np.sqrt(skmetrics.mean_squared_error(yv, fitv))
        r2 = float(skmetrics.r2_score(yv, fitv))
        print("[%s] Mean squared error: %.2f | Variance score: %.2f" % (cond, rmse, r2))

        # Plot 
        ax=axes[ri, 0]
        ax.set_title(cond, fontsize=12, loc='left')
        ax.scatter(xv, yv, c='k', alpha=0.5)
        ax.set_ylabel('RF position (rel. deg.)')
        ax.set_xlabel('FOV position (um)')
        #ax.set_xlim([0, ylim])
        #sns.despine(offset=1, trim=True, ax=ax)
        ax.plot(xv, fitv, 'r')
        ax.set_xlim([0, 1200])
        #ax.set_ylim()    
        r, p = spstats.pearsonr(posdf['%s_fov' % axname], posdf['%s_rf' % axname]) 
        corr_str = 'pearson=%.2f (p=%.2f)' % (r, p)
        ax.plot(ax.get_xlim()[0], ax.get_ylim()[0], alpha=0, label=corr_str)
        ax.legend(loc='upper right', fontsize=8)
    
        ax = axes[ri, 1]
        residuals = fitv - yv
        ax.hist(residuals, histtype='step', color='k')
        #sns.despine(offset=1, trim=True, ax=ax)
        ax.set_xlabel('residuals')
        ax.set_ylabel('counts')
        maxval = max (abs(residuals))
        ax.set_xlim([-maxval, maxval])
         
        ax = axes[ri, 2]
        r2_vals = posdf['r2']
        ax.scatter(r2_vals, abs(residuals), c='k', alpha=0.5)
        ax.set_xlabel('r2')
        ax.set_ylabel('abs(residuals)')
       
        if model=='ridge':
            regr = Ridge()
        elif model=='Lasso':
            regr = Lasso()
        else:
            model = 'ols'
            regr = LinearRegression()
        # Add some metrics
        regr.fit(r2_vals.reshape(-1, 1), residuals.reshape(-1, 1)) #, yv)
        r2_dist_corr = regr.predict(r2_vals.reshape(-1, 1))
        ax.plot(r2_vals, r2_dist_corr, 'r', label=model)
        #sns.despine(offset=1, trim=True, ax=ax)
        r, p = spstats.pearsonr(r2_vals.values, np.abs(residuals))
        corr_str2 = 'pearson=%.2f (p=%.2f)' % (r, p)
        ax.plot(ax.get_xlim()[0], ax.get_xlim()[-1], alpha=0, label=corr_str2)
        ax.legend(loc='upper right', fontsize=8)
    
    pl.subplots_adjust(hspace=0.5, wspace=0.5)    
    return fig

#
#%%

def compare_regr_to_boot_params(eval_results, posdf, xlim=None, ylim=None, 
                                pass_criterion='all', model='ridge', 
                                deviant_color='dodgerblue', marker='o',
                                marker_size=20, fill_marker=True,
                                outdir='/tmp', data_id='DATAID',
                                plot_all_cis=False):

    '''
    deviants:  
        Cells w/ good RF fits (boostrapped, measured lies within some CI), but
               even CI lies outside of estimated regression CI
    bad_fits:  
        Cells w/ measured RF locations that do not fall within 
                the CI from bootstrapping
    
    To get all "pass" rois, include all returned ROIs with fits that are NOT in bad_fits.
    '''
    bootdata = eval_results['data']
    bootcis = eval_results['cis']
    fit_rois = [int(k) for k, g in bootdata.groupby(['cell'])]    
    pass_rois = eval_results['pass_cis'].index.tolist()
    pass_cis = eval_results['pass_cis'].copy()
    reliable_rois = get_reliable_fits(eval_results['pass_cis'], 
                                        pass_criterion=pass_criterion)
    
    #% # Plot bootstrapped param CIs + regression CI
    xaxis_lim = max([xlim, ylim])
    reg_results = {}

    for cond in ['azimuth', 'elevation']:
        fig, regr, regci, reliable_c, deviants, bad_fits = do_regr_on_fov_cis(
                                                        bootdata, bootcis,
                                                        posdf, cond=cond,
                                                        model=model, 
                                                        roi_list=[], #reliable_rois,
                                                        deviant_color=deviant_color,
                                                        fill_marker=fill_marker,
                                                        marker=marker, 
                                                        marker_size=marker_size,
                                                        xaxis_lim=xlim) #xaxis_lim)
        # Get some stats from linear regr
        rmse = np.sqrt(skmetrics.mean_squared_error(regr['yv'], regr['fitv']))
        r2 = skmetrics.r2_score(regr['yv'], regr['fitv'])
        pearson_r, pearson_p = spstats.pearsonr(regr['xv'], regr['yv'])

        pass_rois = [i for i in fit_rois if i not in bad_fits]
        reg_results[cond] = {'cis': [tuple(ci) for ci in regci], 
                            'deviants': deviants, 
                            'bad_fits': bad_fits, 
                            'pass_rois': pass_rois,
                            'reliable_rois': reliable_rois,
                            'regr_coef': float(regr['regr'].coef_[0]), #r_coef,
                            'regr_int': float(regr['regr'].intercept_[0]),
                            'regr_R2': r2, 'regr_RMSE': rmse, 
                            'regr_pearson_p': pearson_p, 'regr_pearson_r': pearson_r}
 
        pplot.label_figure(fig, data_id)
        pl.savefig(os.path.join(outdir, 'VF2RF_regr_deviants_%s%s.svg' \
                                                    % (cond, filter_str)))
        pl.close()

    reg_results['reliable_rois'] = reliable_rois
    reg_results['pass_criterion'] = pass_criterion
   
    with open(os.path.join(outdir, 'regr_results_deviants_bycond.json'), 'w') as f:
        json.dump(reg_results, f, indent=4)    
    print("--- saved roi info after evaluation.")
  
    return reg_results
   

#%%
      
#%%

def plot_eval_summary(meas_df, fit_results, eval_results, plot_rois=[],
                        sigma_scale=2.35, scale_sigma=True, 
                        outdir='/tmp/rf_fit_evaluation', plot_format='svg',
                        data_id='DATA ID'):
    '''
    For all fit ROIs, plot summary of results (fit + evaluation).
    Expect that meas_df has R2>fit_thr, since those are the ones that get bootstrap evaluation 
    '''
    bootdata = eval_results['data']
    roi_list = meas_df.index.tolist() #sorted(bootdata['cell'].unique())
    
    for ri, rid in enumerate(sorted(roi_list)):
        if ri % 20 == 0:
            print("... plotting eval summary (%i of %i)" % (int(ri+1), len(roi_list))) 
        _fitr = fit_results[rid]
        _bootdata = bootdata[bootdata['cell']==rid]
        fig = plot_roi_evaluation(rid, meas_df, _fitr, _bootdata, 
                                        scale_sigma=scale_sigma, sigma_scale=sigma_scale)
        if rid in plot_rois:
            fig.suptitle('rid %i**' % rid)
        pplot.label_figure(fig, data_id)
        pl.savefig(os.path.join(outdir, 'roi%05d.%s' % (int(rid+1), plot_format)))
        pl.close()
    return


#%%
def load_params(params_fpath):
    with open(params_fpath, 'r') as f:
        options_dict = json.load(f)
    return options_dict

def save_params(params_fpath, opts):
    if isinstance(opts, dict):
        options_dict = opts.copy()
    else:
        options_dict = vars(opts)
    with open(params_fpath, 'w') as f:
        json.dump(options_dict, f, indent=4, sort_keys=True) 
    return
  
#%%
def load_eval_results(animalid, session, fov, experiment='rfs',
                        traceid='traces001', response_type='dff', 
                        fit_desc=None, do_spherical_correction=False,
                        rootdir='/n/coxfs01/2p-data'):

    eval_results=None; eval_params=None;            
    if fit_desc is None:
        fit_desc = fitrf.get_fit_desc(response_type=response_type,
                                        do_spherical_correction=do_spherical_correction)
        #fit_desc = 'fit-2dgaus_%s-no-cutoff' % response_type

    rfname = experiment.split('_')[1] if 'combined' in experiment else experiment
    try: 
        #print("Checking to load: %s" % fit_desc)
        rfdir = glob.glob(os.path.join(rootdir, animalid, session, 
                        fov, '*%s_*' % rfname,
                        'traces', '%s*' % traceid, 'receptive_fields', 
                        '%s*' % fit_desc))[0]
        evaldir = os.path.join(rfdir, 'evaluation')
        assert os.path.exists(evaldir), "No evaluation exists\n(%s)\n. Aborting" % evaldir
    except IndexError as e:
        traceback.print_exc()
        return None, None
    except AssertionError as e:
        traceback.print_exc()
        return None, None

    # Load results
    rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    assert os.path.exists(rf_eval_fpath), "No eval result: %s" % rf_eval_fpath
    with open(rf_eval_fpath, 'rb') as f:
        eval_results = pkl.load(f)
   
    #  Load params 
    eval_params_fpath = os.path.join(evaldir, 'evaluation_params.json')
    with open(eval_params_fpath, 'r') as f:
        eval_params = json.load(f)
        
    return eval_results, eval_params

#%%
def load_matching_fit_results(animalid, session, fov, traceid='traces001',
                              experiment='rfs', response_type='dff',
                              nframes_post=0, do_spherical_correction=False,
                              sigma_scale=2.35, scale_sigma=True):
    fit_results=None
    fit_params=None
    try:
        fit_results, fit_params = fitrf.load_fit_results(animalid, session,
                                        fov, traceid=traceid,
                                        experiment=experiment,
                                        response_type=response_type,
                                        do_spherical_correction=do_spherical_correction)
        assert fit_params['nframes_post_onset'] == nframes_post, \
            "Incorrect nframes_post (found %i, requested %i)" \
            % (fit_params['nframes_post_onset'], nframes_post)
        assert fit_params['response_type'] == response_type, \
            "Incorrect response type (found %i, requested %i)" \
            %(fit_params['repsonse_type'], response_type)
        if sigma_scale != fit_params['sigma_scale'] \
            or scale_sigma != fit_params['scale_sigma']:
                print("... updating scale_sigma: %s" % str(fit_params['sigma_scale']))
                scale_sigma=fit_params['scale_sigma']               
                print("... updating sigma_scale to %.2f (from %.2f)" \
                    % (fit_params['sigma_scale'], sigma_scale))
            sigma_scale=fit_params['sigma_scale']
            do_fits=True
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("[err]: unable to load fit results, re-fitting.")
    
    return fit_results, fit_params

#%%
def do_rf_fits_and_evaluation(animalid, session, fov, rfname=None, 
                              traceid='traces001', 
                              response_type='dff', n_processes=1,
                              fit_thr=0.5, 
                              n_resamples=10, n_bootstrap_iters=1000, 
                              post_stimulus_sec=0., 
                              sigma_scale=2.35, scale_sigma=True,
                              ci=0.95, pass_criterion='all', model='ridge', 
                              plot_boot_distns=True, plot_pretty_rfs=False, 
                              deviant_color='dodgerblue', 
                              plot_all_cis=False,
                              do_fits=False, do_evaluation=False, 
                              reload_data=False, create_stats=False,
                              do_spherical_correction=False,
                              rootdir='/n/coxfs01/2p-data', opts=None):

    print("deviant col: %s" % deviant_color)
   
    # Check if should do fitting 
    nframes_post = int(round(post_stimulus_sec*44.65))
    if not do_fits:
        try:
            fit_results, fit_params = load_matching_fit_results(animalid, session, fov,
                                        experiment=rfname, traceid=traceid, 
                                        response_type=response_type, 
                                        nframes_post=nframes_post,
                                        sigma_scale=sigma_scale, scale_sigma=scale_sigma,
                                        do_spherical_correction=do_spherical_correction)
            assert fit_results is not None 
        except Exception as e:
            traceback.print_exc()
            print("[err]: unable to load fit results, re-fitting.")
            do_fits = True
           
    if 'rfs10' in rfname:
        assert fit_params['column_spacing']==10, \
        "WRONG SPACING (%s), is %.2f." % (rfname, fit_params['column_spacing'])
        
    # Set directories
    rfdir = fit_params['rfdir'] 
    fit_desc = fit_params['fit_desc'] 
    data_id = '|'.join([datakey, traceid, fit_desc])

    evaldir = os.path.join(rfdir, 'evaluation')
    roidir = os.path.join(evaldir, 
                'rois_%i-iters_%i-resample' % (n_bootstrap_iters, n_resamples))
    if not os.path.exists(roidir):
        os.makedirs(roidir) 
    if os.path.exists(os.path.join(evaldir, 'rois')):
        shutil.rmtree(os.path.join(evaldir, 'rois'))

    #%% Do bootstrap analysis    
    print("-evaluating (%s)-" % str(do_evaluation))
    if do_evaluation is False: #nd create_new is False:
        try:
            print("... loading eval results")
            eval_results, eval_params = load_eval_results(animalid, session, fov, 
                                                         experiment=rfname,
                                                         fit_desc=fit_desc,
                                                         response_type=response_type) 
            assert 'data' in eval_results.keys(), \
                    "... old datafile, redoing boot analysis"
            assert 'pass_cis' in eval_results.keys(), \
                    "... no criteria passed, redoing"
            print("N eval:", len(eval_results['pass_cis'].index.tolist()))
        except Exception as e:
            traceback.print_exc()
            do_evaluation=True

    if do_evaluation: 
        # Update params to include evaluation info 
        evaluation = {'n_bootstrap_iters': n_bootstrap_iters, 
                      'n_resamples': n_resamples,
                      'ci': ci})   
        fit_params.update({'evaluation': evaluation})
        # Do evaluation 
        print("... doing rf evaluation")
        eval_results = evaluate_rfs(fit_params, 
                                    n_processes=n_processes) 
    if len(eval_results.keys())==0:# is None: # or 'data' not in eval_results:
        return {} #None

    ##------------------------------------------------
    # Load fit results
    fit_results, fit_params = rfutils.load_fit_results(animalid, session, fov,
                                        experiment=run_name, traceid=traceid,
                                        response_type=response_type, 
                                        do_spherical_correction=do_spherical_correction)
    fitdf = fitrf.rfits_to_df(fit_results, fit_params=fit_params, 
                            scale_sigma=fit_params['scale_sigma'], 
                            sigma_scale=fit_params['sigma_scale'])
    fitdf = fitdf[fitdf['r2']>fit_params['fit_thr']]
 
    if plot_boot_distns:
        print("... plotting boot distn") #.\n(to: %s" % outdir)
        plot_eval_summary(fitdf, fit_results, eval_results, 
                          reliable_rois=fit_rois, #reliable_rois,
                          sigma_scale=fit_params['sigma_scale'], 
                          scale_sigma=fit_params['scale_sigma'],
                          outdir=roidir, plot_format='svg', 
                          data_id=data_id)

    # Identify cells w fit params within CIs
    pass_cis = check_reliable_fits(fitdf, eval_results['cis']) 
    # Identify reliable fits (params specified by pass_criterion
    # fall within pass_cis
    reliable_rois = get_reliable_fits(eval_results['pass_cis'], 
                                      pass_criterion=pass_criterion)
    # Plot distribution of params w/ 95% CI
    print("%i out of %i cells w. R2>0.5 are reliable (95%% CI)" 
            % (len(reliable_rois), len(fit_rois)))
    # Update eval results
    eval_results.update({'reliable_rois': reliable_rois})
    eval_results.update({'pass_cis': pass_cis})
    save_eval_results(eval_results, fit_params)


    # Figure out if there are any deviants
    fovcoords = exp.get_roi_coordinates()
    posdf = pd.concat([fitdf[['x0', 'y0']].copy(), 
                       fovcoords['roi_positions'].copy()], axis=1) 
    posdf = posdf.rename(columns={'x0': 'xpos_rf', 'y0': 'ypos_rf',
                                  'ml_pos': 'xpos_fov', 'ap_pos': 'ypos_fov'})

    marker_size=30; fill_marker=True; marker='o';
    reg_results = regr_rf_fov(posdf, eval_results, fit_params, 
                                     data_id=data_id, 
                                     pass_criterion=pass_criterion, model=model,
                                     marker=marker, marker_size=marker_size, 
                                     fill_marker=fill_marker, 
                                     deviant_color=deviant_color)
    
    return eval_results, eval_params


def regr_rf_fov(posdf, fit_params, eval_results, 
                model='ridge', pass_criterion='all', data_id='ID', 
                deviant_color='magenta', marker='o', 
                marker_size=20, fill_marker=True):
    print("~regressing rf on fov~")

    evaldir = os.path.join(fit_params['rfdir'], 'evaluation')
    fig = plot_linear_regr_by_condition( posdf.loc[reliable_rois],model=model)
    pl.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)
    pplot.label_figure(fig, data_id)
    pl.savefig(os.path.join(evaldir, 'RFpos_v_CTXpos_split_axes.svg'))   
    pl.close()
   
    #%% Compare regression fit to bootstrapped params 
    reg_results = compare_regr_to_boot_params(eval_results, posdf, 
                                        outdir=evaldir, data_id=data_id, 
                                        pass_criterion=pass_criterion, model=model, 
                                        deviant_color=deviant_color, marker=marker,
                                        marker_size=marker_size, 
                                        fill_marker=fill_marker)

    #%% Identify "deviants" based on spatial coordinates
    print('%i reliable of %i fit (thr>.5) | regr R2=%.2f' \
                % (len(reg_results['reliable_rois']), 
                   len(meas_df), reg_results['azimuth']['regr_R2']))

    return reg_results #deviants


#%%
def do_evaluation(datakey, fit_results, fit_params, 
                n_bootstrap_iters=500, n_resamples=10, ci=0.95,
                pass_criterion='all', model='ridge', 
                plot_boot_distns=True, 
                deviant_color='dodgerblue', plot_all_cis=False,
                creat_new=False, rootdir='/n/coxfs01/2p-data'):
       
    # Set directories
    rfdir = fit_params['rfdir'] 
    fit_desc = fit_params['fit_desc'] 
    data_id = '|'.join([datakey, fit_desc])

    evaldir = os.path.join(rfdir, 'evaluation')
    roidir = os.path.join(evaldir, 
                'rois_%i-iters_%i-resample' % (n_bootstrap_iters, n_resamples))
    if not os.path.exists(roidir):
        os.makedirs(roidir) 
    if os.path.exists(os.path.join(evaldir, 'rois')):
        shutil.rmtree(os.path.join(evaldir, 'rois'))

    #%% Do bootstrap analysis    
    print("-evaluating (%s)-" % str(create_new))
    if not create_new:
        try:
            print("... loading eval results")
            eval_results, eval_params = load_eval_results(animalid, session, fov, 
                                                     experiment=rfname,
                                                     fit_desc=fit_desc,
                                                     response_type) 
            assert 'data' in eval_results.keys(), \
                    "... old datafile, redoing boot analysis"
            assert 'pass_cis' in eval_results.keys(), \
                    "... no criteria passed, redoing"
            print("N eval:", len(eval_results['pass_cis'].index.tolist()))
        except Exception as e:
            traceback.print_exc()
            do_evaluation=True

    if do_evaluation: 
        # Update params to include evaluation info 
        evaluation = {'n_bootstrap_iters': n_bootstrap_iters, 
                      'n_resamples': n_resamples,
                      'ci': ci})   
        fit_params.update({'evaluation': evaluation})
        # Do evaluation 
        print("... doing rf evaluation")
        eval_results = evaluate_rfs(fit_params, 
                                    n_processes=n_processes) 
    if len(eval_results.keys())==0:# is None: # or 'data' not in eval_results:
        return {} #None

    ##------------------------------------------------
    # Load fit results
    fit_results, fit_params = rfutils.load_fit_results(animalid, session, fov,
                                        experiment=run_name, traceid=traceid,
                                        response_type=response_type, 
                                        do_spherical_correction=do_spherical_correction)
    fitdf = fitrf.rfits_to_df(fit_results, fit_params=fit_params, 
                            scale_sigma=fit_params['scale_sigma'], 
                            sigma_scale=fit_params['sigma_scale'])
    fitdf = fitdf[fitdf['r2']>fit_params['fit_thr']]
 
    if plot_boot_distns:
        print("... plotting boot distn") #.\n(to: %s" % outdir)
        plot_eval_summary(fitdf, fit_results, eval_results, 
                          reliable_rois=fit_rois, #reliable_rois,
                          sigma_scale=fit_params['sigma_scale'], 
                          scale_sigma=fit_params['scale_sigma'],
                          outdir=roidir, plot_format='svg', 
                          data_id=data_id)

    # Identify cells w fit params within CIs
    pass_cis = check_reliable_fits(fitdf, eval_results['cis']) 
    # Identify reliable fits (params specified by pass_criterion
    # fall within pass_cis
    reliable_rois = get_reliable_fits(eval_results['pass_cis'], 
                                      pass_criterion=pass_criterion)
    # Plot distribution of params w/ 95% CI
    print("%i out of %i cells w. R2>0.5 are reliable (95%% CI)" 
            % (len(reliable_rois), len(fit_rois)))
    # Update eval results
    eval_results.update({'reliable_rois': reliable_rois})
    eval_results.update({'pass_cis': pass_cis})
    save_eval_results(eval_results, fit_params)


    # Figure out if there are any deviants
    fovcoords = exp.get_roi_coordinates()
    posdf = pd.concat([fitdf[['x0', 'y0']].copy(), 
                       fovcoords['roi_positions'].copy()], axis=1) 
    posdf = posdf.rename(columns={'x0': 'xpos_rf', 'y0': 'ypos_rf',
                                  'ml_pos': 'xpos_fov', 'ap_pos': 'ypos_fov'})

    marker_size=30; fill_marker=True; marker='o';
    reg_results = regr_rf_fov(posdf, eval_results, fit_params, 
                                     data_id=data_id, 
                                     pass_criterion=pass_criterion, model=model,
                                     marker=marker, marker_size=marker_size, 
                                     fill_marker=fill_marker, 
                                     deviant_color=deviant_color)
    
    return eval_results, eval_params


   
#%%
def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='FOV1_zoom2p0x', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV2_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--rfname', action='store', dest='rfname', default=None, \
                      help="name of rfs to process (default uses rfs10, if exists, else rfs)")
    parser.add_option('--fit', action='store_true', dest='do_fits', default=False, \
                      help="flag to do RF fitting anew")
    parser.add_option('--eval', action='store_true', dest='do_evaluation', default=False, \
                      help="flag to do RF evaluation anew")
    parser.add_option('--load', action='store_true', dest='reload_data', default=False, \
                      help="flag to reload data arrays and save (e.g., dff.pkl)")



    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--data-type', action='store', dest='trace_type', default='corrected', \
                      help="Trace type to use for analysis [default: corrected]")

    parser.add_option('-M', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")    
    parser.add_option('-f', '--fit-thr', action='store', dest='fit_thr', default=0.5, \
                      help="Threshold for RF fits (default: 0.5)")

    parser.add_option('-b', '--n-boot', action='store', dest='n_bootstrap_iters', default=1000, \
                      help="N bootstrap iterations for evaluating RF param fits (default: 1000)")
    parser.add_option('-s', '--n-resamples', action='store', dest='n_resamples', default=10, \
                      help="N trials to sample with replacement (default: 10)")
    parser.add_option('-n', '--n-processes', action='store', dest='n_processes', default=1, \
                      help="N processes (default: 1)")
    
    parser.add_option('-C', '--ci', action='store', dest='ci', default=0.95, \
                      help="CI percentile(default: 0.95)")

    parser.add_option('--no-boot-plot', action='store_false', dest='plot_boot_distns', default=True, \
                      help="flag to not plot bootstrapped distNs of x0, y0 for each roi")
    parser.add_option('--pretty', action='store_true', dest='plot_pretty_rfs', default=False, \
                      help="flag to make pretty plots of RF maps")


#    parser.add_option('--pixels', action='store_false', dest='transform_fov', default=True, \
#                      help="flag to not convert fov space into microns (keep as pixels)")
#
    parser.add_option('--all-cis', action='store_true', dest='plot_all_cis', default=False, \
                      help="flag to plot CIs for all cells (not just deviants)")
    parser.add_option('-c', '--color', action='store', dest='deviant_color', default='dodgerblue', \
            help="color to plot deviants to stand out (default: dodgerblue)")

    parser.add_option('--pass', action='store', dest='pass_criterion', default='all', \
                      help="criterion for ROI passing fit(default: 'all' - all params pass 95% CI)")
 

    parser.add_option('--sigma', action='store', dest='sigma_scale', default=2.35, \
                      help="sigma scale factor for FWHM (default: 2.35)")
    parser.add_option('--no-scale', action='store_false', dest='scale_sigma', default=True, \
                      help="set to scale sigma to be true sigma, rather than FWHM")
    parser.add_option('-p', '--post', action='store', dest='post_stimulus_sec', default=0.0, 
                      help="N sec to include in stimulus-response calculation for maps (default:0.0)")

    parser.add_option('--test', action='store_true', dest='test_run', default=False, 
                      help="Flag to just wait 2 sec, for test")
    parser.add_option('--sphere', action='store_true', dest='do_spherical_correction', 
                    default=False, help="Flag to do spherical correction")



    (options, args) = parser.parse_args(options)

    return options

#%%
rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084' #JC076'
session = '20190522' #'20190501'
fov = 'FOV1_zoom2p0x'
create_new = False

# Data
traceid = 'traces001'
trace_type = 'corrected'
response_type = 'dff'
fit_thr = 0.5
#transform_fov = True

# Bootstrap params
n_bootstrap_iters=1000
n_resamples = 10
plot_boot_distns = True
ci = 0.95
n_processes=1  

sigma_scale = 2.35
scale_sigma = True
post_stimulus_sec=0.5

do_fits=False
do_evaluation=False
reload_data=False

options = ['-i', animalid, '-S', session, '-A', fov, '-t', traceid,
           '-R', 'rfs', '-M', response_type, '-p', 0.5 ]


def main(options):
    opts = extract_options(options)
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    traceid = opts.traceid
    response_type = opts.response_type
    rootdir = opts.rootdir

    do_fits = opts.do_fits
    do_evaluation = opts.do_evaluation
    reload_data = opts.reload_data

    n_resamples = opts.n_resamples
    n_bootstrap_iters = opts.n_bootstrap_iters
 
    n_processes = int(opts.n_processes)
    pass_criterion = opts.pass_criterion
    
    ci = opts.ci
    plot_boot_distns = opts.plot_boot_distns

    rfname = opts.rfname
    plot_all_cis = opts.plot_all_cis
    deviant_color = opts.deviant_color
    plot_pretty_rfs =  opts.plot_pretty_rfs

    sigma_scale = float(opts.sigma_scale)
    scale_sigma = opts.scale_sigma
    post_stimulus_sec = float(opts.post_stimulus_sec)
    fit_thr = float(opts.fit_thr)
    
    do_spherical_correction = opts.do_spherical_correction 
     
    print("STATS?", any([do_fits, do_evaluation, reload_data]))
    if opts.test_run:
        print(">>> testing <<<")
        assert opts.test_run is False, "FAKE ERROR, test."

    else: 
        eval_results, eval_params = do_rf_fits_and_evaluation(animalid, session, fov, 
                              rfname=rfname, traceid=traceid,
                              response_type=response_type, fit_thr=fit_thr,
                              n_bootstrap_iters=n_bootstrap_iters, 
                              n_resamples=n_resamples, ci=ci,
                              #transform_fov=transform_fov, 
                              plot_boot_distns=plot_boot_distns, 
                              plot_pretty_rfs=plot_pretty_rfs, 
                              post_stimulus_sec=post_stimulus_sec,  
                              n_processes=n_processes, 
                              plot_all_cis=plot_all_cis,
                              deviant_color=deviant_color, 
                              scale_sigma=scale_sigma, sigma_scale=sigma_scale,
                              pass_criterion=pass_criterion,
                              do_fits=do_fits, 
                              do_evaluation=do_evaluation, 
                              reload_data=reload_data,
                              create_stats=any([do_fits, do_evaluation, reload_data]),
                              rootdir=rootdir, opts=opts,
                              do_spherical_correction=do_spherical_correction)
        
    print("***DONE!***")

if __name__ == '__main__':
    from pipeline.python.classifications import experiment_classes as util

    main(sys.argv[1:])
           
    #%%
    
    #options = ['-i', 'JC084', '-S', '20190525', '-A', 'FOV1_zoom2p0x', '-R', 'rfs']
    


# %%
