#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:16:05 2021

@author: julianarhee
"""
import os
import glob
import copy
import itertools

import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns

import analyze2p.aggregate_datasets as aggr
import analyze2p.utils as hutils
import analyze2p.plotting as pplot

import scipy.stats as spstats
import analyze2p.gratings.bootstrap_osi as osi
import analyze2p.objects.selectivity as sel
import analyze2p.receptive_fields.utils as rfutils

from scipy import signal
from scipy.optimize import curve_fit
from functools import reduce

# fitting functions
def func_halflife(t, a, tau, c):
    return a * 2**(-t / tau) + c

def func_tau(t, a, tau, c):
    return a * np.exp(-t / tau) + c

def func_decay(t, a, tau, c):
    return a * np.exp(-t * tau) + c


def fit_decay(xdata, ydata, p0=None, func='halflife', 
                return_inputs=False, normalize_x=True, ymax=None):
    in_x = xdata.copy()
    in_y = ydata.copy()
    if normalize_x:
        in_x_norm = (in_x - in_x[0])/(in_x[-1] - in_x[0])   # normalized
    else:
        in_x_norm = in_x.copy()

    if func=='halflife':
        pfunc_ = func_halflife
    else:
        pfunc_ = func_tau

    a_f, tau_f, c_f, r2 = None, None, None, None
    # init params
    if p0 is None:
        p0 = init_decay_params(ydata) # (1, 1, 1)
    #tau_ = 1 #2000 / 2. # divide by 2 since corr coef range [-1, 1]
    #a_lim = 1  
    tau_lim = 1 if normalize_x else 1500 
    ylim = (-1, 1) if ymax is None else (0, ymax)
    bounds = ((ylim[0], 0., -np.inf), (ylim[1], tau_lim, np.inf))
    a_0, tau_0, c_0 = p0
    try:
        popt4, pcov4 = curve_fit(pfunc_, in_x_norm, in_y, 
                                p0=(a_0, tau_0, c_0), bounds=bounds)
        if normalize_x:
            a4, tau4, c4 = popt4
            ##a_f = a4*np.exp(xdata[0]/(xdata[-1] - xdata[0]) / tau4)
            if func=='halflife':
                a_f = a4*2**(xdata[0]/(xdata[-1]-xdata[0]) / tau4)
            else:
                a_f = a4*np.exp(xdata[0]/(xdata[-1] - xdata[0]) / tau4) 
            tau_f = (xdata[-1] - xdata[0]) * tau4
            c_f = c4
            #print(a_f, tau_f, c_f)
        else:
            a_f, tau_f, c_f = popt4
        fitv = pfunc_(in_x, a_f, tau_f, c_f)

        # Get residual sum of squares 
        residuals = in_y - fitv
        ss_res = np.nansum(residuals**2)
        ss_tot = np.nansum((in_y - np.nanmean(in_y))**2)
        r2 = 1 - (ss_res / ss_tot)
    except RuntimeError as e:
        print('    no fit')
    except ValueError as e:
        print('    val out of bounds')

    if return_inputs:
        return a_f, tau_f, c_f, r2, in_x, in_y
    else:
        return af, tau_f, c_f, r2

def init_decay_params(ydata):
#     c_0 = ydata[-1]
#     tau_0 = 1
#     a_0 = (ydata[-1] - ydata[0])
    c_0 = np.nanmin(ydata) #.min() #ydata[-1]
    tau_0 = 1
    a_0 = abs(np.nanmax(ydata) - np.nanmin(ydata)) #ydata[0] #abs(ydata[0] - ydata[-1])
    p0 = (a_0, tau_0, c_0)
    return p0


def fit_decay_on_binned(cc_, use_binned=False, func='halflife', estimator='median',
                       metric='signal_cc', to_quartile='cortical_distance',
                       normalize_x=True, ymax=None,
                       return_inputs=False):
    '''
    Fit exponential decay (returns half-life).
    use_binned: (bool)
        If True, get median cell pair values for each bin, then fit. 
        Otherwise, sample within bin, but fit on raw points.
    '''

    #cc_ = cc0.dropna(axis=1)

    #x_var = 'binned_%s' % to_quartile
    x_var = '%s_label' % to_quartile

    if use_binned:
        if estimator=='median':
            data = cc_.groupby(x_var).median().reset_index()
        else:
            data = cc_.groupby(x_var).mean().reset_index() 
        #normalize_x = True
    else:
        data = cc_.copy()
        #normalize_x = False
    
    if estimator=='median':
        meanvs = cc_.groupby(x_var).median()\
                .reset_index().dropna()
    else:
        meanvs = cc_.groupby(x_var).mean()\
                .reset_index().dropna()
    incl_bins = list(set(meanvs[x_var].values))
  
    xdata = data[data[x_var].isin(incl_bins)]\
                .sort_values(by=to_quartile)[to_quartile].values
    ydata = data[data[x_var].isin(incl_bins)]\
                .sort_values(by=to_quartile)[metric].values
    # ----------------
    mean_y = meanvs.sort_values(by=to_quartile)[metric].values
    p0 = init_decay_params(mean_y)
    #ymax=180 if metric=='pref_dir_diff_abs' else None

    initv, tau, const, r2, xvals, yvals = fit_decay(xdata, ydata, func=func,
                                            normalize_x=normalize_x, ymax=ymax,
                                            return_inputs=True, p0=p0)
    res_ = pd.Series({'init': initv, 'tau': tau, 'constant': const, 'R2': r2})
    
    if return_inputs:
        return res_, xvals, yvals
    else:
        return res_
   

def sample_bins_and_fit(vg, nsamples_per, cnt_groups, resample=True,
                        to_quartile='cortical_distance', metric='signal_cc', 
                        use_binned=False, normalize_x=True, fit_sites=True, ymax=None,
                        func='halflife', estimator='median', randi=None):
    '''
    Resample within bin and site, take <median> across pairs per site.
    Fit decay func to sites (N points per bin is the number of sites in that bin).

    Returns:
    -------
    res_: pd.DataFrame()
        Results of decay fit: init, tau, constant, R2
    
    xvals, yvals: np.ndarray()
        Bin values and fit data values for plotting fit.
    
    '''
    if resample:
        cc_sample = pd.concat([cgrp.sample(nsamples_per[dbin], 
                                 random_state=randi, replace=True) \
                                 for dbin, cgrp in vg.groupby(cnt_groups) \
                                 if dbin in nsamples_per.keys()])
    else:
        cc_sample = pd.concat([cgrp for dbin, cgrp in vg.groupby(cnt_groups) \
                                 if dbin in nsamples_per.keys()])
       
    # data to fit
    if fit_sites:
        fit_cc = cc_sample.groupby(cnt_groups).median().reset_index().dropna()
    else:
        fit_cc = cc_.copy()
    # fit
    res_, xvals, yvals = fit_decay_on_binned(fit_cc, use_binned=use_binned,
                                        normalize_x=normalize_x, ymax=ymax,
                                        func=func,
                                        estimator=estimator, metric=metric,
                                        to_quartile=to_quartile, return_inputs=True)
    return res_, xvals, yvals


def count_nsamples_per_bin(vg, cnt_groups, min_npairs=5):
    cnts = vg.groupby(cnt_groups)['neuron_pair'].count()  
    nsamples_per = dict((k, v) for k, v in zip(\
                                cnts[cnts>=min_npairs].index.tolist(),
                                cnts[cnts>=min_npairs].values)) 
    return nsamples_per 
 
def bootstrap_fitdecay(bcorrs, use_binned=False, func='halflife',
                      estimator='median', min_npairs=5, fit_sites=False,
                      metric='signal_cc', to_quartile='cortical_distance',
                      normalize_x=True, ymax=None, resample=True,
                      n_iterations=500):
    '''
    Cycle thru visual areas, sample w replacement, fit decay func.
    Return df of bootstrapped params.
    '''

    # cnt_grouper = ['binned_%s' % to_quartile] #, 'datakey']
    x_var = '%s_label' % to_quartile
    cnt_groups = [x_var, 'datakey'] if fit_sites else [x_var]

    col_selector = ['visual_area', 'cell_1', 'cell_2', 'neuron_pair', to_quartile, metric]
    col_selector.extend(cnt_groups)

    r_=[]
    for va, vg in bcorrs.groupby('visual_area'):
        cc0 = vg[col_selector].copy()
        # How many to resample per group/bin
        nsamples_per = count_nsamples_per_bin(vg, cnt_groups, min_npairs=min_npairs)
        if len(nsamples_per)==0:
            continue

        for n_iter in np.arange(0, n_iterations):
            # Sample
            res_, xvals, yvals = sample_bins_and_fit(cc0, nsamples_per, cnt_groups, 
                                            use_binned=use_binned, resample=resample,
                                            func=func, estimator=estimator,
                                            metric=metric, 
                                            to_quartile=to_quartile,randi=n_iter,
                                            normalize_x=normalize_x, ymax=ymax)
            res_['iteration'] = n_iter
            res_['visual_area'] = va
            r_.append(res_)
    resdf = pd.concat(r_, axis=1).T
    resdf['tau'] = resdf['tau'].astype(float)
    return resdf


def linregress_on_binned(cc_, metric='signal_cc', to_quartile='cortical_distance',
                        return_inputs=False):
    xdata = np.array([i \
             for i, (xi, xg) in enumerate(cc_.groupby('binned_%s' % to_quartile))])
    ydata = np.array([xg[metric].mean() \
             for xi, xg in cc_.groupby('binned_%s' % to_quartile)])
    # popt, pcov = curve_fit(func, xdata, ydata)
    res = spstats.linregress(xdata, ydata)

    res_ = pd.Series([res.slope, res.intercept, res.rvalue, res.pvalue, 
                  res.stderr, res.intercept_stderr],
                 index=['slope', 'intercept', 'rvalue', 'pvalue', 
                        'stderr', 'intercept_stderr'])
    if return_inputs:
        return res_, xdata, ydata
    else:
        return res_

def bootstrap_linregress(bcorrs, metric='signal_cc', to_quartile='cortical_distance',
                         n_iterations=500, n_samples=10):
    r_=[]
    for va, cc_ in bcorrs.groupby('visual_area'):
        for n_iter in np.arange(0, n_iterations):
            curr_cc = cc_.groupby('binned_%s' % to_quartile).sample(n=n_samples, 
                                                random_state=n_iter, replace=True)
            res = linregress_on_binned(curr_cc, metric=metric,
                                                to_quartile=to_quartile, 
                                                return_inputs=False)
            res['iteration'] = n_iter
            res['visual_area'] = va
            r_.append(res)
    resdf = pd.concat(r_, axis=1).T
    
    return resdf


# ------------------------------------------
# PAIRWISE CALCULATIONS
# ------------------------------------------
def trial_averaged_responses(zscored, sdf, params=['ori', 'sf', 'size', 'speed']):
    '''
    Average all trials for each condition. 
    Params should list stimulus configs (in sdf) 
    
    Returns:
    
    tuning_ : pd.DataFrame
        Trial-averaged repsonses. Columns are cells, rows are stimulus conds.
    '''
    # Get mean response per condition (columns=cells, rows=conditions)
    tuning_ = zscored.groupby(['config']).mean().reset_index()
    ctuples = tuple(sdf[params].values)
    multix = pd.MultiIndex.from_tuples(ctuples, names=params)
    tuning_.index = multix

    return tuning_


def get_roi_pos_and_rfs(neuraldf, curr_rfs=None, rfs_only=True, position_only=False,
                        merge_cols=['cell']):
    '''
    For a given dataset, return the position (and RF fit info) for each cell.
    Specify merge_cols to include visual_area and datakey if neuraldf is aggregate.
    
    Returns:

    roidf: (pd.DataFrame)
        Dataframe, each row is a cell, with cortical position (and RF fit position)

    Args:
    
    neuraldf:  (pd.DataFrame)
        Stacked trial metrics for either 1 datakey, or aggregate (*specific merge_cols)
    
    curr_rfs: (pd.DataFrame or None)
        Provide rfdf (RF fits) 

    rfs_only: (bool)
        If true, drops cells in neuraldf that do not have RFs. 

    position_only: (bool)
       If true, ignore all non-position parameters for RF fits (return x0, y0 only).
    
    merge_cols: (list)
        Merges on cell. If neuraldf is aggregate, *must* specific ['visual_area', 'datakey', 'cell']
     
    '''
    if 'datakey' in neuraldf.columns:
        found_ids = neuraldf['datakey'].unique()
        if len(found_ids)>1 and ('datakey' not in merge_cols):
            if 'experiment' in neuraldf.columns:
                merge_cols = ['visual_area', 'datakey', 'experiment', 'cell']
            else:
                merge_cols = ['visual_area', 'datakey', 'cell']
    # Don't need ALL Rf params, get the min. 
    rf_params = ['visual_area', 'datakey', 'cell', 'x0', 'y0']
    if not position_only:
        non_pos_params = ['fwhm_x', 'fwhm_y', 'theta', 'offset',
                           'amplitude', 'std_x', 'std_y', 'fwhm_avg', 'std_avg', 'area',
                           'fx', 'fy', 'ratio_xy','major_axis',
                           'minor_axis', 'anisotropy', 'aniso_index', 'eccentricity',
                           'eccentricity_ctr', 'rf_theta_deg', 'aspect_ratio']
        rf_params.extend(non_pos_params)
    # Add position info to neuraldata dataframe
    if 'ml_pos' not in neuraldf.columns:
        wpos = aggr.add_roi_positions(neuraldf.copy())
    else:
        wpos = neuraldf.copy()
    roi_pos = wpos[['visual_area', 'datakey', 'cell', 'ml_pos', 'ap_pos']].drop_duplicates().copy()
    # If provided, get RF fit info for cells that have fits
    if curr_rfs is not None:    
        cells_with_rfs = curr_rfs[rf_params].copy()
        roidf = pd.merge(roi_pos, cells_with_rfs, on=merge_cols, how='left')
        #has_rfs = np.intersect1d(roi_pos['cell'].unique(), curr_rfs['cell'].unique())
        #pos_ = roi_pos[roi_pos.cell.isin(has_rfs)].copy()
        #rfs_ = curr_rfs[curr_rfs.cell.isin(has_rfs)].copy()
        #roidf = pd.merge(pos_, rfs_)
    else:
        roidf = roi_pos.copy()

    if rfs_only:
        return roidf.dropna(axis=0)
    else:
        return roidf

def get_ccdist(neuraldf, roidf, return_zscored=False, curr_cfgs=None,
                xcoord='ml_pos', ycoord='ap_pos', label='cortical_distance',
                add_eccentricity=True):
    ''' 
    roidf should be 1 row per cell -- must contan xcoord, ycoord info
    Do get_roi_pos_and_rfs() for ROIDF.
    '''
    if curr_cfgs is None:
        curr_cfgs = neuraldf['config'].unique()
    #if curr_cells is None:
    #    curr_cells = neuraldf['cell'].unique()
    #curr_cells = np.intersect1d(neuraldf['cell'].unique(), roidf['cell'].unique())

    # Dont do zscore within-condition
    corrs, zscored = calculate_corrs(neuraldf, do_zscore=True, return_zscored=True,
                                     curr_cfgs=curr_cfgs)
   
    # Add distance 
    ccdist = get_pw_distance(corrs, roidf, xcoord=xcoord, ycoord=ycoord, label=label,
                        add_eccentricity=add_eccentricity)
    
    if return_zscored:
        return ccdist, zscored
    else:
        return ccdist


def calculate_corrs(ndf, do_zscore=True, return_zscored=False, 
                    curr_cells=None, curr_cfgs=None):
    if curr_cells is None: 
        curr_cells = ndf['cell'].unique()
    if curr_cfgs is None:
        curr_cfgs = ndf['config'].unique()
    ndf1 = ndf[(ndf.config.isin(curr_cfgs)) & (ndf['cell'].isin(curr_cells))].copy()
    # Reshape dataframe to ntrials x nrois
    trial_means0 = aggr.stacked_neuraldf_to_unstacked(ndf1)
    cfgs_by_trial = trial_means0['config']
    if do_zscore:
        # Zscore trials
        zscored = aggr.zscore_dataframe(trial_means0[curr_cells])
        zscored['config'] = cfgs_by_trial
    else:
        zscored = trial_means0.copy() #ndf.copy()
    # Get signal correlations
    signal_corrs = calculate_signal_corrs(zscored)
    # Get Noise correlations
    noise_corrs0 = calculate_noise_corrs(zscored)
    # Average over stimulus conditions 
    noise_corrs = noise_corrs0.groupby(['neuron_pair']).mean().reset_index()
    # Combine
    corrs = pd.merge(signal_corrs, noise_corrs)
    if return_zscored:
        return corrs, zscored
    else:
        return corrs

def calculate_signal_corrs(zscored, included_configs=None):
    ''' Calculate signal correlations.
    Signal correlations are computed as the Pearson correlation between
    trial-averaged stimulus responses for pairs of neurons.
    Get pairwise CC for condition vectors (each point is a cond, avg across trials).
    
    zscored (pd.DataFrame): 
        Unstacked, zscored NDATA for easy column-wise ops.
        Columns are cells + config, rows are trials
        
    Returns: cc (pd.DataFrame)
        columns: cell_1, cell_2, neuron_pair, and correlation coeff. 
    
    Note: Faster than itertools method.
          8.91 ms ± 285 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    '''
    if included_configs is None:
        included_configs = zscored['config'].unique()
        
    # Each entry is the mean response (across trials) for a given stim condition.
    # tuning_ : df, nconds x nrois
    zscored.index.name = None
    zscored.columns.name = None
    tuning_ = zscored[zscored['config'].isin(included_configs)]\
                                   .groupby(['config']).mean().reset_index()
    # columns are cells, rows are stimuli -- get pairwise corrs()
    cc = do_pairwise_cc_melt(tuning_, metric_name='signal_cc')
    
    return cc


def calculate_noise_corrs(zscored, method='pearson'):
    ''' Calculate noise correlations.
    Noise correlations are computed as the Pearson correlation of single-trial 
    responses of a given stimulus condition for a pair of neurons, then 
    averaged over stimuli.
    
    For each condition, get pairwise CC for trial vectors (each point is a trial).
    Should average across conditions to get 1 noise CC per neuron pair.
    
    zscored (pd.DataFrame): 
        Unstacked, zscored NDATA for easy column-wise ops.
        Columns are cells + config, rows are trials
        
    Returns: cc (pd.DataFrame)
        columns: cell_1, cell_2, neuron_pair, and correlation coeff. 
    
    Note: Faster than itertools method.
          8.91 ms ± 285 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    '''
    c_=[]
    for cfg, df_ in zscored.groupby(['config']):
        df_.index.name = None
        df_.columns.name = None
        # columns are cels, rows are trials (for 1 condition)
        cc_ = do_pairwise_cc_melt(df_, metric_name='noise_cc', 
                                  include_diagonal=False)
        cc_['config'] = cfg
        c_.append(cc_)
    cc = pd.concat(c_, axis=0)
    return cc

def do_pairwise_cc_melt(df_, metric_name='cc', include_diagonal=False):
    '''Do pairwise correltion betwen all pairs of columns.
    Get unique pairs' values from correlation matrix'''
    cc = melt_square_matrix(df_.corr(), metric_name=metric_name, 
                            include_diagonal=include_diagonal)
    cc = cc.rename(columns={'row': 'cell_1', 'col': 'cell_2'})

    # Check counts
    nr = len([r for r in df_.columns if hutils.isnumber(r)])
    #nr = len(df_['cell'].unique())
    ncombos = len(list(itertools.combinations(np.arange(0, nr), 2)))
    assert len(cc)==ncombos, "bad merging when creating pw combos (expected %i, have %i)" % (ncombos, len(cc))

    cc[['cell_1', 'cell_2']] = cc[['cell_1', 'cell_2']].astype(int)
    cc['neuron_pair'] = ['%i_%i' % (c1, c2) for \
                         c1, c2 in cc[['cell_1', 'cell_2']].values]
    return cc
    
def melt_square_matrix(df, metric_name='value', add_values={}, include_diagonal=False):
    '''Melt square matrix into unique values only'''
    k = 0 if include_diagonal else 1
    df = df.where(np.triu(np.ones(df.shape), k=k).astype(np.bool))

    df0 = df.stack(dropna=True).reset_index()
    df0.columns=['row', 'col', metric_name]

    df = df0[df0['row']!=df0['col']].reset_index(drop=True)

    if len(add_values) > 0:
        for k, v in add_values.items():
            df[k] = [v for _ in np.arange(0, df.shape[0])]

    return df


def get_pw_distance(cc_, pos_, xcoord='ml_pos', ycoord='ap_pos', label='cortical_distance',
                    add_eccentricity=False):
    '''Given DF of pairwise calcs (cc_), calculate corresponding cells' POS diff
    using pos_. xcoord and ycoord must be columns in pos_.

    Returns cc_ with position diffs.
    '''
    # Get current FOV rfdata and add position info to sigcorrs df
    cc_['cell_1'] = cc_['cell_1'].astype(int)
    cc_['cell_2'] = cc_['cell_2'].astype(int)

    r1 = cc_['cell_1'].unique()
    r2 = cc_['cell_2'].unique()
    #crois_ = np.union1d(r1, r2)
    #assert len([r for r in crois_ if r not in pos_.index.tolist()])==0, \
    #    "[%s, %s]: incorrect roi indexing in RFDATA" % (va, dk)
    if 'cell' in pos_.columns:
        pos_.index = pos_['cell'].values
    # Coords of cell1 in pair, in order
    coords1 = np.array(pos_.loc[cc_['cell_1'].values][[xcoord, ycoord]])
    # Coords of cell2 in pair 
    coords2 = np.array(pos_.loc[cc_['cell_2'].values][[xcoord, ycoord]])
    # Get dists, in order of appearance
    dists = [np.linalg.norm(c1-c2) for c1, c2 in zip(coords1, coords2)]
    cc_[label] = dists
    
    if add_eccentricity and 'x0' in pos_.columns: 
        if 'eccentricity' not in pos_.columns:
            null_ixs = pos_[pos_['x0'].isnull()].index.tolist()
            eccs = np.sqrt((pos_[['x0', 'y0']]**2).sum(axis=1))
            eccs.loc[null_ixs] = np.nan 
            pos_.loc[eccs.index, 'eccentricity'] = eccs

        v1 = pos_.loc[cc_['cell_1'].values]['eccentricity'].values
        v2 = pos_.loc[cc_['cell_2'].values]['eccentricity'].values
        cc_['max_ecc'] = [max([i, j]) if not(any([np.isnan(i), np.isnan(j)])) \
                            else np.nan for i, j in zip(v1, v2)]
        cc_['min_ecc'] = [min([i, j]) if not(any([np.isnan(i), np.isnan(j)])) \
                            else np.nan for i, j in zip(v1, v2)]
 
    if label!='cortical_distance':
        coords1 = np.array(pos_.loc[cc_['cell_1'].values][['ml_pos', 'ap_pos']])
        coords2 = np.array(pos_.loc[cc_['cell_2'].values][['ml_pos', 'ap_pos']])
        dists_c = [np.linalg.norm(c1-c2) for c1, c2 in zip(coords1, coords2)]
        cc_['cortical_distance'] = dists_c


    return cc_

def do_pairwise_diffs_melt(df_, metric_name='morph_sel', include_diagonal=False):
    '''Calculate DIFFERENCE in metric_name for all pairs of cells---untested.'''

    pairwise_diffs = pd.DataFrame(abs(df_[metric_name].values \
                                  - df_[metric_name].values[:, None]), 
                              columns=df_['cell'].values, index=df_['cell'].values)

    diffs = melt_square_matrix(pairwise_diffs, metric_name=metric_name, include_diagonal=False)
    diffs = diffs.rename(columns={'row': 'cell_1', 'col': 'cell_2'})
    diffs[['cell_1', 'cell_2']] = diffs[['cell_1', 'cell_2']].astype(int)
    diffs['neuron_pair'] = ['%i_%i' % (c1, c2) for \
                         c1, c2 in diffs[['cell_1', 'cell_2']].values]
    return diffs

def aggregate_ccdist(NDATA, experiment='gratings', rfdf=None, SDF=None, min_ncells=10, 
                select_stimuli='fullfield', distance_var='rf_distance', verbose=False):
    '''
    Cycle thru all datasets and calculate CCs and PW distances.
    
    Returns:
    CORRS: pd.DataFrame()
        All pw signal- and noise-corrs, plus distances (RF, and/or cortical)
        
    Args:
    
    selective_stimuli: (str, None)
        fullfield: only include FF stimuli when calculating PW corrs (must provide SDF)
        images:  only include apertured or image stimuli (must provide SDF)
        None:  include it all (SDF can be None)
    
    rfdf: (None, pd.DataFrame)
        All RF fit data (rfutils.aggregate_fits()). If None, ignores RF calculations.
    
    SDF: (None, pd.DataFrame)
        Aggregate sdfs across datasets. Include if selective_stimulis is not None.
    
    distance_var: (str)
        rf_distance:  Calculate PW dists bw RF centers (must provide rfdf). 
                      Also calculates cortical_distance anyway.
        cortical_distance:  Calculate cortical dists only (no RFs)
    '''
    # NDATA already contains only unique dkeys
    #print(experiment)
    CORRS=None
    #     min_ncells=10
    #     selective_only=False
    #     select_stimuli = 'fullfield'
    # ------------------------------------------------------------
    distance_var = 'rf_distance' if rfdf is not None else 'cortical_distance'
    print("Dist: %s" % distance_var)

    xcoord = 'x0' if distance_var=='rf_distance' else 'ml_pos'
    ycoord = 'y0' if distance_var=='rf_distance' else 'ap_pos'
    wrong_configs=[]
    no_rfs=[]
    c_list=[]
    for (va, dk, exp), ndf in NDATA.groupby(['visual_area', 'datakey', 'experiment']):
        rfdf_=None
        if rfdf is not None:
            rfdf_ = rfdf[(rfdf.visual_area==va) & (rfdf.datakey==dk)].copy()
            if rfdf_.shape[0]==0:
                no_rfs.append((va, dk, exp))
                continue
        rois_ = ndf['cell'].unique()
        # Select cells
        if len(rois_)<min_ncells:
            print("Skipping - (%s, %s)" % (va, dk))
            continue
        # Select stimuli and trials
        if experiment in ['gratings', 'blobs']:
            sdf=SDF[SDF.datakey==dk].copy()
            curr_cfgs = aggr.get_included_stimconfigs(sdf, experiment=exp,
                                                     select_stimuli=select_stimuli)
            if len(curr_cfgs)==0:
                wrong_configs.append((va, dk))
                continue
        else:
            curr_cfgs = sorted(NDATA['config'].unique())
        roidf_ = get_roi_pos_and_rfs(ndf, curr_rfs=rfdf_, rfs_only=False)
        cc_ = get_ccdist(ndf, roidf_, return_zscored=False,
                            curr_cfgs=curr_cfgs,
                            xcoord=xcoord, ycoord=ycoord, label=distance_var,
                            add_eccentricity=True)
        cc_['visual_area'] = va
        cc_['datakey'] = dk
        cc_['experiment'] = experiment
        cc_['n_cells'] = len(rois_)
        c_list.append(cc_)
    CORRS = pd.concat(c_list, ignore_index=True)
    
    if verbose:
        print('%i datasets w wrong configs:' % len(wrong_configs))
        for w in wrong_configs:
            print("    %s" % str(w))
        print('%i datasets w/out RF fits:' % len(no_rfs))
        for w in no_rfs:
            print("    %s" % str(w))
            
    return CORRS

# --------------------------------------------------------------------
# Tuning similarity calculations  (specific to blobs, gratings)
# --------------------------------------------------------------------
def smallest_signed_angle(x, y, TAU=360):
    a = (x - y) % TAU
    b = (y - x) % TAU
    return -a if a < b else b

def get_pw_angular_dist(df_, tau=180, in_name='input', out_name='output'):
    '''Calculate ANGULAR diffs (corrected), both signed and abs'''

    col_pairs = list(itertools.combinations(df_['cell'], 2))
    pairdf = pd.DataFrame(['%i_%i' % (a, b) for a, b \
                           in col_pairs], columns=['neuron_pair'])
    pairdf['cell_1'] = [a for a, b in col_pairs]
    pairdf['cell_2'] = [b for a, b in col_pairs]
    pairdf[out_name] = [smallest_signed_angle(\
                              float(df_[df_['cell']==a][in_name]), 
                               float(df_[df_['cell']==b][in_name]), TAU=tau) \
                        for a, b in col_pairs]
    pairdf['%s_abs' % out_name] = [abs(smallest_signed_angle(\
                              float(df_[df_['cell']==a][in_name]), 
                               float(df_[df_['cell']==b][in_name]), TAU=tau)) \
                        for a, b in col_pairs]
    return pairdf

def get_pw_diffs(df_, metric='response_pref'):
    '''Get abs DIFFERENCE for a given metric for all pairs of cells
    '''
    col_pairs = list(itertools.combinations(df_['cell'], 2))
    pairdf = pd.DataFrame(['%i_%i' % (a, b) for a, b \
                           in col_pairs], columns=['neuron_pair'])
    pairdf['cell_1'] = [a for a, b in col_pairs]
    pairdf['cell_2'] = [b for a, b in col_pairs]
    pairdf[metric] = [abs(float(df_[df_['cell']==a][metric])\
                      - float(df_[df_['cell']==b][metric])) \
                        for a, b in col_pairs]
    return pairdf

def cosine_similarity(v1, v2):
    '''Cosine similarity bw two vectors'''
    return (v1.dot(v2)) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))


def get_paired_tuning_metrics(fitdf, r1, r2):
    '''
    Unused for now. Creates a "tuning curve" of all fit metrics from OSI-curve fitting.
    '''
    tuning_params = ['response_pref', 'response_null', 'theta_pref', 'sigma',
                     'response_offset', 'asi', 'dsi', 'circvar_asi', 'circvar_dsi', 
                     'sf', 'size', 'speed', 'tf']
    d_=[]
    for ri in [r1, r2]:
        d1 = pd.DataFrame({'param': fitdf.loc[ri][tuning_params].index.tolist(),
                            'value': fitdf.loc[ri][tuning_params].values})
        d1['cell'] = ri
        d_.append(d1)
    d0 = pd.concat(d_, axis=0)
    cosim_m1 = cosine_similarity(d0[d0['cell']==r1]['value'].values, 
                                d0[d0['cell']==r2]['value'].values)
    # normalize values
    d0.loc[d0.param=='size', 'value'] = d0[d0.param=='size']['value'] /200. 
    d0.loc[d0.param=='speed', 'value'] = d0[d0.param=='speed']['value'] /20. 
    d0.loc[d0.param=='theta_pref', 'value'] = d0[d0.param=='theta_pref']['value'] /360. 
    d0.loc[d0.param=='sigma', 'value'] = d0[d0.param=='sigma']['value'] /180. 
    d0.loc[d0.param=='tf', 'value'] = d0[d0.param=='tf']['value'] /10. 

    cosim_m = cosine_similarity(d0[d0['cell']==r1]['value'].values, 
                                d0[d0['cell']==r2]['value'].values)
    return d0, cosim_m


def compare_curves(curve1, curve2, a=0, b=1):
    '''
    Calculate metrics of similarity between 2 vectors (curve1, curve2), 
    corresponding to neuron a and b.
    
    Returns:
    
    res: pd.DataFrame()
    
    '''
    # cross-correlation
    ccorr = signal.correlate(curve1, curve2)
    lags = signal.correlation_lags(len(curve1), len(curve2))
    lagzero = list(lags).index(0)
    xcorr = ccorr[lagzero]
    # do pearson's corr
    cc, pv = spstats.pearsonr(curve1, curve2)
    # do cosine similarity
    cosim = cosine_similarity(curve1, curve2)
    # combine
    res = pd.Series({'xcorr': xcorr, 'pearsons': cc,  'cosim': cosim,
                      'cell_1': int(a), 'cell_2': int(b), 
                      'neuron_pair': '%i_%i' % (a, b)})

    return res


def compare_direction_tuning_curves(thetas, fitdf, a=0, b=1):
    '''Given fit params (gratings fits), get tuning curves, calculate corrs'''
    # Tuning curves
    params = ['response_pref', 'response_null', 'theta_pref', 
              'sigma', 'response_offset']
    curve1 = osi.double_gaussian(thetas, *fitdf[params].loc[a])
    curve2 = osi.double_gaussian(thetas, *fitdf[params].loc[b])

    res = compare_curves(curve1, curve2, a=a, b=b)

    return res


def get_pw_tuning(fitdf, n_intervals=3, stimulus='gratings',
                    sort_best_size=True, normalize=True):
    '''
    Cycle thru cell pairs and calculate cross-corr (0 lag), pearson's corr, and cosine similarity
    for tuning curves.
    Prev. called get_pw_curve_correlations()
    '''
    # Get pairs
    rois_ = fitdf['cell'].unique()
    col_pairs = list(itertools.combinations(rois_, 2))

    if stimulus=='gratings':
        fitdf.index = fitdf['cell'].values
        tested_thetas = np.arange(0, 360, 45)
        thetas = osi.interp_values(tested_thetas, n_intervals=n_intervals, wrap_value=360)
        # do corrs for all pairs
        t = [compare_direction_tuning_curves(thetas, fitdf, a=a, b=b) for (a, b) in col_pairs]
        df_ = pd.concat(t, axis=1).T

    elif stimulus=='blobs':
        # morph and size tuning curves
        morph_mat, size_mat = sel.get_object_tuning_curves(fitdf, 
                                            sort_best_size=sort_best_size, 
                                            normalize=normalize, 
                                            return_stacked=False)
        # pw morph tuning similarities 
        m_list = [compare_curves(morph_mat[a].values, morph_mat[b].values, a=a, b=b) \
                                                for (a, b) in col_pairs]
        df_morph = pd.concat(m_list, axis=1).T
        for p in ['xcorr', 'pearsons', 'cosim']:  
            df_morph[p] = df_morph[p].astype(float)
        # pw size tuning similarities 
        s_list = [compare_curves(size_mat[a].values, \
                                    size_mat[b].values, a=a, b=b) \
                                for (a, b) in col_pairs]
        df_size = pd.concat(s_list, axis=1).T
        for p in ['xcorr', 'pearsons', 'cosim']:  
            df_size[p] = df_size[p].astype(float)
        # combine
        df_ = pd.merge(df_morph, df_size, on=['cell_1', 'cell_2', 'neuron_pair'],
                        suffixes=('_morph', '_size'))

    return df_


def correlate_pw_tuning_in_fov(df_, n_intervals=3, stimulus='gratings',
                        sort_best_size=True, normalize=True):
    '''
    Calculate correlations and distances for cell pairs in FOV.
    posdf_ must be a subset of df_.
    '''
    # Calculate CCs
    tuning_cc = get_pw_tuning(df_, n_intervals=n_intervals, stimulus=stimulus,
                            sort_best_size=sort_best_size, normalize=normalize)
    #diffs_ = cc.copy()
    # Calculate distances
#    if 'x0' in posdf_.columns:
#        dists = get_pw_distance(tuning_cc.copy(), posdf_, xcoord='x0', ycoord='y0', 
#                                 label='rf_distance', add_eccentricity=True)
#    else:
#        dists = get_pw_distance(tuning_cc.copy(), posdf_, xcoord='ml_pos', ycoord='ap_pos', 
#                                 label='cortical_distance', add_eccentricity=False)
#    # check and return
#    assert dists.shape[0]==tuning_cc.shape[0], 'Bad merging: %s, %s' (va, dk)

    if stimulus=='gratings':
        # Calculate angular diffs
        dir_diff = get_pw_angular_dist(df_, tau=360,
                                       in_name='theta_pref', out_name='pref_dir_diff')
        ori_diff = get_pw_angular_dist(df_, tau=180,
                                       in_name='theta_pref', out_name='pref_ori_diff')
        ang_diff = pd.merge(dir_diff, ori_diff, on=['neuron_pair', 'cell_1', 'cell_2'],
                           how='outer')      
        # Calculate standard diffs
        resp_diff = get_pw_diffs(df_, metric='response_pref') 
        sigma_diff = get_pw_diffs(df_, metric='sigma') 
        std_diff = pd.merge(sigma_diff, resp_diff, on=['neuron_pair', 'cell_1', 'cell_2'],
                            how='outer')
        # merge diffs
        diffs = pd.merge(ang_diff, std_diff, on=['neuron_pair', 'cell_1', 'cell_2'],
                         how='outer')
        #assert dists.shape[0] == diffs.shape[0], "bad merging"
        pw_df = pd.merge(tuning_cc, diffs, on=['neuron_pair', 'cell_1', 'cell_2'],
                        how='outer')
    else:
#         # Calculate standard diffs
        max_resp = df_.groupby('cell')['response'].max().reset_index()
        resp_diff = get_pw_diffs(max_resp, metric='response') 
        resp_diff = resp_diff.rename(columns={'response': 'max_response'})
        pw_df = pd.merge(tuning_cc, resp_diff, on=['neuron_pair', 'cell_1', 'cell_2'],
                        how='outer')

    cols = [k for k in pw_df.columns if k not in ['visual_area', 'datakey', 'neuron_pair']]
    for c in cols:
        pw_df[c] = pw_df[c].astype(float)
    
    return pw_df


def aggregate_tuning_curve_ccdist(df, rfdf=None, rfpolys=None, n_intervals=3, 
                                min_ncells=5, stimulus='gratings',
                                sort_best_size=True, normalize=True):
    '''
    Calculate PW diffs for GRATINGS (+ RFs, if have).

    Args:
    
    df: (pd.DataFrame)
        For each cell, fit params (and sometimes position or RF info).

    n_intervals: (int)
        N vals to interp steps (0, 45, 90, etc.), Default is 9.

    '''
    no_rfs=[]
    a_=[]
    for (va, dk), df_ in df.groupby(['visual_area', 'datakey']):
        if len(df_['cell'].unique())<min_ncells:
            print("too few cells: %s, %s" % (va, dk))
            continue

        if rfdf is not None:
            curr_rfs = rfdf[(rfdf.visual_area==va) & (rfdf.datakey==dk)].copy()
            curr_cells = df_[['visual_area', 'datakey', 'experiment', 'cell']].drop_duplicates().copy()
            # Add roi positions and RF fits -- need to sub-select for indexing.
            posdf_ = get_roi_pos_and_rfs(curr_cells, curr_rfs, rfs_only=False, #position_only=False,
                                        merge_cols=['visual_area', 'datakey', 'cell'])
            # to merge:
            # fits_nd_rfs = pd.merge(df_, posdf_, on=['visual_area', 'datakey', 'cell'], how='outer')
            if posdf_.shape[0]<2:
                no_rfs.append((va, dk, exp))
                continue
        else:
            posdf_ = aggr.add_roi_positions(df_)

        # Get PW differences (tuning) and distances (position)
        tuning_dists = correlate_pw_tuning_in_fov(df_, # posdf_, 
                                            n_intervals=n_intervals, stimulus=stimulus,
                                            sort_best_size=sort_best_size, normalize=normalize) 
        # Cortical and RF position distances
        dists = get_pw_distance(tuning_dists, posdf_, xcoord='x0', ycoord='y0', 
                                 label='rf_distance', add_eccentricity=True)

        # RF-to-RF overlaps, if relevant
        curr_polys = None
        if rfpolys is not None:
            dists0 = dists.copy()
            rois_ = df_['cell'].unique() 
            curr_polys = rfpolys[(rfpolys.datakey==dk) & (rfpolys['cell'].isin(rois_))] 
            if len(curr_polys)<=1: # need >1 to compare
                print("    (%s NONE, skipping overlaps)" % dk)  
                curr_polys=None
        if rfdf is not None:
            rf_diffs = rf_diffs_and_dists_in_fov(dists0, posdf_, curr_polys=curr_polys)
            pw_df= pd.merge(dists, rf_diffs, on=['neuron_pair', 'cell_1', 'cell_2'], 
                            how='outer')
            pw_df['area_overlap'] = pw_df['area_overlap'].astype(float)
            pw_df['perc_overlap'] = pw_df['perc_overlap'].astype(float) 
            pw_df['overlap_index'] = 1-pw_df['area_overlap']
            
        else:
            pw_df = dists.copy()

        if pw_df is not None:
            pw_df['visual_area'] = va
            pw_df['datakey'] = dk
            pw_df['n_cells'] = len(df_['cell'].unique())
            a_.append(pw_df)

    aggr_dists = pd.concat(a_, axis=0, ignore_index=True)
    print(no_rfs)
    return aggr_dists


def rf_diffs_and_dists_in_fov(dists, df_, curr_polys=None):
    '''
    For 1 dataset, calculate RF-to-RF VF and CX distances, 
    plus:
    area_overlap	perc_overlap	rf_angle_diff	rf_angle_diff_abs	std_x	std_y
    aspect_ratio	neuron_pair
    '''

    # RF-to-RF overlaps
    overlaps_ = dists[['neuron_pair', 'cell_1', 'cell_2']].copy()
    overlaps_['area_overlap'] = None
    overlaps_['perc_overlap'] = None
    if curr_polys is not None and len(curr_polys)>1:
        try:
            overlaps_ = rfutils.get_rf_overlaps(curr_polys)
            overlaps_ = overlaps_.rename(columns={'poly1': 'cell_1', 'poly2': 'cell_2'})
            overlaps_['neuron_pair'] = ['%i_%i' % (c1, c2) for c1, c2 \
                                            in overlaps_[['cell_1', 'cell_2']].values] 
        except Exception as e:
            pass
    pos_and_overlaps = pd.merge(dists, overlaps_, on=['neuron_pair', 'cell_1', 'cell_2'])
           
    # RF angle diffs, merge w overlap
    angles_ = get_pw_angular_dist(df_, tau=180,
                                  in_name='rf_theta_deg', out_name='rf_angle_diff')
    angles_['rf_angle_diff_abs'] = angles_['rf_angle_diff'].abs()

    ang_diffs = pd.merge(pos_and_overlaps, angles_, on=['neuron_pair', 'cell_1', 'cell_2'])

    # Standard difference metrics, merge 
    sz_x = get_pw_diffs(df_, metric='std_x')
    sz_y = get_pw_diffs(df_, metric='std_y')
    sz_diff = pd.merge(sz_x, sz_y, on=['neuron_pair', 'cell_1', 'cell_2'])
    # non-size stuff
    asp_diff = get_pw_diffs(df_, metric='aspect_ratio')
    sz_and_aspect = pd.merge(sz_diff, asp_diff, on=['neuron_pair', 'cell_1', 'cell_2'])

    # Final df
    pw_df0 = pd.merge(ang_diffs, sz_and_aspect, on=['neuron_pair', 'cell_1', 'cell_2'])
    # Get rid of extra columns for merge
    new_cols = [k for k in pw_df0.columns if k not in dists.columns]
    new_cols.extend(['neuron_pair', 'cell_1', 'cell_2'])
    pw_df = pw_df0[new_cols]

    pw_df['area_overlap'] = pw_df['area_overlap'].astype(float)
    pw_df['perc_overlap'] = pw_df['perc_overlap'].astype(float) 
    pw_df['overlap_index'] = 1-pw_df['area_overlap']

    return pw_df



def aggregate_angular_dists(df, min_ncells=5):
    '''
    Calculate PW diffs for GRATINGS (+ RFs, if have). Not used.
    '''
    a_=[]
    for (va, dk), df_ in df.groupby(['visual_area', 'datakey']):
        if len(df_['cell'].unique())<min_ncells:
            print("too few cells: %s, %s" % (va, dk))
            continue
        # diff in pref. thteas
        dir_diff = get_pw_angular_dist(df_, tau=360,
                                       in_name='theta_pref', out_name='pref_dir_diff')
        ori_diff = get_pw_angular_dist(df_, tau=180,
                                       in_name='theta_pref', out_name='pref_ori_diff')
        ang_diff = pd.merge(dir_diff, ori_diff, on=['neuron_pair', 'cell_1', 'cell_2']) 
        
        resp_diff = get_pw_diffs(df_, metric='response_pref') 
        sigma_diff = get_pw_diffs(df_, metric='sigma') 
        nonang_diff = pd.merge(resp_diff, sigma_diff, on=['neuron_pair', 'cell_1', 'cell_2'])
 
        gratings_diff = pd.merge(ang_diff, nonang_diff, on=['neuron_pair', 'cell_1', 'cell_2'])

        if 'rf_theta_deg' in df_.columns:
            # RF angle diffs
            rf_diff = get_pw_angular_dist(df_, tau=180,
                                          in_name='rf_theta_deg', out_name='rf_angle_diff')
            diffs_ = pd.merge(gratings_diff, rf_diff, on=['neuron_pair', 'cell_1', 'cell_2'])
        else:
            diffs_ = gratings_diff.copy()

        # Cortical and RF difff
        if 'x0' in df_.columns:
            adist = get_pw_distance(diffs_, df_, xcoord='x0', ycoord='y0', 
                                     label='rf_distance', add_eccentricity=True)
        else:
            adist = get_pw_distance(diffs_, df_, xcoord='ml_pos', ycoord='ap_pos', 
                                     label='cortical_distance', add_eccentricity=False)

        assert adist.shape[0]==gratings_diff.shape[0], 'Bad merging: %s, %s' (va, dk)
        adist['visual_area'] = va
        adist['datakey'] = dk
        adist['n_cells'] = len(df_['cell'].unique())
        a_.append(adist)
    angdists = pd.concat(a_, axis=0, ignore_index=True)

    return angdists

def get_pw_rf_diffs(df_):
    '''Calculate standard PW diffs for RF dataframe (1 fov)'''
    diffs=None
    dfs_to_merge=[]
    # RF angle diffs, merge w overlap
    angles_ = get_pw_angular_dist(df_, tau=180,
                                  in_name='rf_theta_deg', out_name='rf_angle_diff')
    angles_['rf_angle_diff_abs'] = angles_['rf_angle_diff'].abs()
    dfs_to_merge.append(angles_)
    # Standard difference metrics, merge 
    sz_x = get_pw_diffs(df_, metric='std_x')
    sz_y = get_pw_diffs(df_, metric='std_y')
    #sz_diff = pd.merge(sz_x, sz_y, on=['neuron_pair', 'cell_1', 'cell_2'])
    # non-size stuff
    asp_diff = get_pw_diffs(df_, metric='aspect_ratio')
    dfs_to_merge.extend([sz_x, sz_y, asp_diff])
    # Cortical and RF position distances
    diffs_for_ix = angles_[['neuron_pair', 'cell_1', 'cell_2']].copy()
    dists = get_pw_distance(diffs_for_ix, df_, xcoord='x0', ycoord='y0', 
                             label='rf_distance', add_eccentricity=True)
    dfs_to_merge.append(dists)
    
    diffs = reduce(lambda  left,right: pd.merge(left,right,\
                                            on=['neuron_pair', 'cell_1', 'cell_2'],
                                            how='outer'), dfs_to_merge)
    return diffs

def aggregate_rf_dists(rfdf, rfpolys=None, min_ncells=5):
    '''
    Calculate PW diffs for RFs
    '''
    a_=[]
    for (va, dk), df_ in rfdf.groupby(['visual_area', 'datakey']):
        if len(df_['cell'].unique())<min_ncells:
            print("too few cells: %s, %s" % (va, dk))
            continue
        # Standard RF diffs
        diffs = get_pw_rf_diffs(df_)
        # RF-to-RF overlaps
        overlaps_ = diffs[['neuron_pair', 'cell_1', 'cell_2']].copy()
        overlaps_['area_overlap'] = None
        overlaps_['perc_overlap'] = None
        if rfpolys is not None:
            try:
                rois_ = df_['cell'].unique()
                curr_polys = rfpolys[(rfpolys.datakey==dk) & (rfpolys['cell'].isin(rois_))]
                print(dk, curr_polys.shape)
                if len(curr_polys)<=1:
                    print("NONE, skipping overlaps")  
                overlaps_ = rfutils.get_rf_overlaps(curr_polys)
                overlaps_ = overlaps_.rename(columns={'poly1': 'cell_1', 'poly2': 'cell_2'})
                overlaps_['neuron_pair'] = ['%i_%i' % (c1, c2) for c1, c2 \
                                                in overlaps_[['cell_1', 'cell_2']].values] 
            except Exception as e:
                pass
        dfs_to_merge = [diffs, overlaps_]
        # combine
        pw_df = reduce(lambda  left,right: pd.merge(left,right,\
                                            on=['neuron_pair', 'cell_1', 'cell_2'],
                                            how='outer'), dfs_to_merge)

        # Final df
        # pw_df = pd.merge(ang_diffs, sz_and_aspect, on=['neuron_pair', 'cell_1', 'cell_2'])
        assert diffs.shape[0]==pw_df.shape[0], 'Bad merging: %s, %s' (va, dk)
        pw_df['visual_area'] = va
        pw_df['datakey'] = dk
        pw_df['n_cells'] = len(df_['cell'].unique())
        a_.append(pw_df)
    aggr_dists = pd.concat(a_, axis=0, ignore_index=True)

    return aggr_dists





# Data binning
# ----------------------------------------------------------------------------

def get_bins_and_cut(DISTS, equal_bins=False, n_bins=10, 
                ctx_step=10, rf_step=2.5, area_step=0.05, overlap_step=0.05):
    #n_bins=10
    df = DISTS.copy()
    # Split distances into X um bins
    ctx_maxdist = np.ceil(DISTS['cortical_distance'].max())
    # ctx_step=10
    if equal_bins:
        ctx_bins = np.linspace(0, ctx_maxdist, n_bins)
    else:
        ctx_bins = np.arange(0, ctx_maxdist+ctx_step, ctx_step)
    #df = cr.cut_bins(df, ctx_bins, 'cortical_distance')

    # rf_step=2.5
    rf_maxdist = np.ceil(DISTS['rf_distance'].max())
    if equal_bins:
        rf_bins = np.linspace(0, rf_maxdist, n_bins)
    else:
        rf_bins = np.arange(0, rf_maxdist+rf_step, rf_step)
    #df = cr.cut_bins(df, rf_bins, 'rf_distance')

    #perc_step = 0.05
    area_bins = np.arange(0, 1+area_step, area_step)
    #df = cr.cut_bins(df, area_bins, 'area_overlap')

    #perc_step = 0.02
    overlap_bins = np.arange(0, 1+overlap_step, overlap_step)
    #df = cr.cut_bins(df, overlap_bins, 'overlap_index')

    # Split
    dist_lut = {'cortical_distance': 
                            {'bins': ctx_bins, 'step': ctx_step, 'max_dist': ctx_maxdist},
                'rf_distance': 
                            {'bins': rf_bins, 'step': rf_step, 'max_dist': rf_maxdist},
                'overlap_index': 
                            {'bins': overlap_bins, 'step': overlap_step, 'max_dist': 1}, 
                'area_overlap': 
                            {'bins': area_bins, 'step': area_step, 'max_dist': 1} 
               }

    for param, pdict in dist_lut.items():
        df = cut_bins(df, pdict['bins'], param)

    return df, dist_lut


def cut_bins(DF, bins, metric='cortical_distance', include_lowest=True):
    DF['binned_%s' % metric] = pd.cut(DF[metric], bins, include_lowest=include_lowest,
                                                labels=bins[0:-1])
    DF['%s_label' % metric] = [float(d) if not np.isnan(d) else d \
                                for d in DF['binned_%s' % metric].values]

    DF = get_bin_values(DF, bin_var=metric)

    return DF

def get_bin_values(DF, bin_var='cortical_distance'):
    grouped_meds = DF.groupby('%s_label' % bin_var).median().reset_index()
    bin_lut = dict((k, float(v[bin_var])) for k, v \
               in grouped_meds.groupby('%s_label' % bin_var))
    DF['%s_value' % bin_var] = [bin_lut[i] if not np.isnan(i) else i \
                                for i in DF['%s_label' % bin_var].values]
    return DF


def get_bins_within_limits(bcorrs, bin_name='cortical_distance', 
                    upper_lim=None, lower_lim=None):
    currdf = bcorrs.copy()
    if upper_lim is not None and lower_lim is not None:
        currdf = bcorrs[(bcorrs[bin_name]<upper_lim)
                    & (bcorrs[bin_name]>lower_lim)].copy()
    elif upper_lim is not None:
        currdf = bcorrs[(bcorrs[bin_name]<upper_lim)].copy()
    elif lower_lim is not None:
        currdf = bcorrs[(bcorrs[bin_name]>lower_lim)].copy()
    else:
        print("No limits specified. Returning same.")

    return currdf


def get_binned_X(currdf, x_label='signal_cc',  x_bins=None, min_npairs=10, labels=None):
    # Check counts
    currdf['binned_%s' % x_label], bin_edges = pd.cut(currdf[x_label], \
                                                x_bins, labels=labels, retbins=True,
                                                include_lowest=True)
    curr_bin_counts = currdf.groupby(['visual_area',\
                     'binned_%s' % x_label])['neuron_pair'].count()

    pass_ = pd.concat([currdf[(currdf.visual_area==va) 
                        & (currdf['binned_%s' % x_label]==bc)] \
                for (va, bc) in curr_bin_counts[curr_bin_counts>=min_npairs]\
                       .index.tolist()], axis=0)
    return pass_


# -------------------------- for exp decay version
def get_bins(n_bins=4, custom_bins=False, cmap='viridis'):
    '''Get generic bins and bin labels to split data up'''
    qcolor_list = sns.color_palette(cmap, n_colors=n_bins)
    # Bin into quartiles
    bins = [0, 200, 350, 400, np.inf] if custom_bins \
                    else np.arange(0, n_bins+1).astype(int)
    bin_labels = custom_bin_labels(bins) if custom_bins \
                    else [hutils.make_ordinal(i) for i in bins[1:]]
    bin_colors = dict((k, v) for k, v in zip(bin_labels, qcolor_list))
    
    return bins, bin_labels, bin_colors


def custom_bin_labels(bins):
    labels=[]
    for bi, b in enumerate(bins[0:-1]):
        if bi==0:
            lb = '<%i' % bins[bi+1]
        elif b==bins[-2]:
            lb = '>%i' % b
        else:
            lb = '%i-%i' % (b, bins[bi+1])
        labels.append(lb)

    return labels
 
def bin_column_values(cc_, to_quartile='cortical_distance', use_quartile=True,
                     n_bins=4, labels=False, bins=None, return_bins=False):
    '''
    Split column into quartiles (n_bins=4) or custom N bins.
    to_quartile: str
        Column to bin
    use_quartile: bool
        Set to use quartiles (evenly populated bins), otherwise, use even sized bins
    n_bins: int
        Number of bins
    '''
    # print("binning: %s" % bin_type)
    # bins=[0, 100, 300, 500, np.inf], 

    if bins is not None:
        if labels is None:
            labels = custom_bin_labels(bins)
        cc_['binned_%s' % to_quartile] = pd.cut(x=cc_[to_quartile], 
                                bins=bins, 
                                labels=labels, include_lowest=True)

    if use_quartile:
        cc_['binned_%s' % to_quartile], bin_edges = pd.qcut(cc_[to_quartile], \
                                        n_bins, labels=labels, retbins=True,
                                        include_lowest=True)
    else:
        cc_['binned_%s' % to_quartile], bin_edges = pd.cut(cc_[to_quartile], \
                                         n_bins,labels=labels, retbins=True,
                                         include_lowest=True)
    if return_bins:
        return cc_, bin_edges
    else:
        return cc_



# --------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------
def get_key_stimulus_params(experiment):
    # Get mean response per condition (columns=cells, rows=conditions)
    if experiment=='gratings':
        params=['ori', 'sf', 'size', 'speed']
    elif experiment=='blobs':
        params=['morphlevel', 'size'] #, 'yrot', 'xpos', 'ypos']
    elif experiment=='rfs':
        params = ['position']

    return params

def get_correlation_matrix(tuning_, sdf, experiment='blobs', method='spearman'):
    '''
    Return correlation matrix (nconfigs x nconfigs)
    Args
    tuning_ (pd.DataFrame)
        rows=trial-averaged response to each config, cols=rois
    method: (str)
        Correlation method to use (should be valid arg for pd.corr())
    sdf: pd.DataFrame
    
    Returns:
    mat_: (pd.DataFrame)
        nconfigs x nconfigs correlation matrix
    msk_: (np.array)
        Mask to just plot lower triangle
    xlabels: (list)
        Strings of all config parameters
    '''
    params = get_key_stimulus_params(experiment)
    rois_ = [i for i in tuning_.columns if hutils.isnumber(i)]
    if experiment == 'gratings':
        sdf['tf'] = sdf['sf']*sdf['speed']
        xlabels = ['%i|%.2f|%i|%i' % (o, sf, sz, sp) for (o, sf, sz, sp) in \
                   sdf.loc[tuning_['config']][['ori', 'sf', 'size', 'speed']].values]
    elif experiment=='blobs':
        xlabels = ['%i|%.i' % (mp, sz) for (mp, sz) in\
                   sdf.loc[tuning_['config']][params].values]
    elif experiment == 'rfs':
        xlabels = ['%s' % str(p[0]) for p in\
                   sdf.loc[tuning_['config']][params].values]
    mat_ = tuning_[rois_].T.corr(method=method)
    msk_ = np.triu(np.ones_like(mat_, dtype=bool))
    return mat_, msk_, xlabels

def plot_correlation_matrix(mat_, plot_rdm=True, ax=None,
                           vmin=None, vmax=None):
    pplot.set_plot_params()

    if plot_rdm:
        pmat = 1-mat_
        cmap='viridis' # 0=totally same, 2+ more dissimilar
        plot_name = 'rdm'
    else:
        pmat = mat_.copy()
        cmap = 'RdBu_r' # red, corr=1, blue, anticorr=-1
        plot_name = 'corr'
    if ax is None:
        fig, ax = pl.subplots(1,1,figsize=(10, 6))
    if vmin is None or vmax is None:
        vmin, vmax = pmat.min().min(), pmat.max().max()

    sns.heatmap(pmat, cmap=cmap, ax=ax, #mask=msk_, 
                square=True, cbar_kws={"shrink": 0.5}, vmin=vmin, vmax=vmax)
    #Below 3 lines remove default labels
    labels = ['' for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    ax.set_ylabel('')
    ax.set_xticks([])
    pplot.label_group_bar_table(ax, mat_, offset=0.1, lw=0.5)
    return ax



def plot_quartile_dists_FOV(cc_, metric='signal_cc', to_quartile='cortical_distance',
                            bin_colors=None, bin_labels=None,
                            plot_median=True, extrema_only=True, ax=None,
                            legend=True, cumulative=False, element='bars', fill=True,
                            lw=1, common_norm=True, common_bins=True):
    pplot.set_plot_params()

    if bin_labels is None:
        bin_labels = sorted(cc_['binned_%s' % to_quartile].unique(), \
                            key=hutils.natural_keys)
    # Plot metric X for top and bottom quartiles
    if extrema_only:
        plot_bins = [bin_labels[0], bin_labels[-1] ] # [0, 1, 2, 3]
    else:
        plot_bins = bin_labels
    # colors 
    if bin_colors is None:
        nb = len(plot_bins)
        qcolor_list = sns.color_palette('cubehelix', n_colors=nb)
        bin_colors = dict((k, v) for k, v in zip(np.arange(0, nb), qcolor_list))
      
    currd = cc_[cc_['binned_%s' % to_quartile].isin(plot_bins)]
    if ax is None:
        fig, ax = pl.subplots()
    sns.histplot(data=currd, x=metric, hue='binned_%s' % to_quartile, ax=ax,
                stat='probability', common_norm=common_norm, common_bins=common_bins,
                palette=bin_colors, legend=legend, cumulative=cumulative, 
                element=element, fill=fill, line_kws={'lw': lw})
    #g = sns.displot(data=currd, x=metric,
    #            hue='binned_%s' % to_quartile, legend=True, palette=bin_colors)
    
    if plot_median:
        for b in plot_bins:
            median_ = currd[currd['binned_%s' % to_quartile]==b][metric].median()
            #g.fig.axes[0].axvline(x=median_, color=bin_colors[b])
            ax.axvline(x=median_, color=bin_colors[b]) 
    # g.fig.axes[0].axvline(x=0, color='k')

    return ax #g


def plot_quartile_dists_by_area(bcorrs, bin_labels, bin_colors,
                        metric='signal_cc', to_quartile='cortical_distance', 
                        extrema_only=True, plot_median=True,
                        visual_areas=['V1', 'Lm', 'Li']): 
    '''Plot distns of 1st and last quartiles for metric X'''
    pplot.set_plot_params()

    # Plot metric X for top and bottom quartiles
    if extrema_only:
        plot_bins = [bin_labels[0], bin_labels[-1] ] # [0, 1, 2, 3]
    else:
        plot_bins = bin_labels
    currd = bcorrs[bcorrs['binned_%s' % to_quartile].isin(plot_bins)]    
    g = sns.FacetGrid(data=currd, col='visual_area', col_order=visual_areas, 
                      height=2., aspect=1.3,
                      hue='binned_%s' % to_quartile, palette=bin_colors) 
    g.map(sns.histplot, metric, stat='probability',
                      common_norm=False, common_bins=False)
    g.fig.axes[-1].legend(bbox_to_anchor=(1,1), loc='upper left',
                      title=to_quartile, fontsize=6)
    if plot_median:
        # Add vertical lines
        for (va, b), c_ in currd.groupby(['visual_area', 'binned_%s' % to_quartile]):
            median_ = c_[c_['binned_%s' % to_quartile]==b][metric].median()
            ai = visual_areas.index(va)
            g.fig.axes[ai].axvline(x=median_, color=bin_colors[b])
    pl.subplots_adjust(left=0.1, bottom=0.2, right=0.85, top=0.75)
    sns.despine(trim=True, offset=2)

    return g


def plot_y_by_binned_x(means_, bin_labels, connect_fovs=False,cmap='viridis',
                       metric='signal_cc', to_quartile='cortical_distance',
                       size=3, visual_areas=['V1', 'Lm', 'Li']):
    '''Bar plot (medians) showing individual FOV as dots'''
    pplot.set_plot_params()

    bw_bin_colors = dict((k, [0.7]*3) for k in bin_labels)
    g = sns.FacetGrid(data=means_, col='visual_area', col_order=visual_areas,
                      hue='binned_%s' % to_quartile, height=3, aspect=0.7)
    g.map(sns.stripplot, 'binned_%s' % to_quartile, metric, palette=cmap, size=size)
    g.map(sns.barplot, 'binned_%s' % to_quartile, metric, palette=bw_bin_colors,
             ci=None, estimator=np.median)
    if connect_fovs:
        for va, fc in means_.groupby(['visual_area']):
            ai = visual_areas.index(va)
            ax = g.fig.axes[ai]
            sns.lineplot(x='binned_%s' % to_quartile, y=metric, data=fc,
                         lw=0.5, hue='datakey', ax=ax) #, linecolor='k')
            ax.legend_.remove()
    pl.subplots_adjust(top=0.8, left=0.1, right=0.95, bottom=0.2)
    
    return g



def plot_fit_distance_curves(bcorrs, finalres, to_quartile='cortical_distance', 
                            metric='signal_cc', use_best_r2=False, fit_sites=True,
                            lw=1, scatter_params=False, x_pos=-50, y_pos=-0.4,
                            markersize=10, param_size=5,
                            elinewidth=1, capsize=2, ylim=None, xlim=None,
                            ylabel='corr. coef.', xlabel='cortical dist. (um)',
                            visual_areas=['V1', 'Lm', 'Li'],
                            area_colors=None):
    '''
    Plots 3 subplots: for each area, plot data points + fit line.
    '''
    pplot.set_plot_params()

    x_var = 'binned_%s' % to_quartile
    cnt_grouper = [x_var, 'datakey'] if fit_sites else [x_var]
    # plot params
#     scatter_params=False
#     lw=1
#     markersize=10
#     param_size=5
#     x_pos = -50
#     y_pos=-0.4
#     elinewidth=1
#     capsize=2
#     xlabel = 'corr. coef.'
#     ylabel = 'cortical dist. (um)'
    # ---------------------------
    cols=['init', 'tau', 'constant', 'R2']
    fig, axn = pl.subplots(1, 3, figsize=(6,3), sharex=True, sharey=True, dpi=150)

    for va, cc_ in bcorrs.groupby('visual_area'):
        ai=visual_areas.index(va)
        ax=axn[ai]
        if len(cc_.dropna())==0:
            ax.set_title('%s (no fit)' % va)
            ax.set_box_aspect(1)
            continue
        # plot data
        data = cc_.groupby(cnt_grouper).median()
        xdata = data.sort_values(by=to_quartile)[to_quartile].values
        ydata = data.sort_values(by=to_quartile)[metric].values
        sns.scatterplot(x=to_quartile, y=metric, data=data, ax=ax,
                        s=markersize, color='k', marker='.', edgecolor=None)
                        #jitter=False)
        # plot fits
        if use_best_r2:
            pars_ = finalres.loc[finalres[finalres.visual_area==va]\
                                          ['R2'].idxmax(),cols]
        else:
            pars_ = finalres[finalres.visual_area==va][cols].median()
        
        init, tau, const, r2 = pars_.init, pars_.tau, pars_.constant, pars_.R2
        fit_y = func_halflife(xdata, init, tau, const)
        ax.plot(xdata, fit_y, area_colors[va], lw=lw) # label='fitted line')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        n_fovs = len(cc_['datakey'].unique())
        ax.set_box_aspect(1)
 
        pars_m = finalres[finalres.visual_area==va][cols].dropna().median()
        init_med, tau_med = pars_m.init, pars_m.tau
        pars_sd = finalres[finalres.visual_area==va][cols].dropna().std()
        init_sd, tau_sd = pars_sd.init, pars_sd.tau
        res_str = 'tau=%.2f+/-%.2f\ncc0=%.2f+/-%.2f' %  (tau_med, tau_sd, init_med, init_sd)

        #ax.plot(0, 0, color=None, lw=0, label=res_str)
        #ax.legend(bbox_to_anchor=(x_pos, 1), loc='upper left', frameon=False, fontsize=6)
        
        ax.set_title("%s (n=%i sites)\n%s" % (va,  n_fovs, res_str), loc='left', fontsize=6)
        ax.set_box_aspect(1)

        # param distns
        paramdf = finalres[finalres.visual_area==va]
        if scatter_params:
            ax.scatter(x=paramdf['tau'], y=[ymin]*len(paramdf), 
                       color=area_colors[va], s=param_size)
            ax.scatter(x=[x_offset]*len(paramdf), y=paramdf['init'], 
                       color=area_colors[va], s=param_size)
        else: # just plot med. and CI
            for par in ['tau', 'init']:
                if len(paramdf[par].dropna())==0:
                    continue
                med = paramdf[par].median()
                xpos = med if par=='tau' else x_pos
                ypos = y_pos if par=='tau' else med
                ci_lo, ci_hi = hutils.get_empirical_ci(paramdf[par].values)
                lo_ = abs(med-ci_lo)
                hi_ = abs(ci_hi-med)
                err_ = np.array([[lo_,], [hi_,]])
                xerr = err_ if par=='tau' else None
                yerr = None if par=='tau' else err_
                p_marker ='|' if par=='tau' else'_'
                ax.errorbar(xpos, ypos, yerr=yerr, xerr=xerr,
                           elinewidth=elinewidth, capsize=capsize, marker=p_marker,
                           ecolor=area_colors[va], mec=area_colors[va])
    if ylim is not None:
        for ax in axn:
            ax.set_ylim(ylim)
    if xlim is not None:
        for ax in axn:
            ax.set_xlim(xli)

    sns.despine(trim=True)
    pl.subplots_adjust(left=0.1, right=0.8, bottom=0.25, wspace=0.2, top=0.7)

    return fig




def heatmap_tuning_v_distance(df_, x_bins, y_bins, ax=None,
                    x_var='cortical_distance', y_var='area_overlap', 
                    hue_var='pearsons', hue_norm=(-1, 1), 
                    cmap=None, cbar=False, cbar_ax=[0.87, 0.3, 0.01, 0.3] ):
    #x_var = 'cortical_distance'
    #y_var = 'area_overlap'
    #hue_var = 'pearsons'
    #hue_min, hue_max = (-.8, 0.8)
    if cmap is None:
        cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True)
    #'coolwarm'
    #x_bins = ctx_bins.copy()
    #y_bins = perc_bins.copy()
    
    # x_var_name = 'binned_%s' % x_var
    # y_var_name = 'binned_%s' % y_var
    x_var_name = '%s_label' % x_var
    y_var_name = '%s_label' % y_var

    hue_min, hue_max = hue_norm
    # 
    if ax is None:
        fig, ax = pl.subplots(figsize=(6, 5), dpi=150)
        fig.patch.set_alpha(1)
    means_ = df_.groupby([y_var_name, x_var_name])\
               .mean().reset_index()
    hmat = means_.pivot(y_var_name, x_var_name, hue_var)
    sns.heatmap(hmat, cmap=cmap, ax=ax, vmin=hue_min, vmax=hue_max,
                cbar=cbar, cbar_ax=cbar_ax, cbar_kws={'label': hue_var}) 
    ax.set_box_aspect(0.75)
    #, center=0)
    ax.set_yticks(np.arange(0, len(y_bins)))
    ax.set_yticklabels([round(i, 2) if i in y_bins[0::2] else '' for i in y_bins])

    ax.set_xticks(np.arange(0, len(x_bins)))
    ax.set_xticklabels([int(i) if i in x_bins[0::3] else '' for i in x_bins])
    ax.invert_yaxis()
   
    ax.tick_params(which='both', axis='both', size=0)


    # make frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)

 
    return ax

