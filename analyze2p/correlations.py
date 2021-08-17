#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:16:05 2021

@author: julianarhee
"""
import os
import glob

import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns

import analyze2p.aggregate_datasets as aggr
import analyze2p.utils as hutils
import analyze2p.plotting as pplot

import scipy.stats as spstats


# do fiting
from scipy.optimize import curve_fit

def func_halflife(t, a, tau, c):
    return a * 2**(-t / tau) + c

def func_tau(t, a, tau, c):
    return a * np.exp(-t / tau) + c

def func_decay(t, a, tau, c):
    return a * np.exp(-t * tau) + c


def fit_decay(xdata, ydata, p0=None, return_inputs=False, normalize_x=True):
    in_x = xdata.copy()
    in_y = ydata.copy()
    if normalize_x:
        in_x_norm = (in_x - in_x[0])/(in_x[-1] - in_x[0])   # normalized
    else:
        in_x_norm = in_x.copy()

    a_f, tau_f, c_f, r2 = None, None, None, None
    # init params
    if p0 is None:
        p0 = init_decay_params(ydata) # (1, 1, 1)
    bounds = ((-1., 0., -np.inf), (1., 3000, np.inf))
    a_0, tau_0, c_0 = p0
    try:
        popt4, pcov4 = curve_fit(func_halflife, in_x_norm, in_y, 
                                p0=(a_0, tau_0, c_0), bounds=bounds)
        if normalize_x:
            a4, tau4, c4 = popt4
            a_f = a4*np.exp(xdata[0]/(xdata[-1] - xdata[0]) / tau4)
            tau_f = (xdata[-1] - xdata[0]) * tau4
            c_f = c4
            #print(a_f, tau_f, c_f)
        else:
            a_f, tau_f, c_f = popt4
        fitv = func_halflife(in_x, a_f, tau_f, c_f)

        # Get residual sum of squares 
        residuals = in_y - fitv
        ss_res = np.nansum(residuals**2)
        ss_tot = np.nansum((in_y - np.nanmean(in_y))**2)
        r2 = 1 - (ss_res / ss_tot)
    except RuntimeError as e:
        print('    no fit')

    if return_inputs:
        return a_f, tau_f, c_f, r2, in_x, in_y
    else:
        return af, tau_f, c_f, r2

def init_decay_params(ydata):
#     c_0 = ydata[-1]
#     tau_0 = 1
#     a_0 = (ydata[-1] - ydata[0])
    c_0 = ydata[-1]
    tau_0 = 1
    a_0 = abs(ydata[0] - ydata[-1])
    p0 = (a_0, tau_0, c_0)
    return p0


def fit_decay_on_binned(cc_, use_binned=False,
                       metric='signal_cc', bin_column='binned_cortical_distance',
                       return_inputs=False):
    '''
    Fit exponential decay (returns half-life).
    use_binned: (bool)
        If True, get median cell pair values for each bin, then fit. 
        Otherwise, sample within bin, but fit on raw points.
    '''
    if use_binned:
        xdata = np.array([xg[bin_column].mean() for xi, xg \
                         in cc_.groupby(bin_column)])
        ydata = np.array([xg[metric].mean() for xi, xg \
                          in cc_.groupby(bin_column)]) 
        normalize_x = True
    else:
        xdata = cc_.sort_values(by=bin_column)[bin_column].values
        ydata = cc_.sort_values(by=bin_column)[metric].values
        normalize_x = False
    # ----------------
    non_nans = np.array([int(i) for i, v in enumerate(xdata) if not np.isnan(v)])
    if len(non_nans)==0:
        print(cc_.head(), xdata)
    xdata = xdata[non_nans]
    ydata = ydata[non_nans]

    p0 = init_decay_params(ydata)
    initv, tau, const, r2, xvals, yvals = fit_decay(xdata, ydata, 
                                            normalize_x=normalize_x,
                                            return_inputs=True, p0=p0)
    res_ = pd.Series({'init': initv, 'tau': tau, 'constant': const, 'R2': r2})
    
    if return_inputs:
        return res_, xvals, yvals
    else:
        return res_
    
def bootstrap_fitdecay(bcorrs, use_binned=False, 
                      metric='signal_cc', bin_column='binned_cortical_distance',
                      n_iterations=500):
    '''
    Cycle thru visual areas, sample w replacement, fit decay func.
    Return df of bootstrapped params.
    '''
    r_=[]
    for va, cc0 in bcorrs.groupby('visual_area'):
        cnts = cc0.groupby(bin_column)['neuron_pair']\
                  .count().reset_index()
        filled_bins = cnts[cnts['neuron_pair']>0][bin_column].values
        cc_ = cc0[cc0[bin_column].isin(filled_bins)].copy()
        n_samples = cc_.groupby(bin_column).count().min().min()
        for n_iter in np.arange(0, n_iterations):
            curr_cc = cc_.groupby(bin_column).sample(n=n_samples,
                                        random_state=n_iter, replace=True)
            res_, xvals, yvals = fit_decay_on_binned(curr_cc, use_binned=use_binned,
                                            metric=metric, 
                                            bin_column=bin_column,
                                            return_inputs=True)
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
# calculate correlations
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


def get_ccdist(neuraldf, return_zscored=False, curr_cfgs=None, curr_cells=None):
    '''
    Calculate pairwise correlation coefs and also add distance. 
    '''
    if curr_cfgs is None:
        curr_cfgs = neuraldf['config'].unique()
    if curr_cells is None:
        curr_cells = neuraldf['cell'].unique()
    # Dont do zscore within-condition
    corrs, zscored = calculate_corrs(neuraldf, do_zscore=False, return_zscored=True,
                                     curr_cells=curr_cells, curr_cfgs=curr_cfgs)
    # Add cortical distances
    wpos = aggr.add_roi_positions(neuraldf.copy())
    roi_pos = wpos[['cell', 'ml_pos', 'ap_pos']].drop_duplicates().copy()
    ccdist = get_pw_cortical_distance(corrs, roi_pos)

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
    cc[['cell_1', 'cell_2']] = cc[['cell_1', 'cell_2']].astype(int)
    cc['neuron_pair'] = ['%i_%i' % (c1, c2) for \
                         c1, c2 in cc[['cell_1', 'cell_2']].values]
    return cc
    
def melt_square_matrix(df, metric_name='value', add_values={}, include_diagonal=False):
    '''Melt square matrix into unique values only'''
    k = 0 if include_diagonal else 1
    df = df.where(np.triu(np.ones(df.shape), k=k).astype(np.bool))

    df = df.stack().reset_index()
    df.columns=['row', 'col', metric_name]

    if len(add_values) > 0:
        for k, v in add_values.items():
            df[k] = [v for _ in np.arange(0, df.shape[0])]

    return df

def get_pw_cortical_distance(cc_, pos_):
    # Get current FOV rfdata and add position info to sigcorrs df
    cc_['cell_1'] = cc_['cell_1'].astype(int)
    cc_['cell_2'] = cc_['cell_2'].astype(int)

    r1 = cc_['cell_1'].unique()
    r2 = cc_['cell_2'].unique()
    crois_ = np.union1d(r1, r2)
    #assert len([r for r in crois_ if r not in pos_.index.tolist()])==0, \
    #    "[%s, %s]: incorrect roi indexing in RFDATA" % (va, dk)
    if 'cell' in pos_.columns:
        pos_.index = pos_['cell'].values
    # Coords of cell1 in pair, in order
    coords1 = np.array(pos_.loc[cc_['cell_1'].values][['ml_pos', 'ap_pos']])
    # Coords of cell2 in pair 
    coords2 = np.array(pos_.loc[cc_['cell_2'].values][['ml_pos', 'ap_pos']])
    # Get dists, in order of appearance
    dists = [np.linalg.norm(c1-c2) for c1, c2 in zip(coords1, coords2)]
    cc_['cortical_distance'] = dists
    
    return cc_





# Data binning

def get_bins(n_bins=4, custom_bins=False, use_quartile=True, cmap='viridis'):
    '''Get generic bins and bin labels to split data up'''
    if custom_bins:
        use_quartile = False
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
                                labels=labels)

    if use_quartile:
        cc_['binned_%s' % to_quartile], bin_edges = pd.qcut(cc_[to_quartile], \
                                        n_bins, labels=labels, retbins=True)
    else:
        cc_['binned_%s' % to_quartile], bin_edges = pd.cut(cc_[to_quartile], \
                                         n_bins,labels=labels, retbins=True)
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

