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

def calculate_corrs(ndf, return_zscored=False, curr_cells=None, curr_cfgs=None):
    if curr_cells is None: 
        curr_cells = ndf['cell'].unique()
    if curr_cfgs is None:
        curr_cfgs = ndf['config'].unique()
    ndf1 = ndf[(ndf.config.isin(curr_cfgs)) & (ndf['cell'].isin(curr_cells))].copy()
    # Reshape dataframe to ntrials x nrois
    trial_means0 = aggr.stacked_neuraldf_to_unstacked(ndf1)
    cfgs_by_trial = trial_means0['config']
    # Zscore trials
    zscored = aggr.zscore_dataframe(trial_means0[curr_cells])
    zscored['config'] = cfgs_by_trial
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
                     n_bins=4, labels=False, bins=None):
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
        labels = custom_bin_labels(bins)
        cc_['binned_%s' % to_quartile] = pd.cut(x=cc_[to_quartile], 
                                bins=bins, 
                                labels=['<100', '100-300', '300-500', '>500'])

    if use_quartile:
        cc_['binned_%s' % to_quartile], bin_edges = pd.qcut(cc_[to_quartile], \
                                        n_bins, labels=labels, retbins=True)
    else:
        cc_['binned_%s' % to_quartile], bin_edges = pd.cut(cc_[to_quartile], \
                                         n_bins,labels=labels, retbins=True)
    return cc_

def plot_quartiles_vs_distance(cc_, to_quartile='cortical_distance',
                              plot_bins=[0, 3], metric='signal_cc',
                              qcolors=None, plot_median=True):
    if qcolors is None:
        nb = len(plot_bins)
        qcolor_list = sns.color_palette('cubehelix', n_colors=nb)
        qcolors = dict((k, v) for k, v in zip(np.arange(0, nb), qcolor_list))
        
    currd = cc_[cc_['binned_%s' % to_quartile].isin(plot_bins)]
    g = sns.displot(data=currd, x=metric,
                hue='binned_%s' % to_quartile, legend=True, palette=qcolors)
    
    if plot_median:
        for b in plot_bins:
            median_ = cc_[cc_['binned_%s' % to_quartile]==b][metric].median()
            g.fig.axes[0].axvline(x=median_, color=qcolors[b])
    # g.fig.axes[0].axvline(x=0, color='k')
    return g

