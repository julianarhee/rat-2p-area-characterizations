#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:48:05 2021

@author: julianarhee
"""
import glob
import os
import sys
import optparse
import cv2
import glob
import json
import copy
import traceback
import sys
epsilon = sys.float_info.epsilon

import _pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import matplotlib as mpl
import scipy.stats as spstats

import pingouin as pg
import analyze2p.plotting as pplot
import analyze2p.aggregate_datasets as aggr

def get_x_curves_at_best_y(df, x='morphlevel', y='size', normalize=False):
    '''Get <size> tuning curves at best <morphlevel>, for ex.'''
    best_y = float(df[df['response']==df.groupby([x])['response']\
                .max().max()][y].values[0])
    df_ = df[df[y]==best_y]
    #df_[y] = best_y
    if normalize:
        max_d = float(df_['response'].max())
        df_['response'] = df_['response']/max_d

    df_ = df_.rename(columns={y: 'best_%s' % y})
    # df_['best_%s' % y] = best_y

    return df_
        

def get_x_curves_at_given_size(df, x='morphlevel', y='size', 
                        val_y=None, normalize=False):
    df_ = df[df[y]==val_y]
    #df_[y] = best_y
    if normalize:
        max_d = float(df_['response'].max())
        df_['response'] = df_['response']/max_d
    return df_


def assign_morph_ix(df, at_best_other=True, name='morph_sel'):
    if at_best_other:
        df_ = get_x_curves_at_best_y(df, x='morphlevel', y='size', 
                                    normalize=False)
    else:
        df_ = df.copy()
    mt = morph_tuning_index(df_['response'].values)

    morph_sel = pd.DataFrame({name: mt, 
                              'best_size': float(df_['best_size'].unique())}, 
                            index=[int(df['cell'].unique())])

    return morph_sel #pd.Series(mt, name=df_['cell'].unique()[0])

def morph_tuning_index(responses):
    '''
    MT = [n - (sum(Ri)/Rmax)]/(n - 1), from: Zoccolan et al, 2007.
    0: no shape selectivity, 1: maximal shape selectivity
    '''
    if min(responses)<0:
        responses =  responses - min(responses)

    n = float(len(responses))
    Rmax = max(responses)
    mt = (n - (sum(responses)/Rmax)) / (n-1)
    return mt


def assign_size_tolerance(df, at_best_other=True, name='size_tol'):
    if at_best_other:
        df_ = get_x_curves_at_best_y(df, x='size', y='morphlevel', 
                                normalize=False)
    else:
        df_ = df.copy()
    mt = size_tolerance(df_['response'].values)
   
    size_tol = pd.DataFrame({name: mt, 
                             'best_morphlevel': float(df_['best_morphlevel'].unique())}, 
                            index=[int(df['cell'].unique())])

     
    return size_tol #pd.Series(mt, name=df_['cell'].unique()[0])

def size_tolerance(responses):
    '''
    ST = mean( Rtest / max(Rtest) ), where mean is taken over all sizes.
    0: no size tolerance, 1: perfect size tolerance.
    from: Zoccolan et al, 2007
    '''
    if min(responses)<0:
        responses =  responses - min(responses)

    normed_tuning = responses/float(max(responses))
    ST = np.mean(normed_tuning[normed_tuning<1.])

    return ST


def assign_sparseness(df, unit='cell', name='sparseness'):
    mt = sparseness(df['response'].values)

    if unit=='cell':
        ix_name = int(df[unit].unique())
    else:
        ix_name = df[unit].unique()
    sparse_val = pd.DataFrame({name: mt}, 
                            index=[ix_name])


    return sparse_val # pd.Series(mt, name=df[name].unique()[0])


def sparseness(responses):
    '''
    num = 1 - [ (sum(Ri/n)**2) / sum( ((Ri**2)/n) ) ] 
    denom = [1 - (1/n)]
    from:  Zoccolan et al, 2007; Rolls & Tovee, 1995; Vinje and Gallant, 2000; Olshausen and Field, 2004.
    1 = most selective, 0 = not selective
    or, 1 = v few cells respond to any given image
    '''
    if min(responses)<0:
        responses =  responses - min(responses)

    n = float(len(responses))
    num = 1. - ((sum(responses/n)**2)/sum((responses**2)/n))
    denom = (1. - (1./n))
    S = num/denom
    
    return S # (1-num) #(num/denom)

def assign_lum_ix(df, at_best_other=True, name='lum_sel'):
    '''
    Get SIZE tuning curve at best MORPH level. 
    Only include morphlevel=-1 to get "luminance" selectivity
    Include only morphlevel!=-1 to get size selectivity at best morph.
    1=vv selective for luminance (or size). 

    '''
    if at_best_other:
        # Get size tuning curve at the best morph
        df_ = get_x_curves_at_best_y(df, x='size', y='morphlevel', 
                    normalize=False)
    else:
        df_ = df.copy()
    mt = morph_tuning_index(df_['response'].values)

    if isinstance(df_, pd.Series):
        roi = df_.name
    else:
        roi = df_['cell'].unique()[0]
    #mt_ = pd.Series(mt, name=name)

    sel_df = pd.DataFrame({name: mt}, 
                            index=[int(roi)]) 
    
    return sel_df #mt_ #pd.Series(mt, name=df_['cell'].unique()[0])


def get_lum_corr(rd):
    '''Get size tuning curve (at best morph) and 
    luminance tuning curve (i.e., size-tuning curve @ morphlevel=-1).
    Calculate correlation coefficient between size- and luminance-tuning curves.
    '''
    lumr = get_x_curves_at_best_y(rd[rd.morphlevel==-1], 
                              x='size', y='morphlevel', normalize=False)
    sizr = get_x_curves_at_best_y(rd[rd.morphlevel!=-1], 
                              x='size', y='morphlevel', normalize=False)
    r_, p_ = spstats.pearsonr(sizr['response'].values, lumr['response'].values)
    
    #pd.Series(mt, name=df_['cell'].unique()[0])
    df = pd.Series({'lum_size_cc': r_, 'lum_size_pval': p_})

    return df

def get_object_tuning_curves(rdf, sort_best_size=True, normalize=True, return_stacked=False):
    '''
    Given trial-avg responses to all conditions (morph, size),
    calculate tuning curves.

    Returns:
    
    morph_mat: pd.DataFrame()
        Default:  Morph tuning curve at best size. 
                  Each col is tuning curve for a cell. Rows are morph levels.
        return_stacked:  Stacked dataframe, cols are cell, morphlevel, response, best_size.
    
    size_mat: pd.DataFrame()
        Size tuning curves at best morphlevel. Each col is size-tuning curve.
        if return_stacked:  COls are cell, size, response, best_morphlevel

    Args:

    rdf: pd.DataFrame()
        Trial-averaged responses to each condition (long-form).
        Columns: cell, config, response, size, morphlevel

    sort_best_size: bool, default True
        Set to sort size tuning curves relative to best size.

    normalize: bool, default True
        Normalize so max value is 1.

    return_stacked: bool
        Don't unstack into table of values. Return w columns:  cell, morphlevel, responses.

    '''
    # Morph curves at best size: 
    morph_curves = rdf.groupby(['cell']).apply(get_x_curves_at_best_y, 
                    x='morphlevel', y='size', normalize=normalize).reset_index(drop=True)
    # Size curves
    size_curves = rdf.groupby(['cell']).apply(get_x_curves_at_best_y, 
                                               x='size', y='morphlevel', normalize=normalize)\
                     .reset_index(drop=True)

    if return_stacked:
        return morph_curves, size_curves

    # Un-stack so each column is a cell, rows are morphlevels
    morph_mat = morph_curves[['cell', 'response', 'morphlevel']]\
                    .pivot(columns='cell', index='morphlevel') 
    morph_mat.columns = morph_mat.columns.droplevel()
    morph_mat = morph_mat.sort_index(ascending=True)

    # Un-stack: columns are cells, rows are size levels (size-tuning at best morph)
    size_mat0 = size_curves[['cell', 'response', 'size']].pivot(columns='cell', index='size')
    size_mat0.columns = size_mat0.columns.droplevel()    

    if sort_best_size:
        xx = size_mat0.copy()
        xx.values.sort(axis=0) #[::-1]
        size_mat = xx[::-1]
        size_mat.index = np.linspace(1, size_mat.shape[0], size_mat.shape[0])
    else:
        size_mat = size_mat0.copy()
        size_mat = size_mat.sort_index(ascending=True)

    return morph_mat, size_mat


# plotting
def plot_overlaid_tuning_curves(morph_mat, rank_order=False, ax=None,
                               rois_plot=None, roi_styles=None, roi_colors=None,
                               roi_labels=None, lw=0.5, lc='gray', roi_lw=2):
    morph_labels = sorted(morph_mat.index.tolist())
    if ax is None:
        fig, ax = pl.subplots(figsize=(4,4), dpi=100)
    if rank_order:
        xx = morph_mat.copy()
        xx.values.sort(axis=0)
        mm = xx[::-1]
    else:
        mm = morph_mat.copy()
    ax.plot(mm.values, color=lc, alpha=1, lw=lw)
    
    if rois_plot is not None:
        if roi_labels is None:
            roi_labels = rois_plot
        for ls, col, rid, rlabel in zip(roi_styles, roi_colors, rois_plot, roi_labels):
            ax.plot(mm[rid].values, color=col, lw=roi_lw, linestyle=ls, label=rlabel)
        ax.legend(bbox_to_anchor=(0.5, 1.2), loc='lower center', fontsize=6)

    # xtick_ixs = np.linspace(0, len(morph_labels)-2, 3, endpoint=True)
    xticks = np.arange(0, len(morph_labels)) # if rank_order else morph_labels
    xtick_labels = np.linspace(1, len(morph_labels), len(morph_labels))\
                        if rank_order else morph_labels
    ax.set_xticks(xticks)
    # xticks = xtick_ixs+1 if rank_order else [0, 0.5, 1]
    ax.set_xticklabels(xticks)
    ax.set_ylim([0, 1.01])
    return ax


# plotting
# CALCULATING
def count_fraction_luminance_preferring(NDATA_all, NDATA_im):
#    cnts_all= aggr.count_n_cells(NDATA_all, name='n_cells') #.reset_index(drop=True)
#    cnts_im = aggr.count_n_cells(NDATA_im, name='n_cells') #.reset_index(drop=True)
#    cnts_all['stimuli'] = 'all'
#    cnts_im['stimuli'] ='images'
    #assert cnts_all.shape[0]==cnts_im.shape[0]
    #cnts = pd.concat([cnts_all, cnts_im], axis=0, ignore_index=True)
    cnts_all= aggr.count_n_cells(NDATA_all, name='n_all', reset_index=False)
    cnts_im = aggr.count_n_cells(NDATA_im, name='n_images', reset_index=False)
    cnts = pd.merge(cnts_all, cnts_im, how='outer', left_index=True, right_index=True)
    cnts = cnts.fillna(value=0)
    cnts = cnts.reset_index()

    for va, g in cnts.groupby('visual_area'):
        dk_lut = dict((k, i) for i, k in enumerate(sorted(g['datakey'].unique())))
        cnts.loc[g.index, 'site_num'] = [dk_lut[k] for k in g['datakey'].values]

    c_=[]
    for (va, dk), curr_ in cnts.groupby('visual_area'):
        #all_c = g[g.stimuli=='all']['n_cells']
        #im_c = g[g.stimuli=='images']['n_cells']
        #curr_ = g[['visual_area', 'datakey', 'site_num']]\
        #            .drop_duplicates().copy().sort_values(by=['datakey', 'site_num'])
        #curr_['n_all'] = all_c.values
        #curr_['n_images'] = im_c.values
        curr_['pref_object'] = curr_['n_images']/curr_['n_all']
        curr_['n_luminance'] = curr_['n_all'] -  curr_['n_images']
        curr_['pref_object'] = curr_['n_images']/curr_['n_all']
        curr_['pref_luminance'] = curr_['n_luminance']/curr_['n_all']
        c_.append(curr_)
    cnt_each = pd.concat(c_, axis=0, ignore_index=True)


#    lum_cnts = cnts['n_all'].values - cnts['n_images'].values
#    sh_copy = cnts.copy().reset_index(drop=True)
#    sh_copy['stimuli'] = 'luminance'
#    sh_copy['n_cells'] = lum_cnts
#    totals = pd.concat([cnts, sh_copy], axis=0, ignore_index=True)
#        
    d_=[]
    for va, vg in cnt_each.groupby('visual_area'):
        df1 = vg[['visual_area', 'datakey', 'site_num', 'n_luminance']].copy()\
                .rename(columns={'n_luminance': 'n_cells'})
        df1['stimuli'] = 'luminance'    
        df2 = vg[['visual_area', 'datakey', 'site_num', 'n_images']].copy()\
                .rename(columns={'n_images': 'n_cells'})
        df2['stimuli'] = 'images'
        df_ = pd.concat([df1, df2], axis=0, ignore_index=True)
        d_.append(df_)
    totals = pd.concat(d_, axis=0)


    return totals, cnt_each



def exclude_lum_is_best(ndf, sdf):
    '''
    From responsive neuraldata, exclude cells whose best config
    is actually the luminance control stimulus. Returns dataframe.
    '''

    lumcontrols = sdf[sdf.morphlevel==-1].index.tolist()

    if len(lumcontrols)==0:
        return ndf
    # Get mean resp to each config
    meanr = ndf[['cell', 'config', 'response']].groupby(['cell', 'config']).mean()
    # Get best config for each cell
    maxr = meanr.loc[meanr.groupby('cell')['response'].idxmax()]

    # Exclude cells if "best config" is luminance
    lum_is_not_max = maxr[~maxr.index.get_level_values(1).isin(lumcontrols)].copy()
    incl_cells = lum_is_not_max.index.get_level_values(0)

    return ndf[ndf['cell'].isin(incl_cells)]

def calculate_metrics(rdf, sdf, iternum=None):
    #rdf = x0.groupby(['cell', 'config']).mean().reset_index().drop('trial', axis=1)
    rdf['size'] = [sdf['size'][c] for c in rdf['config']]
    rdf['morphlevel'] = [sdf['morphlevel'][c] for c in rdf['config']]
    # Calculate morph selectivity (@ best size)
    morph_ixs = rdf[rdf['morphlevel']!=-1].groupby(['cell'])\
                    .apply(assign_morph_ix, at_best_other=True, name='morph_sel')
    morph_ixs.index = morph_ixs.index.droplevel(1)
    # Calculate size tolerance (@ best morph)
    size_tols = rdf[rdf['morphlevel']!=-1].groupby(['cell'])\
                    .apply(assign_size_tolerance, at_best_other=True, name='size_tol')
    size_tols.index = size_tols.index.droplevel(1)

    # calculate sparseness, all morph images
    sparse_morphs = rdf[rdf['morphlevel']!=-1][['cell', 'response']]\
                    .groupby(['cell']).apply(assign_sparseness, unit='cell', name='sparseness_morphs')
    sparse_morphs.index = sparse_morphs.index.droplevel(1)

    # calculate sparseness on anchors only 
    sparse_anchors = rdf[rdf['morphlevel'].isin([0, 106])][['cell', 'response']]\
                    .groupby(['cell']).apply(assign_sparseness, name='sparseness_anchors')
    sparse_anchors.index = sparse_anchors.index.droplevel(1)

    # calculate sparseness on ALL stimuli
    sparse_all = rdf[['cell', 'response']]\
                    .groupby(['cell']).apply(assign_sparseness, name='sparseness_total')
    sparse_all.index = sparse_all.index.droplevel(1)
 
    # Calculate how selective to SIZE (@ best morph)
    size_sel = rdf[rdf['morphlevel']!=-1].groupby(['cell'])\
                    .apply(assign_lum_ix, at_best_other=True, name='size_sel')
    size_sel.index = size_sel.index.droplevel(1)

    if -1 in rdf['morphlevel'].unique():
        # How selective is each cell to the LUMINANCE levels
        lum_sel = rdf[rdf['morphlevel']==-1].groupby(['cell'])\
                        .apply(assign_lum_ix, at_best_other=True, name='lum_sel')
        lum_sel.index = lum_sel.index.droplevel(1)
        # Luminance corrs: corr coef. bw size-tuning at best morph 
        # vs "size"-tuning at lum control (no morph)
        lum_ccs = rdf.groupby(['cell']).apply(get_lum_corr)
        #lum_ccs.index = lum_ccs.index.droplevel()
    else:
        lum_sel = pd.DataFrame({'lum_sel': [None]*len(morph_ixs)},
                                 index=morph_ixs.index)
        lum_ccs = pd.DataFrame({'lum_size_cc': [None]*len(morph_ixs),
                                'lum_size_pval': [None]*len(morph_ixs)}, 
                                 index=morph_ixs.index)
    # Combine 
    ixs_ = pd.concat([morph_ixs, size_tols, sparse_all, sparse_morphs, sparse_anchors,
                      size_sel, lum_sel, lum_ccs], axis=1)
    ixs_['cell'] = ixs_.index
    if iternum is not None:
        ixs_['iteration'] = iternum
    return ixs_.reset_index(drop=True)



def normalize_curve_by_lum_response(rdf0, sdf):
    '''
    Normalize average response curve for neuron by responses to size-matched lum.
    
    rdf: (pd.DataFrame), columns 'cell' and 'config'
        Mean response to each stim condition (long-form)
    '''
    rdf = rdf0.copy().reset_index(drop=True)
    epsilon = sys.float_info.epsilon
    for c, g in rdf.groupby('cell'):
#        if g['response'].min()<0:
#            g['response'] -= g['response'].min()
#            #['response'] += epsilon
        for sz, s_df in sdf.groupby('size'):
            curr_sz_cfgs = s_df.index.tolist() # all the configs at this size
            lum = s_df[s_df['morphlevel']==-1].index.tolist()    # which lum is it
            lum_r = float(g[g.config.isin(lum)]['response'])     # response at lum size X
            resp_at_size = g[g.config.isin(curr_sz_cfgs)].copy() # should be 10 morphs per size
            new_vs = (resp_at_size['response']-lum_r)/lum_r
            if new_vs.isin([np.nan, np.inf, -np.inf]).all() or all(new_vs<=0):
                rdf.loc[resp_at_size.index, 'response'] = 0
            else:
                rdf.loc[resp_at_size.index, 'response'] = new_vs

    return rdf

def subtract_min(rdf0):
    #rdf_ = rdf0.copy()
    rdf_minsub = rdf0.copy()
    for ci, rd_ in rdf_minsub.groupby('cell'):
        if rd_['response'].min()<0:
            #print(ci, x0['response'].min())
            offset_c = rd_['response'] - rd_['response'].min() #+ epsilon
            rdf_minsub.loc[rd_.index, 'response'] = offset_c
    return rdf_minsub

def half_rectify(rdf0):
    rdf_rectify = rdf0.copy()
    for ci, rd_ in rdf_rectify.groupby('cell'):
        if rd_['response'].min()<0:
            #print(ci, x0['response'].min())
            offset_c = rd_['response'].copy()
            offset_c[offset_c<0] = 0 #+ epsilon
            rdf_rectify.loc[rd_.index, 'response'] = offset_c.values
    return rdf_rectify

def correct_offset(rdf0, offset='none'):
    curr_rdf = None
    rdf = rdf0.copy()

    if offset=='none':
        return rdf0.copy()
    elif offset=='minsub':
        # min-subtract
        curr_rdf = subtract_min(rdf)
    elif offset=='rectify':
        curr_rdf = half_rectify(rdf)
    else:
        print("Invalid type: %s" % offset)
        return None

    return curr_rdf

def correct_luminance(rdf0, sdf, lcorrection='none'):
    curr_rdf=None
    rdf = rdf0.copy()
    if lcorrection=='none':
        return rdf
    elif lcorrection =='normalize':
        curr_rdf = normalize_curve_by_lum_response(rdf, sdf)
    elif lcorrection == 'exclude':
        curr_rdf = exclude_lum_is_best(rdf, sdf)
    else:
        print("Invalid type: %s" % lcorrection)
        return None
 
    return curr_rdf

def aggregate_cell_metrics(NDATA, offset_type='none', lcorrection='none',
                            experiment='blobs', exclude=[]):
    '''Cycle thru all datasets, calculate all the metrics.
    Args.

    offset_type: (str)
        How to deal with negative values for metrics.
        'none': don't do anything, just min-subtract if is neg.
        'minsub': same as none
        'rectify': half-rectify if neg.

    lcorrection: (str)
        'none': don't do anything
        'exclude': exclude cells whose BEST stimulus is luminance.

    '''
    d_=[]
    for (va, dk), x0 in NDATA.groupby(['visual_area', 'datakey']):
        if dk in exclude:
            continue
        sdf = aggr.get_stimuli(dk, experiment=experiment, match_names=True)
        if -1 not in sdf['morphlevel'].values and lcorrection!='none':
            print("    skippping, %s, %s (no lum)" % (va, dk))
            continue
        configs = sdf.index.tolist()
    #     if remove_offset:
    #         x0['response'] = x0['response'] - x0.groupby(['cell'])['response'].transform('min')
        rdf0 = x0.groupby(['cell', 'config']).mean().reset_index().drop('trial', axis=1)
        rdf_offset = correct_offset(rdf0, offset=offset_type)
        rdf = correct_luminance(rdf_offset, sdf, lcorrection=lcorrection)

        ixs_ = calculate_metrics(rdf[rdf.config.isin(configs)], sdf)
        ixs_['visual_area'] = va
        ixs_['datakey'] = dk
        ixs_['n_cells'] = len(x0['cell'].unique())
        d_.append(ixs_.reset_index(drop=True))
    ixdf = pd.concat(d_, axis=0, ignore_index=True)
    ixdf['lum_sel'] = ixdf['lum_sel'].astype(float) #.dtypes
    ixdf['lum_size_cc'] = ixdf['lum_size_cc'].astype(float) #.dtypes
    ixdf['lum_size_pval'] = ixdf['lum_size_pval'].astype(float) #.dtypes

    return ixdf


def aggregate_population_sparseness(NDATA, offset_type='none', lcorrection='none', 
                                    experiment='blobs', exclude=[]):
    '''Calculate pop sparseness, same as lifetime, but over configs''' 
    pop_sparse=None

    p_=[]
    for (va, dk), x0 in NDATA.groupby(['visual_area', 'datakey']):
        # x0['response'] = x0['response'].abs()
        if dk in exclude:
            continue
        rdf0 = x0.groupby(['cell', 'config']).mean().reset_index().drop('trial', axis=1)
        sdf = aggr.get_stimuli(dk, experiment, match_names=True)
        rdf0['size'] = [sdf['size'][c] for c in rdf0['config']]
        rdf0['morphlevel'] = [sdf['morphlevel'][c] for c in rdf0['config']]
        if -1 not in sdf['morphlevel'].unique():
            continue
        rdf_offset = correct_offset(rdf0, offset=offset_type)
        rdf = correct_luminance(rdf_offset, sdf, lcorrection=lcorrection)
        #rdf = correct_offset(rdf0.copy(), offset=offset_type)
        psparse = rdf.groupby('config').apply(assign_sparseness, name='config')\
                        .rename(columns={0:'pop-sparseness'})
        psparse['visual_area'] = va
        psparse['datakey'] = dk
        psparse['n_cells'] = len(rdf['cell'].unique())
        psparse['config'] = psparse.index.tolist()
        p_.append(psparse.reset_index(drop=True))
    pop_sparse = pd.concat(p_, axis=0, ignore_index=True)

    return pop_sparse

