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
    return df_
        

def get_x_curves_at_given_size(df, x='morphlevel', y='size', 
                        val_y=None, normalize=False):
    df_ = df[df[y]==val_y]
    #df_[y] = best_y
    if normalize:
        max_d = float(df_['response'].max())
        df_['response'] = df_['response']/max_d
    return df_


def assign_morph_ix(df, at_best_other=True):
    if at_best_other:
        df_ = get_x_curves_at_best_y(df, x='morphlevel', y='size', 
                                    normalize=False)
    else:
        df_ = df.copy()
    mt = morph_tuning_index(df_['response'].values)
    return pd.Series(mt, name=df_['cell'].unique()[0])

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


def assign_size_tolerance(df, at_best_other=True):
    if at_best_other:
        df_ = get_x_curves_at_best_y(df, x='size', y='morphlevel', 
                                normalize=False)
    else:
        df_ = df.copy()
    mt = size_tolerance(df_['response'].values)
    
    return pd.Series(mt, name=df_['cell'].unique()[0])

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


def assign_sparseness(df, name='cell'):
    mt = sparseness(df['response'].values)
    return pd.Series(mt, name=df[name].unique()[0])


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

def assign_lum_ix(df, at_best_other=True):
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
        name = df_.name
    else:
        name = df_['cell'].unique()[0]
    mt_ = pd.Series(mt, name=name)
    
    return mt_ #pd.Series(mt, name=df_['cell'].unique()[0])


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


# plotting
def stripplot_metric_by_area(plotdf, metric='morph_sel', markersize=1,
                area_colors=None, posthoc='fdr_bh', 
                y_loc=1.01, offset=0.01, ylim=(0, 1.03), aspect=4,
                sig_fontsize=6, sig_lw=0.25, errwidth=0.5, scale=1, 
                jitter=True, return_stats=False, plot_means=True,
                mean_style='point', mean_type='median',
                visual_areas=['V1', 'Lm', 'Li'], fig=None, ax=None):

    if mean_type=='median':
        estimator = np.median
    else:
        estimator = np.mean

    pplot.set_plot_params()
    print(ylim)
    if ax is None:
        fig, ax = pl.subplots( figsize=(2,2), dpi=150)

    #for ai, metric in enumerate(plot_params):
    sns.stripplot(x='visual_area', y=metric, data=plotdf, ax=ax,
                hue='visual_area', palette=area_colors, order=visual_areas, 
                size=markersize, zorder=-10000, jitter=jitter)
    if plot_means:
        if mean_style=='point':
            sns.pointplot(x='visual_area', y=metric, data=plotdf, ax=ax,
                        color='k', order=visual_areas, scale=scale,
                        hue='visual_area', estimator=estimator,
                        markers='_', errwidth=errwidth, zorder=10000, ci='sd')
        else:
            sns.barplot(x='visual_area', y=metric, data=plotdf, ax=ax,
                   order=visual_areas, color=[0.8]*3, ecolor='w', ci=None,
                   zorder=-1000000, estimator=estimator)

    sts = pg.pairwise_ttests(data=plotdf, dv=metric, between='visual_area', 
                  parametric=False, padjust=posthoc, effsize='eta-square')
    pplot.annotate_multicomp_by_area(ax, sts, y_loc=y_loc, offset=offset, 
                                         fontsize=sig_fontsize, lw=sig_lw)
    ax.legend_.remove()
    ax.set_ylim(ylim)
    sns.despine(bottom=True, trim=True, ax=ax)
    ax.tick_params(which='both', axis='x', size=0)
    ax.set_xlabel('')
    pl.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.8)
    ax.set_aspect(aspect)

    if return_stats:
        return fig, sts
    else:
        return fig

# CALCULATING
def count_fraction_luminance_preferring(NDATA_all, NDATA_im):
    cnts_all= aggr.count_n_cells(NDATA_all, name='n_cells').reset_index(drop=True)
    cnts_im = aggr.count_n_cells(NDATA_im, name='n_cells').reset_index(drop=True)
    cnts_all['stimuli'] = 'all'
    cnts_im['stimuli'] ='images'
    assert cnts_all.shape[0]==cnts_im.shape[0]
    cnts = pd.concat([cnts_all, cnts_im], axis=0, ignore_index=True)

    for va, g in cnts.groupby('visual_area'):
        dk_lut = dict((k, i) for i, k in enumerate(sorted(g['datakey'].unique())))
        cnts.loc[g.index, 'site_num'] = [dk_lut[k] for k in g['datakey'].values]

    c_=[]
    for (va, dk), g in cnts.groupby('visual_area'):
        all_c = g[g.stimuli=='all']['n_cells']
        im_c = g[g.stimuli=='images']['n_cells']
        curr_ = g[['visual_area', 'datakey', 'site_num']]\
                    .drop_duplicates().copy().sort_values(by=['datakey', 'site_num'])
        curr_['n_all'] = all_c.values
        curr_['n_images'] = im_c.values
        curr_['pref_object'] = curr_['n_images']/curr_['n_all']
        curr_['n_luminance'] = curr_['n_all'] -  curr_['n_images']
        curr_['pref_object'] = curr_['n_images']/curr_['n_all']
        curr_['pref_luminance'] = curr_['n_luminance']/curr_['n_all']
        c_.append(curr_)
    cnt_each = pd.concat(c_, axis=0, ignore_index=True)


    lum_cnts = cnts[cnts.stimuli=='all']['n_cells'].values - cnts[cnts.stimuli=='images']['n_cells'].values
    sh_copy = cnts[cnts.stimuli=='all'].copy().reset_index(drop=True)
    sh_copy['stimuli'] = 'luminance'
    sh_copy['n_cells'] = lum_cnts
    totals = pd.concat([cnts, sh_copy], axis=0, ignore_index=True)
        
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
                    .apply(assign_morph_ix, at_best_other=True)\
                    .rename(columns={0:'morph_sel'})
    # Calculate size tolerance (@ best morph)
    size_tols = rdf[rdf['morphlevel']!=-1].groupby(['cell'])\
                    .apply(assign_size_tolerance, at_best_other=True)\
                    .rename(columns={0:'size_tol'})
    # calculate sparseness, all morph images
    sparse_morphs = rdf[rdf['morphlevel']!=-1][['cell', 'response']]\
                    .groupby(['cell']).apply(assign_sparseness)\
                    .rename(columns={0:'sparseness_morphs'})
    # calculate sparseness on anchors only 
    sparse_anchors = rdf[rdf['morphlevel'].isin([0, 106])][['cell', 'response']]\
                    .groupby(['cell']).apply(assign_sparseness)\
                    .rename(columns={0:'sparseness_anchors'})
    # calculate sparseness on anchors only 
    sparse_all = rdf[['cell', 'response']]\
                    .groupby(['cell']).apply(assign_sparseness)\
                    .rename(columns={0:'sparseness_total'})
    # Calculate how selective to SIZE (@ best morph)
    size_sel = rdf[rdf['morphlevel']!=-1].groupby(['cell'])\
                    .apply(assign_lum_ix, at_best_other=True)\
                    .rename(columns={0:'size_sel'})
    if -1 in rdf['morphlevel'].unique():
        # How selective is each cell to the LUMINANCE levels
        lum_sel = rdf[rdf['morphlevel']==-1].groupby(['cell'])\
                        .apply(assign_lum_ix, at_best_other=True)\
                        .rename(columns={0:'lum_sel'})
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
    '''
    d_=[]
    for (va, dk), x0 in NDATA.groupby(['visual_area', 'datakey']):
        if dk in exclude:
            continue
        sdf = aggr.get_stimuli(dk, experiment=experiment, match_names=True)
        if -1 not in sdf['morphlevel'].values:
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

