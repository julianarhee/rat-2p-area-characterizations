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

import _pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import matplotlib as mpl
import scipy.stats as spstats



def get_x_curves_at_best_y(df, x='morphlevel', y='size', normalize=False):
    '''Get <size> tuning curves at best <morphlevel>, for ex.'''
    best_y = float(df[df['response']==df.groupby([x])['response']\
                .max().max()][y])
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
    return pd.Series(mt, name=df_['cell'].unique()[0])


def get_lum_corr(rd):
    '''Get size tuning curve (at best morph) and 
    luminance tuning curve (same as size-tuning but for morphlevel=-1).
    Calculate correlation coefficient between size- and luminance-tuning curves.
    '''
    lumr = get_x_curves_at_best_y(rd[rd.morphlevel==-1], 
                              x='size', y='morphlevel', normalize=False)
    sizr = get_x_curves_at_best_y(rd[rd.morphlevel!=-1], 
                              x='size', y='morphlevel', normalize=False)
    r_, p_ = spstats.pearsonr(sizr['response'].values, lumr['response'].values)
    
    #pd.Series(mt, name=df_['cell'].unique()[0])
    df = pd.DataFrame({'lum_cc': r_, 'lum_p': p_}, 
                        index=[rd['cell'].unique()[0]])

    return df


# wrappers

