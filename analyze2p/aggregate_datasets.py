#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:24:03 2020

@author: julianarhee
"""

import os
import sys
import re
import glob
import json
import traceback
import optparse
import copy
import matplotlib as mpl
mpl.use('agg')

import statsmodels as sm
import _pickle as pkl
import numpy as np
import pylab as pl
import seaborn as sns
import scipy.stats as spstats
import pandas as pd
import importlib

import scipy as sp
import itertools

from matplotlib.lines import Line2D
import statsmodels as sm
#import statsmodels.api as sm

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)       

import analyze2p.utils as hutils
import analyze2p.extraction.rois as roiutils
#import analyze2p.gratings.utils as utils
import analyze2p.retinotopy.utils as retutils
import analyze2p.extraction.traces as traceutils

# ###############################################################
# Analysis specific
# ###############################################################
def decode_analysis_id(visual_area=None, prefix='split_pupil', response_type='dff', 
                        responsive_test='ROC', overlap_thr=None, trial_epoch='plushalf', 
                        C_str='tuneC'):
    '''
    Generate identifier string for decoding analysis results.
    '''
    overlap_str = 'noRF' if overlap_thr in [None, 'None'] else 'overlap%.2f' % overlap_thr
    results_id = '%s_%s__%s-%s_%s__%s__%s' \
                    % (prefix, visual_area, response_type, responsive_test, overlap_str, trial_epoch, C_str)

    return results_id

def load_split_pupil_input(animalid, session, fovnum, curr_id='results_id', 
                            traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    '''
    Loads saved dataframe with input data for split-pupil decoding analysis (inputdata_<analysisid>.pkl)
    '''
    curr_inputfiles = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovnum,
                        'combined_blobs_static', 'traces',
                       '%s*' % traceid, 'decoding', 'inputdata_%s.pkl' % curr_id))
    try:
        assert len(curr_inputfiles)==1, "More than 1 input file: %s" % str(curr_inputfiles)
        with open(curr_inputfiles[0], 'rb') as f:
            res = pkl.load(f, encoding='latin1')  
    except UnicodeDecodeError:
        with open(curr_inputfiles[0], 'rb') as f:
            res = pkl.load(f, encoding='latin1')  
    except Exception as e:
        traceback.print_exc()
        return None
    
    return res


# --------------------------------------------------------------------
# Stimuli
# --------------------------------------------------------------------
#def reformat_morph_values(sdf, verbose=False):
#    '''
#    Rounds values for stimulus parameters, checks to make sure true aspect ratio is used.
#    '''
#
#
#    sdf = sdf.sort_index()
#    aspect_ratio=1.75
#    control_ixs = sdf[sdf['morphlevel']==-1].index.tolist()
#    if len(control_ixs)==0: # Old dataset
#        if 17.5 in sdf['size'].values:
#            sizevals = sdf['size'].divide(aspect_ratio).astype(float).round(0)
#            #np.array([round(s/aspect_ratio,0) for s in sdf['size'].values])
#            sdf['size'] = sizevals
#    else:  
#        sizevals = np.array([round(s, 1) for s in sdf['size'].unique() \
#                            if s not in ['None', None] and not np.isnan(s)])
#        sdf.loc[control_ixs, 'size'] = pd.Series(sizevals, index=control_ixs).astype(float)
#        sdf['size'] = sdf['size'].astype(float).round(decimals=1)
#        #[round(s, 1) for s in sdf['size'].values]
#
#    xpos = [x for x in sdf['xpos'].unique() if x is not None]
#    ypos =  [x for x in sdf['ypos'].unique() if x is not None]
#    #assert len(xpos)==1 and len(ypos)==1, "More than 1 pos? x: %s, y: %s" % (str(xpos), str(ypos))
#    if verbose and (len(xpos)>1 or len(ypos)>1):
#        print("warning: More than 1 pos? x: %s, y: %s" % (str(xpos), str(ypos)))
#    sdf.loc[control_ixs, 'xpos'] = xpos[0]
#    sdf.loc[control_ixs, 'ypos'] = ypos[0]
#
#    return sdf
#

def get_stimuli(datakey, experiment, match_names=False,
                    rootdir='/n/coxfs01/2p-data', verbose=False):
    '''
    Get stimulus info for a given experiment and imaging site.
    Returns 'sdf' (dataframe, index='config001', etc. columns=stimulus parameters).
    '''
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    if 'rfs' in experiment and int(session)<20190511:
        experiment_name = 'gratings'
    else:
        experiment_name = experiment
    if verbose:
        print("... getting stimulus info for <%s>: %s" % (experiment, experiment_name))
    try:
        dset_path = glob.glob(os.path.join(rootdir, animalid, session, 
                            'FOV%i_*' % fovn,
                            'combined_%s_static' % experiment_name, \
                            'traces/traces*', 'data_arrays', 'labels.npz'))[0]
        dset = np.load(dset_path, allow_pickle=True)
        sdf = pd.DataFrame(dset['sconfigs'][()]).T
        if 'blobs' in experiment:
            sdf = traceutils.reformat_morph_values(sdf)
        if match_names:
            sdf = match_config_names(sdf.copy())

    except IndexError as e:
        print("(%s) No labels.npz for exp name ~%s~. Found:" \
                        % (datakey, experiment_name))
        tdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn,
                        'combined_%s_static' % experiment_name, \
                        'traces/traces*', 'data_arrays', '*.npz'))
        for t in tdir:
            print('   %s' % t)
        return None
    except Exception as e:
        traceback.print_exc()
        return None
 
    return sdf.sort_index()


def check_sdfs_gratings(dkey_list, experiment='gratings',
                    return_incorrect=False, return_all=False, 
                    verbose=False):
    exclude_=[] 
    #['20190602_JC091_fov1', '20190525_JC084_fov1', '20190522_JC084_fov1']
    # ^Need to re-aggregate these for gratings, exclude for now
    print("Checking gratings configs")
    incorrect=[]
    s_=[]
    for dk in dkey_list:
        if dk in exclude_:
            continue
        sdf = get_stimuli(dk, experiment, match_names=False)
        if 'aspect' in sdf.columns:
            sdf = sdf.drop(columns=['aspect'])
        # Check params
        param_combos = list(itertools.product(
                            sdf['sf'].unique(), 
                            sdf['size'].unique(), 
                            sdf['speed'].unique()))
        nonori_params = sdf[['sf', 'size', 'speed']].drop_duplicates()
        if len(param_combos)!=8 or len(nonori_params)!=len(param_combos):
            if verbose:
                print('    skipping: %s' % dk)
            incorrect.append(dk)
            if not return_all:
                continue
            #continue
            # and sdf.shape[0]!=64:
        sdf['datakey'] = dk
        s_.append(sdf)
    SDF = pd.concat(s_, axis=0)

    if return_incorrect:
        return SDF, incorrect
    else:
        return SDF

def get_master_sdf(experiment='blobs', images_only=False):
    '''
    Get "standard" stimulus info.
    '''
    if experiment=='blobs':
        sdf_master = get_stimuli('20190522_JC084_fov1', experiment, 
                                match_names=False)
        if images_only:
            sdf_master=sdf_master[sdf_master['morphlevel']!=-1].copy()
   
    elif experiment=='gratings':
        sdf_master = get_stimuli('20190522_JC084_fov1', experiment,
                                match_names=False)
        if images_only:
            sdf_master=sdf_master[sdf_master['size']<200].copy()

    return sdf_master


def check_sdfs(stim_datakeys, experiment='blobs', images_only=False, 
                rename=True, return_incorrect=False, return_all=False):

    wrong_configs = {
        'blobs': ['20190314_JC070_fov1', '20190327_JC073_fov1'],
        'gratings': []
    }
    diff_configs = wrong_configs[experiment]

    if experiment=='blobs':
        sdfs, incorrect = check_sdfs_blobs(stim_datakeys, images_only=images_only, 
                            rename=rename,
                            return_incorrect=True,
                            diff_configs=diff_configs)

    elif experiment=='gratings':
        sdfs0, incorrect = check_sdfs_gratings(stim_datakeys, return_incorrect=True,
                            return_all=True)
        if return_all:
            sdfs = sdfs0[~sdfs0.datakey.isin(incorrect)]
        else:
            sdfs = sdfs0.copy()

    if return_incorrect:
        return sdfs, incorrect
    else:
        return sdfs


def check_sdfs_blobs(stim_datakeys, images_only=False,
                rename=True, return_incorrect=False, 
                diff_configs=['20190314_JC070_fov1', '20190327_JC073_fov1'] ):
    '''
    Checks config names and reutrn master dict of all stimconfig dataframes
    Notes: only tested with blobs, and renaming only works with blobs.
    '''
    experiment='blobs'
    sdf_master = get_master_sdf(experiment='blobs', images_only=False)
    n_configs = sdf_master.shape[0]
    #### Check that all datasets have same stim configs
    SDF={}
    renamed_configs={}
    for datakey in stim_datakeys:
        session, animalid, fov_ = datakey.split('_')
        fovnum = int(fov_[3:])
        sdf = get_stimuli(datakey, experiment)
        if len(sdf['xpos'].unique())>1 or len(sdf['ypos'].unique())>1:
            print("*Warning* <%s> More than 1 pos? x: %s, y: %s" \
                    % (datakey, str(sdf['xpos'].unique()), str(sdf['ypos'].unique())))
        key_names = ['morphlevel', 'size']
        updated_keys={}
        for old_ix in sdf.index:
            #try:
            new_ix = sdf_master[(sdf_master[key_names] == sdf.loc[old_ix,  key_names]).all(1)].index[0]
            updated_keys.update({old_ix: new_ix})
        if rename: # and (datakey not in diff_configs): 
            sdf = sdf.rename(index=updated_keys)
        # Save renamed key 
        renamed_configs[datakey] = updated_keys
        if images_only:
            SDF[datakey] = sdf[sdf['morphlevel']!=-1].copy()
        else:
            SDF[datakey] = sdf
    ignore_params = ['xpos', 'ypos', 'position', 'color']
    #if experiment != 'blobs':
    #    ignore_params.extend(['size'])
    compare_params = [p for p in sdf_master.columns if p not in ignore_params]
    different_configs = renamed_configs.keys()
    assert all([all(sdf_master[compare_params]==d[compare_params]) \
            for k, d in SDF.items() \
            if k not in different_configs]), "Incorrect stimuli..."
    if return_incorrect:
        return SDF, renamed_configs
    else:
        return SDF

def match_config_names(sdf, experiment='blobs'):
    sdf_master = get_master_sdf(experiment=experiment, images_only=False)
    key_names = ['morphlevel', 'size']
    updated_keys={}
    for old_ix in sdf.index:
        # Find corresponding index in "master"
        new_ix = sdf_master[(sdf_master[key_names]==sdf.loc[old_ix,  key_names])\
                            .all(1)].index[0]
        updated_keys.update({old_ix: new_ix})
    # rename
    sdf0 = sdf.rename(index=updated_keys)
    # Save renamed key 

    return sdf0

def select_stimulus_configs(datakey, experiment, select_stimuli=None):
    '''
    Get config list for datakey and experiment.

    select_stimuli: (str or None)
        - None: Returns all configs in sdf.
        - fullfield: Return only full-field stimuli 
                   This will be FF (gratings) or morphlevel=-1 (blobs)
        - images: Return non-FF stimuli.
                This will be apertured (gratings) or images (blobs)

    Returns list ['config001', 'config002', etc.]
    '''
    curr_cfgs=None
    sdf = get_stimuli(datakey, experiment=experiment)
    if sdf is None:
        return None
    if experiment not in ['gratings', 'blobs']:
        curr_cfgs = sdf.index.tolist()
        return curr_cfgs

    if select_stimuli is not None:
        if experiment=='gratings':
            curr_cfgs = sdf[sdf['size']==200].index.tolist() \
                        if select_stimuli=='fullfield' \
                        else sdf[sdf['size']!=200].index.tolist()
        elif experiment=='blobs':
            curr_cfgs = sdf[sdf['morphlevel']!=-1].index.tolist() \
                        if select_stimuli=='images' \
                        else sdf[sdf['morphlevel']==-1].index.tolist()
    else:
        curr_cfgs = sdf.index.tolist()
        
    return curr_cfgs


def get_stimulus_coordinates(dk, experiment):
    sdf = get_stimuli(dk, experiment, match_names=True)
    if len(sdf['xpos'].unique())>1 or len(sdf['ypos'].unique())>1:
        print("*Warning* <%s> More than 1 pos? x: %s, y: %s" \
                    % (dk, str(sdf['xpos'].unique()), str(sdf['ypos'].unique())))

    xpos = sdf['xpos'].unique()
    ypos = sdf['ypos'].unique()

    if len(xpos)>1 or len(ypos)>1:
        print("*Warning* <%s> More than 1 pos? x: %s, y: %s" \
                    % (dk, str(xpos), str(ypos)))

        return np.array(xpos), np.array(ypos)
    else:
        return float(xpos), float(ypos)

    
# --------------------------------------------------------------------
def aggregate_alignment_info(edata, traceid='traces001'):
    exp=str(edata['experiment'].unique()) 
    i=0
    d_=[]
    for (va, dk), g in edata.groupby(['visual_area', 'datakey']):
        # Alignment info
        alignment_info = traceutils.get_trial_alignment(dk, exp, traceid=traceid)
        if alignment_info==-1:
            print("Realign: %s" % dk)
            continue
        iti_pre_ms = float(alignment_info['iti_pre'])*1000
        iti_post_ms = float(alignment_info['iti_post'])*1000
        #print("ITI pre/post: %.1f ms, %.1f ms" % (iti_pre_ms, iti_post_ms))
        d_.append(pd.DataFrame({'visual_area': va, 
                                 'iti_pre': float(alignment_info['iti_pre']),
                                 'iti_post': float(alignment_info['iti_post']),
                                 'stim_dur': float(alignment_info['stim_on_sec']),
                                 'datakey': dk}, index=[i]))
        i+=1
    A = pd.concat(d_, axis=0).reset_index(drop=True)  

    return A


# ###############################################################
# Data formatting
# ###############################################################
def get_zscored_from_ndf(ndf):
    '''Reshape stacked neural metrics, calculate zscore, add config labels back in'''
    trial_means0 = stacked_neuraldf_to_unstacked(ndf)
    rois_ = ndf['cell'].unique()
    cfgs_by_trial = trial_means0['config']
    zscored = zscore_dataframe(trial_means0[rois_])
    zscored['config'] = cfgs_by_trial

    return zscored.sort_index()

def zscore_dataframe(xdf):
    rlist = [r for r in xdf.columns if hutils.isnumber(r)]
    z_xdf = (xdf[rlist]-xdf[rlist].mean()).divide(xdf[rlist].std())
    return z_xdf

def unstacked_neuraldf_to_stacked(ndf, response_type='response', id_vars=['config', 'trial']):
    '''
    Returns stacked df for neuraldata. 
    ndf : input df, columns=cell IDs (and 'config'), rows=response metric for each trial.
    melted : output df, 'cell' is a column
    '''
    ndf['trial'] = ndf.index.tolist()
    melted = pd.melt(ndf, id_vars=id_vars,
                     var_name='cell', value_name=response_type)

    return melted

def stacked_neuraldf_to_unstacked(ndf): #neuraldf):
    '''
    Take stacked neuraldf (NDATA) and make columns from cell IDs (config as last column). 
    Rows are trial values for all trials.
    '''
    other_cols = [k for k in ndf.columns if k not in ['cell', 'response']]
    n2 = ndf.pivot_table(columns=['cell'], index=other_cols)

    rdf = pd.DataFrame(data=n2.values, columns=n2.columns.get_level_values('cell'),
                 index=n2.index.get_level_values('trial'))
    rdf['config'] = n2.index.get_level_values('config')

    return rdf

# ###############################################################
# Data selection 
# ###############################################################
def count_n_responsive(NDATA0, u_dkeys=None):
    '''Returns counts by visual area, datakey'''
    if u_dkeys is None:
        counts = NDATA0[['visual_area', 'datakey','cell']].drop_duplicates()\
                .groupby(['visual_area', 'datakey']).count().reset_index()
        u_dkeys = drop_repeats(counts) 
  
    NDATA = pd.concat([g for (va, dk), g in NDATA0\
                           .groupby(['visual_area', 'datakey'])\
                          if (va, dk) in u_dkeys])       
    counts = NDATA[['visual_area', 'datakey','cell']].drop_duplicates()\
            .groupby(['visual_area', 'datakey']).count().reset_index()\
            .rename(columns={'cell': 'n_responsive'})
    
    return counts, u_dkeys

def count_n_total(assigned_cells, u_dkeys):
    '''Count the number of cells per datakey (UNIQUE)'''
    incl_cells = pd.concat([g for (va, dk), g in assigned_cells\
                                .groupby(['visual_area', 'datakey'])\
                                if (va, dk) in u_dkeys])
    n_total = incl_cells.groupby(['visual_area', 'datakey'])\
                            .count()['cell'].reset_index()\
                            .rename(columns={'cell': 'n_total'})

    return n_total

def count_n_cells(NDATA, name='n_cells'):
    counts = NDATA[['visual_area', 'datakey','cell']].drop_duplicates()\
            .groupby(['visual_area', 'datakey']).count().reset_index()\
            .rename(columns={'cell': name})
 
    return counts

def get_best_fit(CELLS, resp_desc, traceid='traces001', metric='gof'):
    gdata, no_fits, missing_ = osi.aggregate_ori_fits(CELLS, traceid=traceid, 
                                fit_desc=resp_desc, return_missing=True, 
                                verbose=False) 
    # Get best GoF for each 
    best_ixs = gdata.groupby(['visual_area', 'datakey', 'cell'])['gof']\
                    .transform(max) == gdata[metric]
    assert gdata.loc[best_ixs].groupby(['visual_area', 'datakey', 'cell'])\
            .count().max().max()==1
    bestg = gdata.loc[best_ixs].copy()
    bestg = hutils.split_datakey(bestg)
    return bestg
    

def select_assigned_cells(cells0, sdata, experiments=[]):
    '''
    Return assigned cells for a specified experiment.
    cells0: master df of all assigned cells
    sdata: metadata df with all datakeys, experiments, visual areas
    '''
    if not isinstance(experiments, list):
        experiments = [experiments]
    meta_ = sdata[sdata.experiment.isin(experiments)].copy()
    dkeys_ = [(va, dk) for (va, dk), g in meta_.groupby(['visual_area', 'datakey'])]
    cells_ = pd.concat([g for (va, dk), g in \
                            cells0.groupby(['visual_area', 'datakey'])\
                            if (va, dk) in dkeys_])
    return cells_, meta_

def drop_repeats(counts, criterion='max', colname='cell'):
    '''
    From df of counts (N cells per datakey), drop repeats by criterion.
    criterion: takes "max" (or whatever func) along column <colname>
    '''
    counts = hutils.split_datakey(counts)
    unique_dsets = select_best_fovs(counts, criterion=criterion, colname=colname)
    u_dkeys = list([tuple(k) for k in unique_dsets[['visual_area', 'datakey']].values])
    return u_dkeys
    

def choose_best_fov(which_fovs, criterion='max', colname='cell'):
    '''
    Select datakey (SESSION_ANIMAL_FOV) that meets specified criterion.
    Usually, this just finds the max, i.e., returns the index of a "counts" dataframe
    that has the MAX cell #.
    '''
    if criterion=='max':
        max_loc = np.where(which_fovs[colname]==which_fovs[colname].max())[0]
    else:
        max_loc = np.where(which_fovs[colname]==which_fovs[colname].min())[0]

    return max_loc


def unique_cell_df(CELLS, countby=['visual_area', 'datakey', 'cell'], 
                    criterion='max', colname='cell'):

    counts = CELLS[countby].drop_duplicates()\
                        .groupby(['visual_area', 'datakey']).count().reset_index()
    counts = hutils.split_datakey(counts) 
    best_dfs = select_best_fovs(counts, criterion=criterion, colname=colname)
    dkeys = [(v, k) for (v, k), g in best_dfs.groupby(['visual_area', 'datakey'])]
    final_cells = pd.concat([g for (v, k), g in \
                            CELLS.groupby(['visual_area', 'datakey']) \
                            if (v, k) in dkeys])
    return final_cells
 
# def load_sorted_fovs(aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
#     dsets_file = os.path.join(aggregate_dir, 'data-stats', 'sorted_datasets.json')
#     with open(dsets_file, 'r') as f:
#         fov_keys = json.load(f)
    
#     return fov_keys

def select_best_fovs(counts_by_fov, criterion='max', colname='cell'):
    '''
    Select datakey (SESSION_ANIMAL_FOV) across all datakeys based on the counts per datakey.
    Usually, this selects the datkey with the MAX (criterion) number of CELLS (colname) if 
    there are duplicate FOVs.
    '''
    if 'animalid' not in counts_by_fov.columns:
        counts_by_fov = hutils.split_datakey(counts_by_fov)
    # Cycle thru all dsets and drop repeats
    fovkeys = get_sorted_fovs()
    incl_dsets=[]
    for (visual_area, animalid), g in counts_by_fov.groupby(['visual_area', 'animalid']):
        curr_dsets=[]
        try:
            # Check for FOVs that had wrongly assigned visual areas compared to assigned
            if visual_area not in fovkeys[animalid].keys():
                v_area=[]
                for v, vdict in fovkeys[animalid].items():
                    for dk in g['datakey'].unique():
                        a_match = [k for k in vdict for df in g['datakey'].unique() if \
                                    '%s_%s' % (dk.split('_')[0], dk.split('_')[[2]]) in k \
                                     or dk.split('_')[0] in k]
                        if len(a_match)>0:
                            v_area.append(v)
                if len(v_area)>0:
                    curr_dsets = fovkeys[animalid][v_area[0]]
            else:
                curr_dsets = fovkeys[animalid][visual_area]
        except Exception as e:
            print("[%s] Animalid does not exist: %s " % (visual_area, animalid))
            continue

        # Check for sessions/dsets NOT included in current visual area dict
        # This is correctional: if a given FOV is NOT in fovkeys dict, it was a non-repeat FOV
        # for that visual area.
        dkeys_flat = list(itertools.chain(*curr_dsets))
        # These are datakeys assigned to current visual area:
        reformat_dkeys_check = ['%s_%s' % (s.split('_')[0], s.split('_')[2]) \
                                    for s in g['datakey'].unique()]
        # Assigned dkeys not in original source dict (which was made manually)
        missing_segmented_fovs = [s for s in reformat_dkeys_check \
                                if (s not in dkeys_flat) and (s.split('_')[0] not in dkeys_flat) ]

        #for s in missing_segmented_fovs:
        #    curr_dsets.append(s)

        missing_dsets=[]
        for fkey in missing_segmented_fovs:
            found_areas = [k for k, v in fovkeys[animalid].items() \
                             if any([fkey in vv for vv in v]) or any([fkey.split('_')[0] in vv for vv in v])]
            for va in found_areas:
                if fovkeys[animalid][va] not in missing_dsets:
                    missing_dsets.append(fovkeys[animalid][va])
        curr_dsets.extend(list(itertools.chain(*missing_dsets)))
        
        # Select "best" dset if there is a repeat
        if g.shape[0]>1:
            for dkeys in curr_dsets:
                if isinstance(dkeys, tuple):
                    # Reformat listed session strings in fovkeys dict.
                    curr_datakeys = ['_'.join([dk.split('_')[0], animalid, dk.split('_')[-1]])
                            if len(dk.split('_'))>1 \
                            else '_'.join([dk.split('_')[0], animalid, 'fov1']) for dk in dkeys]
                    # Get df data for current "repeat" FOVs
                    which_fovs = g[g['datakey'].isin(curr_datakeys)]
                    # Find which has most cells
                    max_loc = choose_best_fov(which_fovs, criterion=criterion, colname=colname)
                    #max_loc = np.where(which_fovs['cell']==which_fovs['cell'].max())[0]
                    incl_dsets.append(which_fovs.iloc[max_loc])
                else:
                    # THere are no repeats, so just format, then append df data
                    curr_datakey = '_'.join([dkeys.split('_')[0], animalid, dkeys.split('_')[-1]]) \
                                    if len(dkeys.split('_'))>1 \
                                    else '_'.join([dkeys.split('_')[0], animalid, 'fov1'])
                    incl_dsets.append(g[g['datakey']==curr_datakey])
        else:
            #if curr_dsets=='%s_%s' % (session, fov) or curr_dsets==session:
            incl_dsets.append(g)
    incl = pd.concat(incl_dsets, axis=0).reset_index(drop=True)

    return incl.drop_duplicates()



# ------------------------------------------------

def add_roi_positions(rfdf, calculate_position=False, traceid='traces001'):
    '''
    Add ROI position info to RF dataframe (converted and pixel-based).
    Set calculate_position=True, to re-calculate. 
    Note: prev. called 'add_rf_positions'
    '''
    #from . import roi_utils as rutils
    if 'fovnum' not in rfdf.columns:
        rfdf = hutils.split_datakey(rfdf)

    #print("Adding ROI position info...")
    pos_params = ['fov_xpos', 'fov_xpos_pix', 'fov_ypos', 'fov_ypos_pix', 'ml_pos','ap_pos']
    for p in pos_params:
        rfdf[p] = None
    p_list=[]
    for (va, dk, exp), g in rfdf.groupby(['visual_area', 'datakey', 'experiment']):
        if va in [None, 'None']:
            continue
        session, animalid, fovnum = hutils.split_datakey_str(dk)
        try:
            fcoords = roiutils.get_roi_coords(animalid, session, 'FOV%i_zoom2p0x' % fovnum,
                                      traceid=traceid, create_new=False)
            cell_ids = np.array(g['cell'].values) #unique()
            p_ = fcoords['roi_positions'].loc[cell_ids].copy()
            for p in pos_params:
                rfdf.loc[g.index, p] = p_[p].values
            rfdf[pos_params] = rfdf[pos_params].astype(float)
        except Exception as e:
            print('{ERROR} %s, %s' % (va, dk))
            traceback.print_exc()

    return rfdf


def assign_global_cell_ids(cells0):
    cells1 = cells0.reset_index(drop=True)
    cells1['global_ix'] = None
    for va, g in cells1.groupby(['visual_area']):
        ncells_in_area = g.shape[0]
        global_ids = np.arange(0, ncells_in_area)
        cells1.loc[g.index, 'global_ix'] = global_ids
    return cells1

#def assign_global_cell_id(cells):
#    cells['global_ix'] = 0
#    for v, g in cells.groupby(['visual_area']):
#        cells['global_ix'].loc[g.index] = np.arange(0, g.shape[0])
#    return cells.reset_index(drop=True)
#
#def global_cells(cells, remove_too_few=True, min_ncells=5,  return_counts=False):
#    '''
#    cells - dataframe, each row is a cell, has datakey/visual_area fields
#
#    Returns:
#    
#    roidf (dataframe)
#        Globally-indexed rois ('dset_roi' = roi ID in dataset, 'roi': global index)
#    
#    roi_counters (dict)
#        Counts of cells by area (optional)
#
#    '''
#    visual_areas=cells['visual_area'].unique() #['V1', 'Lm', 'Li']
#    print("Assigned visual areas: %s" % str(visual_areas))
# 
#    incl_keys = []
#    if remove_too_few:
#        for (v, k), g in cells.groupby(['visual_area', 'datakey']):
#            if len(g['cell'].unique()) < min_ncells:
#                continue
#            incl_keys.append(k) 
#    else:
#        incl_keys = cells['datakey'].unique()
# 
#    nocells=[]; notrials=[];
#    roi_counters = dict((v, 0) for v in visual_areas)
#    roidf = []
#    for (visual_area, datakey), g in cells[cells['datakey'].isin(incl_keys)].groupby(['visual_area', 'datakey']):
#
#        roi_counter = roi_counters[visual_area]
#
#        # Reindex roi ids for global
#        roi_list = sorted(g['cell'].unique()) 
#        nrs = len(roi_list)
#        roi_ids = [i+roi_counter for i, r in enumerate(roi_list)]
#      
#        # Append to full df
#        roi_dict = {'roi': roi_ids,
#                   'dset_roi': roi_list,
#                   'visual_area': [visual_area for _ in np.arange(0, nrs)],
#                   'datakey': [datakey for _ in np.arange(0, nrs)]}
#        if 'global_ix' in g.columns:
#            roi_dict.update({'global_ix': g['global_ix'].values})
#
#        roidf.append(pd.DataFrame(roi_dict))
#      
#        # Update global roi id counter
#        roi_counters[visual_area] += len(roi_ids)
#
#    if len(roidf)==0:
#        if return_counts:
#            return None, None
#        else:
#            return None
#
#    roidf = pd.concat(roidf, axis=0).reset_index(drop=True)        
#    roidf['animalid'] = [d.split('_')[1] for d in roidf['datakey']]
#    roidf['session'] = [d.split('_')[0] for d in roidf['datakey']]
#    roidf['fovnum'] = [int(d.split('_')[2][3:]) for d in roidf['datakey']]
#   
#    if return_counts:
#        return roidf, roi_counters
#    else:
#        return roidf
#
#

# ===============================================================
# Dataset selection
# ===============================================================
def get_sorted_fovs(filter_by='drop_repeats', excluded_sessions=[]):
    '''
    For each animal, dict of visual areas and list of tuples (each tuple is roughly similar fov)
    Use this to filter out repeated FOVs.
    This list is done manually based on initial targeting, i.e., intentional attempts to
    return to the same FOV.
    '''
    fov_keys = {'JC076': {'V1': [('20190420', '20190501')],
                          'Lm': [('20190423_fov1')],
                          'Li': [('20190422', '20190502')]},

                'JC078': {'Lm': [('20190426', '20190504', '20190509'),
                                 ('20190430', '20190513')]},

                'JC080': {'Lm': [('20190506', '20190603'),
                                 ('20190602_fov2')],
                          'Li': [('20190602_fov1')]},

                'JC083': {'V1': [('20190507', '20190510', '20190511')],
                          'Lm': [('20190508', '20190512', '20190517')]},

                'JC084': {'V1': [('20190522')],
                          'Lm': [('20190525')]},

                'JC085': {'V1': [('20190622')]},

                'JC089': {'Li': [('20190522')]},

                'JC090': {'Li': [('20190605')]},

                'JC091': {'Lm': [('20190627')],
                          'Li': [('20190602', '20190607'),
                                 ('20190606', '20190614'),
                                 ('20191007', '20191008')]},

                'JC092': {'Li': [('20190527_fov2'),
                                 ('20190527_fov3'),
                                 ('20190528')]},

                'JC097': {'V1': [('20190613'),
                                 ('20190615_fov1', '20190617'),
                                 ('20190615_fov2', '20190616')],
                          'Lm': [('20190615_fov3'),
                                 ('20190618')]},

                'JC099': {'Li': [('20190609', '20190612'),
                                 ('20190617')]},

                'JC110': {'V1': [('20191004_fov2', '20191006')],
                          'Lm': [('20191004_fov3'), ('20191004_fov4')]},


                'JC111': {'Li': [('20191003')]},

                'JC113': {'Lm': [('20191012_fov3')],
                          'Li': [('20191012_fov1'), ('20191012_fov2', '20191017', '20191018')]},

                'JC117': {'V1': [('20191111_fov1')],
                          'Lm': [('20191104_fov2'), ('20191111_fov2')],
                          'Li': [('20191104_fov1', '20191105')]},

                'JC120': {'V1': [('20191106_fov3')],
                          'Lm': [('20191106_fov4')],
                          'Li': [('20191106_fov1', '20191111')]},
                #}

                'JC061': {'Lm': [('20190306_fov2'), ('20190306_fov3')]},

                'JC067': {'Li': [('20190319'), ('20190320')]},

                'JC070': {'Li': [('20190314_fov1', '20190315'), # 20190315 better, more reps
                                  ('20190315_fov2'),
                                  ('20190316_fov1'),
                                  ('20190321_fov1', '20190321_fov2')],
                          'Lm': [('20190314_fov2', '20190315_fov3')]},
                'JC073': {'Lm': [('20190322', '20190327')],
                          'Li': [('20190322', '20190327')]} #920190322 better
                }


    return fov_keys

             
    
# ###############################################################
# STATS
# ###############################################################
def get_strongest_response(NDATA0):
    '''Get trial-avg responses, pick stim cond with max response per cell'''
    base_cols = ['visual_area', 'datakey', 'cell']
    if 'experiment' in NDATA0.columns:
        base_cols.append('experiment')

    mean_cols = copy.copy(base_cols)
    mean_cols.append('config')
    meanr = NDATA0.groupby(mean_cols).mean().reset_index()
    best_ixs = meanr.groupby(base_cols)['response']\
                    .transform(max) == meanr['response']
    assert meanr.loc[best_ixs].groupby(base_cols)\
            .count().max().max()==1
    bestg = meanr.loc[best_ixs].copy()
    
    return bestg
   

# ###############################################################
# Data loading
# ###############################################################
def load_frame_labels(datakey, experiment, traceid='traces001',
                        rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovnum = hutils.split_datakey_str(datakey) 
    fname = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovnum,
                        'combined_%s_static' % experiment, 'traces', 
                        '%s*' % traceid, 'data_arrays', 'labels.npz'))[0]
    l = np.load(fname, allow_pickle=True)
    labels = pd.DataFrame(data=l['labels_data'], columns=l['labels_columns'])
    labels = hutils.convert_columns_byte_to_str(labels)

    return labels


def load_responsive_neuraldata(experiment, meta=None, traceid='traces001',
                      response_type='dff', trial_epoch='plushalf',
                      responsive_test='nstds', responsive_thr=10,n_stds=2.5,
                      retino_thr=0.01, retino_delay=0.5):
    '''
    Load ALL aggregate data for ALL FOV, with correctly assigned cells (NDATA).
    --> calls get_aggregate_data()

    Only cells that pass specified responsivity tests are included.
    Returns all data, so use count_n_responsive() to filter any repeat FOVs.
    '''
    if experiment=='gratings':
        NDATA = get_aggregate_data(experiment, meta=meta, traceid=traceid, 
                              response_type=response_type, epoch=trial_epoch,
                              responsive_test=responsive_test, 
                              responsive_thr=responsive_thr, n_stds=n_stds)
    elif experiment=='blobs':
        NDATA = get_aggregate_data(experiment, meta=meta, traceid=traceid, 
                              response_type=response_type, epoch=trial_epoch,
                              responsive_test=responsive_test, 
                              responsive_thr=responsive_thr, n_stds=n_stds)
    elif experiment=='retino':
        retinodata = get_aggregate_retinodata(meta=meta, traceid=traceid, 
                                mag_thr=retino_thr, delay_thr=retino_delay)
        NDATA = get_responsive_retino(retinodata, mag_thr=retino_thr)
    elif experiment in ['rfs', 'rfs10']:
        # TODO: what is the right way to select for responsive cells, separate
        # from RF fitting?
        nd_=[]
        for exp in ['rfs', 'rfs10']:
            nd0 = get_aggregate_data(exp, meta=meta, traceid=traceid, 
                              response_type=response_type, epoch=trial_epoch,
                              responsive_test=responsive_test, 
                              responsive_thr=responsive_thr, n_stds=n_stds)
            nd0['experiment'] = exp
            nd_.append(nd0)
        NDATA = pd.concat(nd_, axis=0).reset_index(drop=True)
    else:
        print("Unknown experiment type: %s" % experiment)
        return None
    return NDATA


def load_corrected_dff_traces(animalid, session, fov, experiment='blobs', traceid='traces001',
                              return_traces=True, epoch='stimulus', metric='mean', return_labels=False,
                              rootdir='/n/coxfs01/2p-data'):
    
    print('... calculating F0 for df/f')
    #experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] and int(session)<20190512) else experiment

    # Load corrected
    soma_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov,
                                    '*%s_static' % (experiment), 'traces', '%s*' % traceid,
                                    'data_arrays', 'np_subtracted.npz'))[0]
    dset = np.load(soma_fpath, allow_pickle=True)
    Fc = pd.DataFrame(dset['data']) # np_subtracted:  Np-corrected trace, with baseline subtracted

    # Load raw (pre-neuropil subtraction)
    raw = np.load(soma_fpath.replace('np_subtracted', 'raw'), allow_pickle=True)
    F0_raw = pd.DataFrame(raw['f0'])

    # Calculate df/f
    dff = Fc.divide(F0_raw) # dff 

    if return_traces:
        if return_labels:
            labels = pd.DataFrame(data=dset['labels_data'],columns=dset['labels_columns'])
            labels = hutils.convert_columns_byte_to_str(labels)

            return dff, labels
        else:
            return dff
    else:
        labels = pd.DataFrame(data=dset['labels_data'],columns=dset['labels_columns'])
        labels = hutils.convert_columns_byte_to_str(labels)
        dfmat = traces_to_trials(dff, labels, epoch=epoch, metric=metric)
        return dfmat


def traces_to_trials(traces, labels, epoch='stimulus', metric='mean', n_on=None):
    '''
    Returns dataframe w/ columns = roi ids, rows = mean response to stim ON per trial
    Last column is config on given trial.
    
    epoch: str
        stimulus: avg over stimulus period (overrides n_on)
        firsthalf: avg over first half of stimulus on period
        plushalf: avg over stimulus period + half post
        baseline: avg over baseline period

    '''
    print(labels.columns)
    s_on = int(labels['stim_on_frame'].mean())
    if epoch=='stimulus':
        n_on = int(labels['nframes_on'].mean()) 
    elif epoch=='firsthalf':
        n_on = int(labels['nframes_on'].mean()/2.)
    elif epoch=='plushalf':
        half_dur = labels['nframes_on'].mean()/2.
        n_on = int(labels['nframes_on'].mean() + half_dur) 

    roi_list = traces.columns.tolist()
    trial_list = np.array([int(trial[5:]) for trial, g in labels.groupby(['trial'])])
    if epoch in ['stimulus', 'firsthalf', 'plushalf']:
        mean_responses = pd.DataFrame(np.vstack([np.nanmean(traces.iloc[g.index[s_on:s_on+n_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
        if metric=='zscore':
            std_responses = pd.DataFrame(np.vstack([np.nanstd(traces.iloc[g.index[0:s_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
            tmp = mean_responses.divide(std_baseline)
            mean_responses = tmp.copy()
    elif epoch == 'baseline':
        mean_responses = pd.DataFrame(np.vstack([np.nanmean(traces.iloc[g.index[0:s_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
        if metric=='zscore':
            std_baseline = pd.DataFrame(np.vstack([np.nanstd(traces.iloc[g.index[0:s_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
            tmp = mean_responses.divide(std_baseline)
            mean_responses = tmp.copy()
 
    condition_on_trial = np.array([g['config'].unique()[0] for trial, g in labels.groupby(['trial'])])
    mean_responses['config'] = condition_on_trial

    return mean_responses





def get_cells_by_area(sdata, create_new=False, excluded_datasets=[], 
                return_missing=False, verbose=False,
                rootdir='/n/coxfs01/2p-data',
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Use retionrun to ID area boundaries. If more than 1 retino, combine.
    '''
    #from . import roi_utils as rutils
    cells=None
    missing_segmentation=[]
    # Check existing
    cells_fpath = os.path.join(aggregate_dir, 'assigned_cells.pkl')
    if os.path.exists(cells_fpath) and create_new is False:
        try:
            with open(cells_fpath, 'rb') as f:
                results = pkl.load(f, encoding='latin1')
            cells = results['cells']
            missing_segmentation = results['missing']
        except Exception as e:
            create_new=True

    # bad segmentation
    excluded_datasets = ['20190321_JC073_fov1',
                         '20190314_JC070_fov2',
                         '20190602_JC080_fov1', 
                         '20190605_JC090_fov1',
                         '20191003_JC111_fov1', 
                         '20191104_JC117_fov1', '20191104_JC117_fov2', 
                         #'20191105_JC117_fov1',
                         '20191108_JC113_fov1', '20191004_JC110_fov3',
                         '20191008_JC091_fov'] 
    if create_new:
        print("Assigning cells")
        d_ = []
        for (animalid, session, fov, datakey), g \
                in sdata.groupby(['animalid', 'session', 'fov', 'datakey']):
            if datakey in excluded_datasets:
                continue
            roi_assignments=dict()
            try:
                # Get best retino
                all_retinos = retutils.get_average_mag_across_pixels(datakey)     
                retinorun = all_retinos.iloc[all_retinos[1].idxmax()][0]
                #retinorun = all_retinos.loc[all_retinos[1].idxmax()][0] 
                roi_assignments = roiutils.load_roi_assignments(animalid, 
                                                    session, \
                                                    fov, retinorun=retinorun)
                if roi_assignments is None:
                    missing_segmentation.append((datakey, retinorun))
                    continue
            except Exception as e:
                if verbose:
                    print("... no seg. %s (%s)" % (datakey, retinorun))
                    print(e)
                missing_segmentation.append((datakey, retinorun))
                continue 
            for varea, rlist in roi_assignments.items():
                if hutils.isnumber(varea):
                    continue  
                tmpd = pd.DataFrame({'cell': list(set(rlist))})
                session, animalid, fovn = hutils.split_datakey_str(datakey)
                metainfo = {'visual_area': varea, 
                            'animalid': animalid, 'session': session,
                            'fov': fov, 'fovnum': fovn, 
                            'datakey': datakey}
                tmpd = hutils.add_meta_to_df(tmpd, metainfo)
                if verbose:
                    print('    %s (%s): got %i cells' % (dk, varea, len(tmpd)))
                d_.append(tmpd)
        cells = pd.concat(d_, axis=0).reset_index(drop=True)
        cells = cells[~cells['datakey'].isin(excluded_datasets)]
       
        # Save
        results = {'cells': cells, 'missing': missing_segmentation}
        with open(cells_fpath, 'wb') as f:
            pkl.dump(results, f, protocol=2)

    #print("Missing %i datasets for segmentation:" % len(missing_segmentation)) 
    if verbose: 
        print("Segmentation, missing:")
        for r in missing_segmentation:
            print(r)
    else:
        print("Segmentation: missing %i dsets" % len(missing_segmentation))
    if return_missing:
        return cells, missing_segmentation
    else:
        return cells

def get_aggregate_info(traceid='traces001', fov_type='zoom2p0x', state='awake',
                visual_areas=['V1', 'Lm', 'Li'], 
                return_cells=False, create_new=False,
                return_missing=False,
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info_assigned.pkl')
    if create_new is False:
        print(sdata_fpath)
        sdata=None
        try:
            assert os.path.exists(sdata_fpath), \
                        "Path does not exist: %s" % sdata_fpath
            with open(sdata_fpath, 'rb') as f:
                sdata = pkl.load(f, encoding='latin1')
            assert sdata is not None and isinstance(sdata, pd.DataFrame), \
                                        "Bad metadata loading, creating new"
            if return_cells:
                cells, missing_seg = get_cells_by_area(sdata, create_new=False,
                                                    return_missing=True)
                cells = cells[cells.visual_area.isin(visual_areas)]
            all_sdata = sdata.copy()
        except Exception as e:
            traceback.print_exc()
            create_new=True

    if create_new:
        print("Loading old...")
        unassigned_fp = os.path.join(aggregate_dir, 'dataset_info.pkl') 
        with open(unassigned_fp, 'rb') as f:
            sdata = pkl.load(f, encoding='latin1')
        cells, missing_seg = get_cells_by_area(sdata, create_new=create_new,
                                            return_missing=True)
        cells = cells[cells.visual_area.isin(visual_areas)]

        d_=[]
        all_ = cells[['visual_area', 'datakey', 'fov']]\
                        .drop_duplicates().reset_index(drop=True)
        for (va, dk, fov), g in all_.groupby(['visual_area', 'datakey','fov']):
            if va not in visual_areas:
                print("... skipping %s" % va)
                continue
            found_exps = sdata[(sdata['datakey']==dk)]['experiment'].values
            tmpd = pd.DataFrame({'experiment': found_exps})
            tmpd['visual_area'] = va 
            tmpd['datakey'] = dk
            tmpd['fov'] = fov
            d_.append(tmpd)
        all_sdata = pd.concat(d_, axis=0).reset_index(drop=True)
        all_sdata = hutils.split_datakey(all_sdata)
        all_sdata['fovnum'] = [int(f.split('_')[0][3:]) \
                                    for f in all_sdata['fov']]
        with open(sdata_fpath, 'wb') as f:
            pkl.dump(all_sdata, f, protocol=2)
    
    if return_cells:
        if return_missing:
            return all_sdata, cells, missing_seg
        else:
            return all_sdata, cells
    else:
        if return_missing:
            return all_sdata, missing_seg
        else:
            return all_sdata



def get_aggregate_filepath(experiment, traceid='traces001', response_type='dff', 
                        epoch='stimulus', 
                       responsive_test='ROC', responsive_thr=0.05, n_stds=0.0,
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''Get filepath for specified aggregate data
    (prev called get_aggregate_data_filepath()
    '''
    #### Get DATA   
    data_dir = os.path.join(aggregate_dir, 'data-stats')
    data_desc_base = create_dataframe_name(traceid=traceid, 
                            response_type=response_type, 
                            responsive_test=responsive_test, 
                            responsive_thr=responsive_thr,
                            epoch=epoch)    
    data_desc = 'aggr_%s_%s' % (experiment, data_desc_base)
    data_outfile = os.path.join(data_dir, '%s.pkl' % data_desc)

    return data_outfile #print(data_desc)
    


def create_dataframe_name(traceid='traces001', response_type='dff', 
                             epoch='stimulus',
                             responsive_test='ROC', responsive_thr=0.05, n_stds=0.0): 

    data_desc = 'trialmeans_%s_%s-thr-%.2f_%s_%s' \
                    % (traceid, str(responsive_test), responsive_thr, response_type, epoch)
    return data_desc


def get_aggregate_retinodata(meta=None, traceid='traces001', 
                        mag_thr=None, delay_thr=None, return_missing=False,
                        visual_areas=['V1', 'Lm', 'Li'],
                        create_new=False, redo_fov=False,
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Load or create aggregate retinodata for all datasets specified.
    Includes ALL assigned cells (ret_cells). 
    Use get_responsive_retino() to filter by mag_thr.
    
    Returns df with phase and magnitude (columns) for 
    AZ/EL conditions for ALL cells.
    '''
    sdata, cells0 = get_aggregate_info(visual_areas=visual_areas, 
                                        return_cells=True)
    if meta is None:
        meta = sdata[sdata.experiment=='retino'].copy() 
    # Only get cells for current experiment
    all_dkeys = [(va, dk) for (va, dk), g \
                    in meta.groupby(['visual_area', 'datakey'])]
    CELLS = pd.concat([g for (va, dk), g \
                    in cells0.groupby(['visual_area', 'datakey'])\
                    if (va, dk) in all_dkeys])

    retinodata=None
    print("Aggregating retinodata and saving")
    # Create output path
    stats_dir = os.path.join(aggregate_dir, 'data-stats')
    retino_dfile = os.path.join(stats_dir, 'aggr_retinodata.pkl')
        
    if not create_new:
        try:
            with open(retino_dfile, 'rb') as f:
                retinodata = pkl.load(f)
        except Exception as e:
            traceback.print_exc()
            create_new=True
            
    if create_new:
        r_=[]; errs=[];
        for (va, dk), curr_cells in CELLS.groupby(['visual_area', 'datakey']):
            df = retutils.get_retino_fft(dk, curr_cells=curr_cells, 
                                    mag_thr=mag_thr, delay_thr=delay_thr,
                                    create_new=redo_fov)
            if df is None:
                errs.append((va, dk))
                continue
            #assert 'retinorun' in df.columns
            df['visual_area'] = va
            df['datakey'] = dk
            df['cell'] = df.index
            r_.append(df)
        retinodata = pd.concat(r_, axis=0).reset_index(drop=True)
        with open(retino_dfile, 'wb') as f:
            pkl.dump(retinodata, f, protocol=2)
    if return_missing:
        return retinodata, errs
    else:
        return retinodata

def get_responsive_retino(retinodata, mag_thr=0.01):
    '''
    retinodata (df): columns are phase and mag for AZ/EL conds (rows=cells).
    Expects aggregate data (cycles by visual area and datakey).
    Returns retino data for cells responsive (mag_thr)
    '''
    # get responsive
    p_ = []
    for (va, dk), rdf in retinodata.dropna().groupby(['visual_area', 'datakey']):
        pass_ = np.where(rdf[['mag_az', 'mag_el']].mean(axis=1)>mag_thr)[0]
        df_ = rdf.iloc[pass_]
        p_.append(df_)
    retino_responsive = pd.concat(p_, axis=0)

    return retino_responsive


def get_aggregate_data(experiment, meta=None, traceid='traces001', 
                    response_type='dff', epoch='stimulus', 
                    responsive_test='ROC', responsive_thr=0.05, n_stds=0.0,
                    rename_configs=True, equalize_now=False, zscore_now=False,
                    return_configs=False, images_only=False, 
                    diff_configs = ['20190327_JC073_fov1', '20190314_JC070_fov1'], # 20190426_JC078 (LM, backlight)
                    aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                    visual_areas=['V1','Lm', 'Li'], verbose=False):
    '''
    Oldv:  
    load_aggregate_data(), then get_neuraldata() combines CELLS + MEANS into NEURALDATA.

    Do it this non-streamlined way bec easier to keep data for 1 FOV
    as is, then just sub-select based on cell assignments (CELLS).

   NEURALDATA (dict)
        keys=visual areas
        values = MEANS (i.e., dict of dfs) for each visual area
        Only inclues cells that are assigned to the specified area.
    '''
    sdata, cells0 = get_aggregate_info(visual_areas=visual_areas, 
                                        return_cells=True)
    if meta is None:
        meta = sdata[sdata.experiment==experiment].copy()
 
    # Only get cells for current experiment
    all_dkeys = [(va, dk) for (va, dk), g \
                    in meta.groupby(['visual_area', 'datakey'])]
    CELLS = pd.concat([g for (va, dk), g \
                    in cells0.groupby(['visual_area', 'datakey'])\
                    if (va, dk) in all_dkeys])

    visual_areas = CELLS['visual_area'].unique()
    # Get "MEANS" dict (output of classification.aggregate_data_stats.py, p2)
    if experiment!='blobs':
        rename_configs=False
        return_configs=False
    # Load trial metrics dicts (from <aggr_dir>/data-stats/aggr_EXP_...pkl)
    MEANS = load_aggregate_data(experiment, traceid=traceid, 
                        response_type=response_type, epoch=epoch,
                        responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, n_stds=n_stds,
                        rename_configs=rename_configs, 
                        equalize_now=equalize_now, zscore_now=zscore_now,
                        return_configs=return_configs, 
                        images_only=images_only)
    # Combine into dict by visual area
    NEURALDATA = dict((visual_area, {}) for visual_area in visual_areas)
    rf_=[]
    for (visual_area, datakey), curr_c in CELLS.groupby(['visual_area', 'datakey']):
        #print(datakey)
        if visual_area not in NEURALDATA.keys():
            print("... skipping: %s" % visual_area)
            continue
        if datakey not in MEANS.keys(): #['datakey'].values:
            print("... not in exp: %s" % datakey)
            continue
        # Get neuradf for these cells only
        neuraldf = get_neuraldf_for_cells_in_area(curr_c, MEANS, 
                        datakey=datakey, visual_area=visual_area)
        if verbose:
            # Which cells are in assigned area
            n_resp = int(MEANS[datakey].shape[1]-1)
            curr_assigned = curr_c['cell'].unique() 
            print("[%s] %s: %i cells responsive (%i in fov)" \
                        % (visual_area, datakey, len(curr_assigned), n_resp))
            if neuraldf is not None:
                print("Neuraldf: %s" % str(neuraldf.shape)) 
            else:
                print("No keys: %s|%s" % (visual_area, datakey))

        if neuraldf is not None:            
            NEURALDATA[visual_area].update({datakey: neuraldf})
    # Convert final dict to dataframe
    NDATA = neuraldf_dict_to_dataframe(NEURALDATA)

    return NDATA


def neuraldf_dict_to_dataframe(NEURALDATA, response_type='response', add_cols=[]):
    ndfs = []
    id_vars = ['datakey', 'config', 'trial']
    id_vars.extend(add_cols)
    k1 = list(NEURALDATA.keys())[0]
    if isinstance(NEURALDATA[k1], dict):
        id_vars.append('visual_area')
        for visual_area, vdict in NEURALDATA.items():
            for datakey, neuraldf in vdict.items():
                neuraldf['visual_area'] = visual_area
                neuraldf['datakey'] = datakey
                neuraldf.loc[neuraldf.index, 'trial'] = neuraldf.index.tolist()
                melted = pd.melt(neuraldf, id_vars=id_vars, var_name='cell', 
                                value_name=response_type)
                ndfs.append(melted)
    else:
        for datakey, neuraldf in NEURALDATA.items():
            neuraldf['datakey'] = datakey
            neuraldf.loc[neuraldf.index, 'trial'] = neuraldf.index.tolist()
            melted = pd.melt(neuraldf, id_vars=id_vars, 
                             var_name='cell', value_name=response_type)
            ndfs.append(melted)

    NDATA = pd.concat(ndfs, axis=0)
   
    return NDATA



def load_aggregate_data(experiment, traceid='traces001', 
                    response_type='dff', epoch='stimulus', 
                    responsive_test='ROC', responsive_thr=0.05, n_stds=0.0,
                    rename_configs=True, equalize_now=False, zscore_now=False,
                    return_configs=False, images_only=False, 
            diff_configs = ['20190327_JC073_fov1', '20190314_JC070_fov1'], # 20190426_JC078 (LM, backlight)
            aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Return dict of neural dataframes (keys are datakeys).

    rename_configs (bool) : Get each dataset's stim configs (sdf), and rename matching configs to match master.
    Note, rename_configs *only* tested with experiment=blobs. (Prev. called check_configs).

    equalize_now (bool) : Random sample trials per config so that same # trials/config.
    zscore_now (bool) : Zscore neurons' responses.
    ''' 
    MEANS=None
    SDF=None
    data_outfile = get_aggregate_filepath(experiment, traceid=traceid, 
                        response_type=response_type, epoch=epoch,
                        responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, n_stds=n_stds,
                        aggregate_dir=aggregate_dir)
    # print("...loading: %s" % data_outfile)

    with open(data_outfile, 'rb') as f:
        MEANS = pkl.load(f, encoding='latin1')
    print("...loading: %s" % data_outfile)

    #### Fix config labels  
    if experiment=='blobs':
        if (rename_configs or return_configs):
            SDF, renamed_configs = check_sdfs_blobs(MEANS.keys(),
                                          images_only=images_only, 
                                          return_incorrect=True)
            if rename_configs:
                sdf_master = get_master_sdf(experiment='blobs',
                                            images_only=images_only)
                for k, cfg_lut in renamed_configs.items():
                    updated_cfgs = [cfg_lut[cfg] for cfg \
                                        in MEANS[k]['config']]
                    MEANS[k]['config'] = updated_cfgs
        if images_only is True: #Update MEANS dict
            for k, md in MEANS.items():
                incl_configs = SDF[k].index.tolist()
                MEANS[k] = md[md['config'].isin(incl_configs)]
    elif experiment=='gratings':
        SDF, incorrect = check_sdfs_gratings(MEANS.keys(), return_incorrect=True,
                            return_all=False)
        
    if equalize_now:
        # Get equal counts
        print("---equalizing now---")
        MEANS = equal_counts_per_condition(MEANS)

    if return_configs: 
        return MEANS, SDF
    else:
        return MEANS

def equal_counts_per_condition(MEANS):
    '''
    MEANS: dict
        keys = datakeys
        values = neural dataframes (columns=rois, index=trial numbers)
    
    Resample so that N trials per condition is the same as min N.
        '''

    for k, v in MEANS.items():
        v_df = equal_counts_df(v)
        
        MEANS[k] = v_df

    return MEANS

def equal_counts_df(ndf, equalize_by='config'): #, randi=None):
    neuraldf = ndf.copy()
    curr_counts = neuraldf[equalize_by].value_counts()
    if len(curr_counts.unique())==1:
        return neuraldf #continue
        
    min_ntrials = curr_counts.min()
    all_cfgs = ndf[equalize_by].unique()
    drop_trial_col=False
    if 'trial' not in neuraldf.columns:
        neuraldf['trial'] = None
        trialvals = ndf.index.tolist()
        neuraldf.loc[ndf.index.tolist(), 'trial'] = trialvals
        drop_trial_col = True

    #kept_trials=[]
    #for cfg in all_cfgs:
        #curr_trials = neuraldf[neuraldf[equalize_by]==cfg].index.tolist()
        #np.random.shuffle(curr_trials)
        #kept_trials.extend(curr_trials[0:min_ntrials])
    #kept_trials=np.array(kept_trials)
    kept_trials = neuraldf[['config', 'trial']].drop_duplicates().groupby(['config'])\
        .apply(lambda x: x.sample(n=min_ntrials, replace=False, random_state=None))['trial'].values
    
    subdf = neuraldf[neuraldf.trial.isin(kept_trials)]
    assert len(subdf[equalize_by].value_counts().unique())==1, \
            "Bad resampling... Still >1 n_trials"
    if drop_trial_col:
        subdf = subdf.drop('trial', axis=1)
 
    return subdf #neuraldf.loc[kept_trials]



# Overlaps, cell assignments, etc.
def get_neuraldf_for_cells_in_area(cells, MEANS, datakey=None, visual_area=None):
    '''
    For a given dataframe (index=trials, columns=cells), only return cells
    in specified visual area
    '''
    neuraldf=None
    try:
        if isinstance(MEANS, dict):
            if 'V1' in MEANS.keys(): # dict of dict
                neuraldf_dict = MEANS[visual_area]
            else:
                neuraldf_dict = MEANS
        elif isinstance(MEANS, pd.DataFrame):
            MEANS = neuraldf_dataframe_to_dict(MEANS)
            neuraldf_dict = MEANS[visual_area] 

        assert datakey in neuraldf_dict.keys(), "%s--not found in RESPONSES" % datakey
        assert datakey in cells['datakey'].values, "%s--not found in SEGMENTED" % datakey

        curr_rois = cells[(cells['datakey']==datakey) 
                        & (cells['visual_area']==visual_area)]['cell'].astype(int).values
        curr_cols = [i for i in np.array(curr_rois.copy()) if i in neuraldf_dict[datakey].columns.tolist()]
        #curr_cols = list(curr_rois.copy())
        neuraldf = neuraldf_dict[datakey][curr_cols].copy()
        neuraldf['config'] = neuraldf_dict[datakey]['config'].copy()
    except Exception as e:
        return neuraldf
 
    return neuraldf




def get_neuraldf(datakey, experiment, traceid='traces001', 
                   response_type='dff', epoch='stimulus',
                   responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, 
                   create_new=False, redo_stats=False, n_processes=1,
                   rootdir='/n/coxfs01/2p-data'):
    '''
    epoch options
        stimulus: use full stimulus period
        baseline: average over baseline period
        firsthalf: use first HALF of stimulus period
        plushalf:  use stimulus period + extra half 
    '''
    # output
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment
    try:
        traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                            'FOV%i_*' % fovnum, 'combined_%s_static' % experiment_name,
                            'traces', '%s*' % traceid))[0]
    except Exception as e:
        print("%s %s %i %s - no traceid!" % (session, animalid, fovnum, experiment_name))
        return None

    if responsive_test is not None:
        statdir = os.path.join(traceid_dir, 'summary_stats', str(responsive_test))
    else:
        statdir = os.path.join(traceid_dir, 'summary_stats')
    data_desc_base = create_dataframe_name(traceid=traceid, 
                                            response_type=response_type, 
                                            responsive_test=responsive_test,
                                            responsive_thr=responsive_thr,
                                            epoch=epoch)
    ndf_fpath = os.path.join(statdir, '%s.pkl' % data_desc_base)
    
    create_new = redo_stats is True
    if not create_new:
        try:
            with open(ndf_fpath, 'rb') as f:
                mean_responses = pkl.load(f, encoding='latin1')
        except Exception as e:
            print(e)
            print("Unable to get neuraldf. Creating now.")
            create_new=True

    if create_new:
        # Load traces
        if response_type=='dff0':
            meanr = load_corrected_dff_traces(animalid, session, 'FOV%i_zoom2p0x' % fovnum, 
                                    experiment=experiment_name, traceid=traceid,
                                  return_traces=False, epoch=epoch, metric='mean') 
            tmp_desc_base = create_dataframe_name(traceid=traceid, 
                                            response_type='dff', 
                                            responsive_test=responsive_test,
                                            responsive_thr=responsive_thr,
                                            epoch=epoch) 
            tmp_fpath = os.path.join(statdir, '%s.pkl' % tmp_desc_base)
            with open(tmp_fpath, 'rb') as f:
                tmpd = pkl.load(f, encoding='latin1')
            cols = tmpd.columns.tolist()
            mean_responses = meanr[cols]
            print("min:", mean_responses.min().min())

        else:
            trace_type = 'df' if response_type=='zscore' else response_type
            traces, labels, sdf = load_traces(datakey, experiment_name, traceid=traceid, 
                                              response_type=trace_type,
                                              responsive_test=responsive_test, 
                                              responsive_thr=responsive_thr, 
                                              n_stds=n_stds,
                                              redo_stats=redo_stats, 
                                              n_processes=n_processes)
            if traces is None:
                return None
            # Calculate mean trial metric
            metric = 'zscore' if response_type=='zscore' else 'mean'
            mean_responses = traces_to_trials(traces, labels, epoch=epoch, 
                    metric=response_type)

        # save
        with open(ndf_fpath, 'wb') as f:
            pkl.dump(mean_responses, f, protocol=2)

    return mean_responses

#def process_and_save_traces(trace_type='dff',
#                            animalid=None, session=None, fov=None, 
#                            experiment=None, traceid='traces001',
#                            soma_fpath=None,
#                            rootdir='/n/coxfs01/2p-data'):
#    '''Process raw traces (SOMA ONLY), and calculate dff'''
#
#    print("... processing + saving data arrays (%s)." % trace_type)
#    assert (animalid is None and soma_fpath is not None) or (soma_fpath is None and animalid is not None), "Must specify either dataset params (animalid, session, etc.) OR soma_fpath to data arrays."
#
#    if soma_fpath is None:
#        # Load default data_array path
#        search_str = '' if 'combined' in experiment else '_'
#        soma_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov,
#                                '*%s%s*' % (experiment, search_str), 
#                                'traces', '%s*' % traceid, 
#                                'data_arrays', 'np_subtracted.npz'))[0]
#    dset = np.load(soma_fpath, allow_pickle=True)
#    
#    # Stimulus / condition info
#    labels = pd.DataFrame(data=dset['labels_data'], 
#                          columns=dset['labels_columns'])
#    sdf = pd.DataFrame(dset['sconfigs'][()]).T
#    if 'blobs' in soma_fpath: #self.experiment_type:
#        sdf = reformat_morph_values(sdf)
#    run_info = dset['run_info'][()]
#
#    xdata_df = pd.DataFrame(dset['data'][:]) # neuropil-subtracted & detrended
#    F0 = pd.DataFrame(dset['f0'][:]).mean().mean() # detrended offset
#    
#    #% Add baseline offset back into raw traces:
#    neuropil_fpath = soma_fpath.replace('np_subtracted', 'neuropil')
#    npdata = np.load(neuropil_fpath, allow_pickle=True)
#    neuropil_f0 = np.nanmean(np.nanmean(pd.DataFrame(npdata['f0'][:])))
#    neuropil_df = pd.DataFrame(npdata['data'][:]) 
#    add_np_offsets = list(np.nanmean(neuropil_df, axis=0)) 
#    print("    adding NP offset (NP f0 offset: %.2f)" % neuropil_f0)
#
#    # # Also add raw 
#    raw_fpath = soma_fpath.replace('np_subtracted', 'raw')
#    rawdata = np.load(raw_fpath, allow_pickle=True)
#    raw_f0 = np.nanmean(np.nanmean(pd.DataFrame(rawdata['f0'][:])))
#    raw_df = pd.DataFrame(rawdata['data'][:])
#    print("    adding raw offset (raw f0 offset: %.2f)" % raw_f0)
#
#    raw_traces = xdata_df + add_np_offsets + raw_f0 
#    #+ neuropil_f0 + raw_f0 # list(np.nanmean(raw_df, axis=0)) #.T + F0
#     
#    # SAVE
#    data_dir = os.path.split(soma_fpath)[0]
#    data_fpath = os.path.join(data_dir, 'corrected.npz')
#    print("... Saving corrected data (%s)" %  os.path.split(data_fpath)[-1])
#    np.savez(data_fpath, data=raw_traces.values)
#  
#    # Process dff/df/etc.
#    stim_on_frame = labels['stim_on_frame'].unique()[0]
#    tmp_df = []
#    tmp_dff = []
#    for k, g in labels.groupby(['trial']):
#        tmat = raw_traces.loc[g.index]
#        bas_mean = np.nanmean(tmat[0:stim_on_frame], axis=0)
#        
#        #if trace_type == 'dff':
#        tmat_dff = (tmat - bas_mean) / bas_mean
#        tmp_dff.append(tmat_dff)
#
#        #elif trace_type == 'df':
#        tmat_df = (tmat - bas_mean)
#        tmp_df.append(tmat_df)
#
#    dff_traces = pd.concat(tmp_dff, axis=0) 
#    data_fpath = os.path.join(data_dir, 'dff.npz')
#    print("... Saving dff data (%s)" %  os.path.split(data_fpath)[-1])
#    np.savez(data_fpath, data=dff_traces.values)
#
#    df_traces = pd.concat(tmp_df, axis=0) 
#    data_fpath = os.path.join(data_dir, 'df.npz')
#    print("... Saving df data (%s)" %  os.path.split(data_fpath)[-1])
#    np.savez(data_fpath, data=df_traces.values)
#
#    if trace_type=='dff':
#        return dff_traces, labels, sdf, run_info
#    elif trace_type == 'df':
#        return df_traces, labels, sdf, run_info
#    else:
#        return raw_traces, labels, sdf, run_info
#

#def load_dataset(soma_fpath, trace_type='dff', is_neuropil=False,
#                add_offset=True, make_equal=False, create_new=False):
#    '''
#    Loads all the roi traces and labels.
#    If want to load corrected NP traces, set flag is_neuropil.
#    To load raw NP traces, set trace_type='neuropil' and is_neuropil=False.
#
#    '''
#    traces=None
#    labels=None
#    sdf=None
#    run_info=None
#    try:
#        data_fpath = soma_fpath.replace('np_subtracted', trace_type)
#        if not os.path.exists(data_fpath) or create_new is True:
#            # Process data and save
#            traces, labels, sdf, run_info = process_and_save_traces(
#                                                    trace_type=trace_type,
#                                                    soma_fpath=soma_fpath
#            )
#        else:
#            if is_neuropil:
#                np_fpath = data_fpath.replace(trace_type, 'neuropil')
#                traces = traceutils.load_corrected_neuropil_traces(np_fpath)
#            else:
#                #print("... loading saved data array (%s)." % trace_type)
#                traces_dset = np.load(data_fpath, allow_pickle=True)
#                traces = pd.DataFrame(traces_dset['data'][:]) 
#
#            # Stimulus / condition info
#            labels_fpath = data_fpath.replace(\
#                            '%s.npz' % trace_type, 'labels.npz')
#            labels_dset = np.load(labels_fpath, allow_pickle=True, 
#                            encoding='latin1') 
#            labels = pd.DataFrame(data=labels_dset['labels_data'], 
#                                  columns=labels_dset['labels_columns'])
#            labels = hutils.convert_columns_byte_to_str(labels)
#            sdf = pd.DataFrame(labels_dset['sconfigs'][()]).T.sort_index()
#            if 'blobs' in data_fpath: 
#                sdf = reformat_morph_values(sdf)
#            # Format condition info:
#            if 'image' in sdf['stimtype']:
#                aspect_ratio = sdf['aspect'].unique()[0]
#                sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]
#            # Get run info 
#            run_info = labels_dset['run_info'][()]
#        if make_equal:
#            print("... making equal")
#            traces, labels = check_counts_per_condition(traces, labels)      
#    except Exception as e:
#        traceback.print_exc()
#        print("ERROR LOADING DATA")
#
#    return traces, labels, sdf, run_info
#

def load_run_info(animalid, session, fov, run, traceid='traces001',
                  rootdir='/n/coxfs01/2p-ddata'):
   
    search_str = '' if 'combined' in run else '_'  
    labels_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s%s*' % (run, search_str),
                           'traces', '%s*' % traceid, 'data_arrays', 'labels.npz'))[0]
    labels_dset = np.load(labels_fpath, allow_pickle=True, encoding='latin1')
    
    # Stimulus / condition info
    labels = pd.DataFrame(data=labels_dset['labels_data'], 
                          columns=labels_dset['labels_columns'])
    labels = hutils.convert_columns_byte_to_str(labels)

    sdf = pd.DataFrame(labels_dset['sconfigs'][()]).T
    if 'blobs' in labels_fpath: #self.experiment_type:
        sdf = traceutils.reformat_morph_values(sdf)
    run_info = labels_dset['run_info'][()]

    return run_info, sdf
   


def load_traces(datakey, experiment, traceid='traces001',
                response_type='dff', 
                responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
                redo_stats=False, n_processes=1, 
                rootdir='/n/coxfs01/2p-data'):
    '''
    redo_stats: use carefully, will re-run responsivity test if True
   
    To return ALL selected cells, set responsive_test to None
    '''
    #experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] and int(session)<20190512) else experiment
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    soma_fpath = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovnum,
                                    '*%s_static' % (experiment), 'traces', '%s*' % traceid,
                                    'data_arrays', 'np_subtracted.npz'))[0]
 
    # Load experiment neural data
    traces, labels, sdf, run_info = traceutils.load_dataset(soma_fpath, trace_type=response_type,create_new=False)

    # Get responsive cells
    if responsive_test is not None:
        responsive_cells, ncells_total = get_responsive_cells(datakey, run=experiment,
                                            responsive_test=responsive_test, 
                                            responsive_thr=responsive_thr,
                                            create_new=redo_stats)
        #print("%i responsive" % len(responsive_cells))
        if responsive_cells is None:
            print("NO LOADING")
            return None, None, None
        traces = traces[responsive_cells]
    else:
        responsive_cells = [c for c in traces.columns if hutils.isnumber(c)]

    return traces[responsive_cells], labels, sdf



def process_traces(raw_traces, labels, trace_type='zscore', 
                    response_type='zscore', trial_epoch='stimulus',
                    nframes_post_onset=None):
    '''
    Calculate raw traces into dff traces (or zscore) and calculate trial metrics
    
    Inputs: 
    
    raw_traces: pd.DataFrame
        Raw traces (i.e., corrected)
    
    labels: pd.DataFrame
        Frame labels for all trials

    response_type: str
        zscore:  (F - mean_bas)/std_bas
        dff, (F - mean_bas)/mean_bas
        snr: mean_stim/mean_bas
        mean: mean_stim (i.e., metric value)

    trial_epoch: str
        stimulus: avg over stim period
        plushalf: avg over stim * 1.5
        pre: baseline only

    nframes_post_onset: int
        overrides trial_epoch 

    '''
    print("--- processed traces: %s" % response_type)
    # Get stim onset frame: 
    stim_on_frame = labels['stim_on_frame'].unique()
    assert len(stim_on_frame) == 1, \
        "---[stim_on]: > 1 stim onset found: %s" % str(stim_on_frame)
    stim_on_frame = stim_on_frame[0]

    # Get n frames stimulus on:
    nframes_on = labels['nframes_on'].unique()
    assert len(nframes_on) == 1, \
        "---[nframes_on]: > 1 stim dur found: %s" % str(nframes_on)
    nframes_on = nframes_on[0]

    if nframes_post_onset is not None:
        metric_ixs = np.arange(stim_on_frame, stim_on_frame+nframes_on+nframes_post_onset).astype(int)
    else:
        if trial_epoch == 'stimulus':
            metric_ixs = np.arange(stim_on_frame, stim_on_frame+nframes_on).astype(int)
        elif trial_epoch == 'plushalf':
            metric_ixs = np.arange(stim_on_frame, stim_onframe+nframes_on*1.5).astype(int)

        elif trial_epoch in ['pre', 'bas']:
            metric_ixs = np.arange(0, stim_on_frame).astype(int)
        else:
            metric_ixs = np.arange(stim_on_frame, stim_on_frame+nframes_on).astype(int)


    traces_list = []
    metrics_list = []
    #snrs_list = []
    for (trial, cfg), tmat in labels.groupby(['trial', 'config']):

        # Get traces using current trial's indices: divide by std of baseline
        raw_ = raw_traces.iloc[tmat.index]
        bas_std = raw_.iloc[0:stim_on_frame].std(axis=0)
        bas_mean = raw_.iloc[0:stim_on_frame].mean(axis=0)
        stim_mean = raw_.iloc[metric_ixs].mean(axis=0)

        if trace_type == 'zscore':
            curr_traces = pd.DataFrame(raw_).subtract(bas_mean)\
                            .divide(bas_std, axis='columns')
        elif trace_type == 'dff':
            curr_traces = pd.DataFrame(raw_).subtract(bas_mean)\
                            .divide(bas_mean, axis='columns')
        else:
            curr_traces = pd.DataFrame(raw_).subtract(bas_mean)

        # Also get zscore (single value) for each trial:
        if response_type=='zscore':
            curr_metrics = (stim_mean-bas_mean)/bas_std
        elif response_type == 'snr':
            curr_metrics = stim_mean/bas_mean
        elif response_type == 'mean':
            curr_metrics = stim_mean
        elif response_type == 'dff':
            curr_metrics = (stim_mean-bas_mean) / bas_mean 
        else:           
            curr_metrics = stim_mean

        traces_list.append(curr_traces) 
        curr_metrics['config'] = cfg
        metrics_list.append(curr_metrics)

    processed_traces = pd.concat(traces_list, axis=0)
    trial_metrics = pd.concat(metrics_list, axis=1).T 
    # cols=rois, rows = trials
    roi_list = raw_traces.columns.tolist()
    trial_metrics[roi_list] = trial_metrics[roi_list].astype(float)

    return processed_traces, trial_metrics 



def get_responsive_cells(datakey, run=None, traceid='traces001',
                         response_type='dff',create_new=False, n_processes=1,
                         responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
                         rootdir='/n/coxfs01/2p-data', verbose=False, return_stats=False):
    '''Load specified responsivity test results.'''
    rstats=None; roi_list=None; nrois_total=None;
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    fov = 'FOV%i_zoom2p0x' % fovnum
    run_name = 'gratings' if (('rfs' in run or 'rfs10' in run) \
                            and int(session)<20190511) else run 
    roi_list=None; nrois_total=None;
    rname = run if 'combined' in run else 'combined_%s_' % run
    traceid_dir =  glob.glob(os.path.join(rootdir, animalid, session, 
                        'FOV%i_*' % fovnum, '%s*' % rname, \
                        'traces/%s*' % traceid))[0]        
    stat_dir = os.path.join(traceid_dir, \
                        'summary_stats', responsive_test)
    if not os.path.exists(stat_dir):
        os.makedirs(stat_dir) 
    # move old dir
    if responsive_test=='nstds':
        stats_fpath = glob.glob(os.path.join(stat_dir, 
                        '%s-%.2f_result*.pkl' \
                        % (responsive_test, n_stds)))
    else:
        stats_fpath = glob.glob(os.path.join(stat_dir, 'roc_result*.pkl'))

    # Check if test=nstds and need to make
    if responsive_test=='nstds' and len(stats_fpath)==0:
        print("Running NSTDS")
        create_new=True

    if create_new and run!='retino': #(('gratings' in run) or ('blobs' in run)):
        print("... calculating responsive, might take awhile (%s)" % (datakey))
        try:
            if responsive_test=='ROC':
                print("... [%s] NOT implemented, need to run bootstrap" % datakey) 
                #DOING BOOT - run: %s" % run) 
#                bootstrap_roc_func(animalid, session, fov, traceid, run, 
#                            trace_type='corrected', rootdir=rootdir,
#                            n_processes=n_processes, plot_rois=True, n_iters=1000)
            elif responsive_test=='nstds':
                fdf = calculate_nframes_above_nstds(animalid, session, fov, 
                            run=run, traceid=traceid, n_stds=n_stds, 
                            #response_type=response_type, 
                            n_processes=n_processes, rootdir=rootdir, 
                            create_new=True)

            print('... finished responsivity test (%s)' % (datakey))
        except Exception as e:
            traceback.print_exc()
            print("[%s] ERROR finding responsive cells" % datakey)
            return None, None 

    if responsive_test=='nstds':
        stats_fpath = glob.glob(os.path.join(stat_dir, 
                            '%s-%.2f_result*.pkl' % (responsive_test, n_stds)))
    else:
        stats_fpath = glob.glob(os.path.join(stat_dir, 'roc_result*.pkl'))

    try:
        #stats_fpath = glob.glob(os.path.join(stats_dir, '*results*.pkl'))
        #assert len(stats_fpath) == 1, "Stats results paths: %s" % str(stats_fpath)
        with open(stats_fpath[0], 'rb') as f:
            if verbose:
                print("... loading stats")
            rstats = pkl.load(f, encoding='latin1')
        # print("...loaded")        
        if responsive_test == 'ROC':
            roi_list = [r for r, res in rstats.items() if res['pval'] < responsive_thr]
            nrois_total = len(rstats.keys())
        elif responsive_test == 'nstds':
            assert n_stds == rstats['nstds'], "... incorrect nstds, need to recalculate"
            #print rstats
            roi_list = [r for r in rstats['nframes_above'].columns \
                            if any(rstats['nframes_above'][r] > responsive_thr)]
            nrois_total = rstats['nframes_above'].shape[-1]
    except Exception as e:
        print(e)
        print(stats_fpath)
        traceback.print_exc()

    if verbose:
        print("... %i of %i cells responsive" % (len(roi_list), nrois_total))

    if return_stats:
        return roi_list, nrois_total, rstats
    else:
        return roi_list, nrois_total
 
def calculate_nframes_above_nstds(animalid, session, fov, run=None, 
                        traceid='traces001',
                         #response_type='dff', 
                        n_stds=2.5, create_new=False,
                         n_processes=1, rootdir='/n/coxfs01/2p-data'):
    '''Calculate N frames above baseline*n_stds (default=2.5). Saves pd.DataFrame()'''
    if 'combined' in run:
        rname = run
    else:
        rname = 'combined_%s_' % run

    traceid_dir =  glob.glob(os.path.join(rootdir, animalid, session, 
                                fov, '%s*' % rname, 'traces/%s*' % traceid))[0]        
    stat_dir = os.path.join(traceid_dir, 'summary_stats', 'nstds')
    if not os.path.exists(stat_dir):
        os.makedirs(stat_dir) 
    results_fpath = os.path.join(stat_dir, 'nstds-%.2f_results.pkl' % n_stds)
    
    calculate_frames = False
    if os.path.exists(results_fpath) and create_new is False:
        try:
            with open(results_fpath, 'rb') as f:
                results = pkl.load(f, encoding='latin1')
            assert results['nstds'] == n_stds, \
                    "... different nstds requested. Re-calculating"
            framesdf = results['nframes_above']            
        except Exception as e:
            calculate_frames = True
    else:
        calculate_frames = True
   
    if calculate_frames:
        print("... Testing responsive (n_stds=%.2f)" % n_stds)
        # Load data
        soma_fpath = glob.glob(os.path.join(traceid_dir, 
                                    'data_arrays', 'np_subtracted.npz'))[0]
        traces, labels, sdf, run_info = traceutils.load_dataset(soma_fpath, 
                                            trace_type='corrected', 
                                            add_offset=True, 
                                            make_equal=False) 
        ncells_total = traces.shape[-1]        
        # Calculate N frames 
        framesdf = pd.concat([find_n_responsive_frames(traces[roi], labels, 
                                n_stds=n_stds) for roi in range(ncells_total)], axis=1)
        results = {'nframes_above': framesdf, 'nstds': n_stds}
        # Save    
        with open(results_fpath, 'wb') as f:
            pkl.dump(results, f, protocol=2)
        print("... Saved: %s" % results_fpath) #os.path.split(results_fpath)[-1])
 
    return framesdf

def find_n_responsive_frames(roi_traces, labels, n_stds=2.5):
    roi = roi_traces.name
    stimon = labels['stim_on_frame'].unique()[0]
    nframes_on = labels['nframes_on'].unique()[0]
    rtraces = pd.concat([pd.DataFrame(data=roi_traces.values, 
                        columns=['values'], index=labels.index), labels], axis=1)

    n_resp_frames = {}
    for config, g in rtraces.groupby(['config']):
        tmat = np.vstack(g.groupby(['trial'])['values'].apply(np.array))
        tr = tmat.mean(axis=0)
        b_mean = np.nanmean(tr[0:stimon])
        b_std = np.nanstd(tr[0:stimon])
        #threshold = abs(b_mean) + (b_std*n_stds)
        #nframes_trial = len(tr[0:stimon+nframes_on])
        #n_frames_above = len(np.where(np.abs(tr[stimon:stimon+nframes_on]) > threshold)[0])
        thr_lo = abs(b_mean) - (b_std*n_stds)
        thr_hi = abs(b_mean) + (b_std*n_stds)
        nframes_above = len(np.where(np.abs(tr[stimon:stimon+nframes_on]) > thr_hi)[0])
        nframes_below = len(np.where(np.abs(tr[stimon:stimon+nframes_on]) < thr_lo)[0])
        n_resp_frames[config] = nframes_above + nframes_below

    #rconfigs = [k for k, v in n_resp_frames.items() if v>=min_nframes]
    #[stimdf['sf'][cfg] for cfg in rconfigs]
    cfs = pd.DataFrame(n_resp_frames, index=[roi]).T #columns=[roi])
   
    return cfs
 




def aggregate_and_save(experiment, traceid='traces001', 
                       response_type='dff', epoch='stimulus',
                       responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, 
                       create_new=False, redo_stats=False, redo_fov=False,
                       always_exclude=['20190426_JC078'], n_processes=1,
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    create_new: remake aggregate file
    redo_stats: for each loaded FOV, re-calculate stats 
    redo_fov: create new neuraldf (otherwise just loads existing)
    '''
    #if experiment=='gratings':
    #    always_exclude.append('20190517_JC083')

    #### Load mean trial info for responsive cells
    data_dir = os.path.join(aggregate_dir, 'data-stats')
    sdata = get_aggregate_info(traceid=traceid)
    
    #### Get DATA   
    data_outfile = get_aggregate_filepath(experiment, traceid=traceid, 
                        response_type=response_type, epoch=epoch,
                        responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, n_stds=n_stds,
                        aggregate_dir=aggregate_dir)
    data_desc = os.path.splitext(os.path.split(data_outfile)[-1])[0]
    print(data_desc)
    if not os.path.exists(data_outfile):
        create_new=True

    no_stats = []
    DATA = {}
    if create_new:
        print("Getting data: %s" % experiment)
        print("Saving data to %s" % data_outfile)
        dsets = sdata[sdata['experiment']==experiment].copy()
        for (animalid, session, fovnum), g in dsets.groupby(['animalid', 'session', 'fovnum']):
            datakey = '%s_%s_fov%i' % (session, animalid, fovnum)
            if '%s_%s' % (session, animalid) in always_exclude:
                continue 
            else:
                print(datakey)
            mean_responses = get_neuraldf(datakey, experiment, 
                                traceid=traceid, 
                                response_type=response_type, epoch=epoch,
                                responsive_test=responsive_test, 
                                responsive_thr=responsive_thr, n_stds=n_stds,
                                create_new=redo_fov, redo_stats=any([redo_fov, redo_stats]))          
            if mean_responses is None:
                print("NO stats, rerun: %s" % datakey)
                no_stats.append(datakey)
                continue
            DATA[datakey] = mean_responses

        # Save
        with open(data_outfile, 'wb') as f:
            pkl.dump(DATA, f, protocol=2)
        print("Done!")

    print("There were %i datasets without stats:" % len(no_stats))
    for d in no_stats:
        print(d)
    
    print("Saved aggr to: %s" % data_outfile)

    return data_outfile


def get_common_cells_from_dataframes(NEURALDATA, RFDATA):
    ndf_list=[]
    rdf_list=[]
    for (visual_area, datakey), rfdf in RFDATA.groupby(['visual_area', 'datakey']):
        rf_rois = rfdf['cell'].unique()
        if isinstance(NEURALDATA, pd.DataFrame):
            neuraldf = NEURALDATA[(NEURALDATA['visual_area']==visual_area)
                                & (NEURALDATA['datakey']==datakey)]
            blob_rois = neuraldf['cell'].unique()
            common_rois = np.intersect1d(blob_rois, rf_rois)
            new_neuraldf = neuraldf[neuraldf['cell'].isin(common_rois)]
        else:
            if 'V1' in NEURALDATA.keys():
                neuraldf = NEURALDATA[visual_area][datakey]
            else:
                neuraldf = NEURALDATA[datakey] 
            blob_rois = neuraldf['cell'].unique()
            common_rois = np.intersect1d(blob_rois, rf_rois)
            new_neuraldf = neuraldf[common_rois]
            new_neuraldf['config'] = neuraldf['config']
            
        ndf_list.append(new_neuraldf)
        new_rfdf = rfdf[rfdf['cell'].isin(common_rois)]
        rdf_list.append(new_rfdf)
    N = pd.concat(ndf_list, axis=0)
    R = pd.concat(rdf_list, axis=0)

    return N, R



def cells_in_experiment_df(assigned_cells, rfdf):
    '''
    Return df of assigned cells that are included in neural data df (rfdf).
    '''
    if isinstance(rfdf, dict):
        rfdf = neuraldf_dict_to_dataframe(rfdf) #, response_type='response'):

    updated_cells = pd.concat([assigned_cells[(assigned_cells['visual_area']==v) 
                              & (assigned_cells['datakey']==dk) 
                              & (assigned_cells['cell'].isin(g['cell'].unique()))] \
                        for (v, dk), g in rfdf.groupby(['visual_area', 'datakey'])])
    return updated_cells




def main(options):
    opts = extract_options(options)
    experiment = opts.experiment
    traceid = opts.traceid
    response_type = opts.response_type
    responsive_test = None if opts.responsive_test in ['None', 'none', None] else opts.responsive_test
    responsive_thr = 0 if responsive_test is None else float(opts.responsive_thr) 
    n_stds = float(opts.nstds_above) if responsive_test=='nstds' else 0.
    create_new = opts.create_new
    epoch = opts.epoch
    n_processes = int(opts.n_processes)
    redo_stats = opts.redo_stats 
    redo_fov = opts.redo_fov

    run_aggregate = opts.aggregate
    aggregate_dir = opts.aggregate_dir
    fov_type=opts.fov_type
    state=opts.state

    do_metrics = opts.do_metrics
    if run_aggregate: 
        data_outfile = aggregate_and_save(experiment, traceid=traceid, 
                                       response_type=response_type, epoch=epoch,
                                       responsive_test=responsive_test, 
                                       n_stds=n_stds,
                                       responsive_thr=responsive_thr, 
                                       create_new=any([create_new, redo_stats, redo_fov]),
                                       n_processes=n_processes,
                                       redo_stats=redo_stats, redo_fov=redo_fov)

    elif do_metrics:
         save_trial_metrics_cycle(experiment, traceid=traceid, 
                                       response_type=response_type, epoch=epoch,
                                       responsive_test=responsive_test, 
                                       n_stds=n_stds,
                                       responsive_thr=responsive_thr, 
                                       create_new=any([create_new, redo_stats, redo_fov]),
                                       n_processes=n_processes,
                                       redo_stats=redo_stats, redo_fov=redo_fov)
 
    return


def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-G', '--aggr', action='store', dest='aggregate_dir', default='/n/coxfs01/julianarhee/aggregate-visual-areas', 
                      help='aggregate analysis dir [default: aggregate-visual-areas]')
    parser.add_option('--zoom', action='store', dest='fov_type', default='zoom2p0x', 
                      help="fov type (zoom2p0x)") 
    parser.add_option('--state', action='store', dest='state', default='awake', 
                      help="animal state (awake)") 
  
    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") 

    choices_e = ('stimulus', 'firsthalf', 'plushalf', 'baseline')
    default_e = 'stimulus'
    parser.add_option('-e', '--epoch', action='store', dest='epoch', 
            default=default_e, type='choice', choices=choices_e,
            help="Trial epoch to average, choices: %s. (default: %s" % (choices_e, default_e))


    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")

    choices_c = ('all', 'ROC', 'nstds', None, 'None')
    default_c = 'nstds'
    parser.add_option('-R', '--responsive_test', action='store', dest='responsive_test', 
            default=default_c, type='choice', choices=choices_c,
            help="Responsive test, choices: %s. (default: %s" % (choices_c, default_c))

    parser.add_option('--thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test thr (default: 0.05 for ROC)")
    parser.add_option('-d', '--response', action='store', dest='response_type', default='dff', 
                      help="response type (default: dff)")
    parser.add_option('--nstds', action='store', dest='nstds_above', default=2.5, 
                      help="only for test=nstds, N stds above (default: 2.5)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="flag to create new")
    
    parser.add_option('-X', '--exclude', action='store', dest='always_exclude', 
                      default=['20190426_JC078'],
                      help="Datasets to exclude bec incorrect or overlap")

    parser.add_option('-n', '--nproc', action='store', dest='n_processes', 
                      default=1,
                      help="N processes (default=1)")
    parser.add_option('--redo-stats', action='store_true', dest='redo_stats', 
                      default=False,
                      help="Flag to redo tests for responsivity")
    parser.add_option('--redo-fov', action='store_true', dest='redo_fov', 
                      default=False,
                      help="Flag to recalculate neuraldf from traces")

    parser.add_option('-i', '--animalid', action='store', dest='animalid', default=None,
                      help="animalid (e.g., JC110)")
    parser.add_option('-S', '--session', action='store', dest='session', default=None,
                      help="session (format: YYYYMMDD)")
    parser.add_option('-A', '--fovnum', action='store', dest='fovnum', default=None,
                      help="fovnum (default: all fovs)")

    parser.add_option('--all',  action='store_true', dest='aggregate', default=False,
                      help="Set flag to cycle thru ALL dsets")

    parser.add_option('--metrics',  action='store_true', dest='do_metrics', default=False,
                      help="Set flag to cycle thru and save all metrics for each dset")

    (options, args) = parser.parse_args(options)

    return options


if __name__ == '__main__':
    main(sys.argv[1:])


