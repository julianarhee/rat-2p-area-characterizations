#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:14:03 2020

@author: julianarhee
"""
import sys
import re
import os
import glob
import json
import copy

import traceback
import datetime
import numpy as np
import pandas as pd
import dill as pkl

#from pipeline.python.classifications import aggregate_data_stats as aggr
import analyze2p.aggregate_datasets as aggr
import analyze2p.utils as hutils

# ===================================================================
# Data loading 
# ====================================================================
def create_parsed_traces_id(alignment_type='ALIGN', 
                             snapshot=391800):
    '''
    Common name for pupiltraces datafiles.
    '''
    #fname = 'traces_%s_align-%s_%s_snapshot-%i' % (feature_name, alignment_type, experiment, snapshot)
    fname = 'snapshot-%i_%s' % (snapshot, alignment_type)

    return fname

def create_trial_metrics_id(trial_epoch='stimulus', snapshot=391800):
    '''
    Common name for trial metrics datafiles.
    '''
    #fname = 'traces_%s_align-%s_%s_snapshot-%i' % (feature_name, alignment_type, experiment, snapshot)
    fname = 'snapshot-%i_%s_metrics' % (snapshot, trial_epoch)

    return fname


def create_aggr_traces_id(experiment='EXP', alignment_type='ALIGN', 
                            feature_name='FEAT', snapshot=391800):
    '''
    Common name for pupiltraces datafiles.
    '''
    #fname = 'traces_%s_align-%s_%s_snapshot-%i' % (feature_name, alignment_type, experiment, snapshot)
    fname = '%s_snapshot-%i_traces_%s' % (experiment, snapshot, alignment_type)

    return fname


def create_aggr_metrics_id(experiment, trial_epoch, snapshot):
    '''Name for some per-trial metric calculated on traces'''

    #fname = 'metrics_%s_%s_%s_snapshot-%i' \
    #            % (experiment, feature_name, trial_epoch, snapshot)
    fname = '%s_snapshot-%i_metrics_%s' \
                % (experiment, snapshot, trial_epoch )

    return fname

def save_traces_and_params(datakey, experiment, ptraces, params, 
                        rootdir='/n/coxfs01/2p-data'):
    '''
    Save parsed pupil traces (aligned to trials) for FOV.
    '''
    # Set output dir
    session, animalid, fovnum = hutils.split_datakey_str(datakey)                 
    experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment

    rundir = glob.glob(os.path.join(rootdir, animalid, session, 
                    'FOV%i_*' % fovnum, 'combined_%s_static' % experiment_name))[0]
    dst_dir = os.path.join(rundir, 'facetracker')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # Setup output files. Create parse id.
    parse_id = create_parsed_traces_id( alignment_type=params['alignment_type'],
                                       snapshot=params['snapshot']) 

    params_fpath = os.path.join(dst_dir, '%s_params.json' % parse_id)
    results_fpath = os.path.join(dst_dir, '%s_traces.pkl' % parse_id)

    # Save params
    with open(params_fpath, 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)
    # Save aligned traces
    with open(results_fpath, 'wb') as f:
        pkl.dump(ptraces, f, protocol=pkl.HIGHEST_PROTOCOL)
    print("    saved traces, %s" % datakey)

    return

def load_fov_metrics(datakey, experiment,  
                    trial_epoch='stimulus', snapshot=391800,
                    rootdir='/n/coxfs01/2p-data'):
    df_ = None
    params_ = None
    # Set output dir
    session, animalid, fovnum = hutils.split_datakey_str(datakey)                 
    experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment

    rundir = glob.glob(os.path.join(rootdir, animalid, session, 
                    'FOV%i_*' % fovnum, 'combined_%s_static' % experiment_name))[0]
    dst_dir = os.path.join(rundir, 'facetracker')
    metric_id = create_trial_metrics_id(snapshot=snapshot, 
                                trial_epoch=trial_epoch)
    
    results_fpath = os.path.join(dst_dir, '%s.pkl' % metric_id)
    params_fpath = os.path.join(dst_dir, '%s_params.json' % metric_id)

    with open(results_fpath, 'rb') as f:
        df_ = pkl.load(f)
    with open(params_fpath, 'r') as f:
        params_ = json.load(f)

    return df_, params_


def save_fov_metrics(datakey, experiment, df_, params_, 
                    trial_epoch='stimulus', snapshot=391800,
                    rootdir='/n/coxfs01/2p-data'):
    # Set output dir
    session, animalid, fovnum = hutils.split_datakey_str(datakey)                 
    experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment

    rundir = glob.glob(os.path.join(rootdir, animalid, session, 
                    'FOV%i_*' % fovnum, 'combined_%s_static' % experiment_name))[0]
    dst_dir = os.path.join(rundir, 'facetracker')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    metric_id = create_trial_metrics_id(snapshot=snapshot, 
                                trial_epoch=trial_epoch)
    
    results_fpath = os.path.join(dst_dir, '%s.pkl' % metric_id)
    params_fpath = os.path.join(dst_dir, '%s_params.json' % metric_id)

    with open(results_fpath, 'wb') as f:
        pkl.dump(df_, f, protocol=2)
    with open(params_fpath, 'w') as f:
        json.dump(params_, f, indent=4, sort_keys=True)
    print("    saved metrics, %s" % datakey)

    return

 
def load_fov_traces(datakey, experiment, alignment_type='trial', 
                    snapshot=391800, rootdir='/n/coxfs01/2p-data'):
    '''
    Load pupil traces for one dataset.
    results & params are created and saved in ./arousal/parse_face_data.py
    '''
    ptraces=None; params=None;

    # Set output dir
    session, animalid, fovnum = hutils.split_datakey_str(datakey)                 
    experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment

    rundir = glob.glob(os.path.join(rootdir, animalid, session, 
                    'FOV%i_*' % fovnum, 'combined_%s_static' % experiment_name))[0]
    dst_dir = os.path.join(rundir, 'facetracker')
    # Setup output files. Create parse id.
    parse_id = create_parsed_traces_id(alignment_type=alignment_type,
                                       snapshot=snapshot) 
    params_fpath = os.path.join(dst_dir, '%s_params.json' % parse_id)
    results_fpath = os.path.join(dst_dir, '%s_traces.pkl' % parse_id)

    if not os.path.exists(results_fpath):
        print("    results do not exist:\n    %s" % results_fpath)
        return None, None

    try:
        # load results
        with open(results_fpath, 'rb') as f:
            ptraces = pkl.load(f) #, encoding='latin1')
        assert 'stim_dur_ms' in ptraces.columns.tolist(), \
                        "No stim dur for FOV."
        # load params
        with open(params_fpath, 'r') as f:
            params = json.load(f)
    except Exception as e:
        print("ERROR: %s" % datakey)
        return None, None    

    return ptraces, params


def aggregate_traces(experiment, traceid='traces001', 
                realign=False, recombine=False, create_new=False,
                snapshot=391800, alignment_type='trial',
                iti_pre=1., iti_post=1., feature_list=['pupil'],
                verbose=False, return_missing=False,
                exclude=['20190517_JC083_fov1'],
                rootdir='/n/coxfs01/2p-data', fov_type='zoom2p0x', state='awake',
                eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp',
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Load or create AGGREGATED dict of parsed pupuil traces.
        Saves to: <aggregate_dir>/behavior-state/<traces_fname>.pkl
        Calls load_fov_traces()
   
    create_new: bool
        Set TRUE to recreate aggregated .pkl file with ALL traces.
    
    realign: bool
        Set TRUE to realign traces to trials.

    recombine: bool
        Set TRUE to re-aggregate run data for each dataset.

    Returns
    
    aggr_traces: dict
        keys, datakey
        results, parsed pupil traces (pd.DataFrames)

    '''
    print("~~~~~ Aggregating pupil traces. ~~~~~~")
    # Set output dir for aggregated datafile
    if not os.path.exists(os.path.join(aggregate_dir, 'behavior-state')):
        os.makedirs(os.path.join(aggregate_dir, 'behavior-state'))

    exclude_for_now = exclude_missing_data(experiment)

    missing_traces=[]
    aggr_traces={}; aggr_params={};
    # Get all datasets
    sdata = aggr.get_aggregate_info(traceid=traceid, 
                            fov_type=fov_type, state=state, 
                            return_cells=False)
    edata = sdata[sdata['experiment']==experiment].copy()
    all_dkeys = edata['datakey'].unique()

    if create_new is False:
        try:
            aggr_traces, aggr_params, missing_traces = load_traces(experiment, 
                                                alignment_type=alignment_type, 
                                                snapshot=snapshot,
                                                aggregate_dir=aggregate_dir, 
                                                return_missing=True)

            existing = aggr_traces.keys()
            dsets_to_run = edata[~edata.datakey.isin(existing)]['datakey'].unique()
            print("Loaded fov-traces for %i of %i datasets" \
                    % (len(existing), len(all_dkeys)))
    
        except Exception as e:
            create_new=True 
    else:
        dsets_to_run=copy.copy(all_dkeys)
    
    # Loop thru datakeys and load parsed trials or parse them.
    for datakey in dsets_to_run: #, g in dsets_to_run.groupby(['datakey']):
        #if (iti_pre+iti_post)>1 and (datakey in exclude):
        #    print(" - skipping %s - " % datakey)
        #    continue
        if datakey in exclude_for_now:
            print("    excluding %s for now" % datakey)
            continue

        if realign is False and recombine is False:
            try:
                fov_traces, fov_params = load_fov_traces(datakey, experiment,
                                    alignment_type=alignment_type,
                                    snapshot=snapshot, rootdir=rootdir)
                assert fov_traces is None
            except Exception as e:
                print("Error loading traces: %s, %s" % (datakey, experiment))
                realign=True
        if realign or recombine:
            # Reparse
            fov_traces, fov_params = parse_pose_data(datakey, experiment, 
                                    traceid=traceid, 
                                    iti_pre=iti_pre, iti_post=iti_post, 
                                    feature_list=feature_list,
                                    alignment_type=alignment_type, 
                                    snapshot=snapshot, verbose=verbose,
                                    rootdir=rootdir,
                                    eyetracker_dir=eyetracker_dir,
                                    realign=realign, recombine=recombine) 
            # save_traces_and_params(datakey, experiment, fov_traces, fov_params)
        if fov_traces is None:
            print("... missing traces: %s" % datakey)
            missing_traces.append(datakey)
            continue
        #### Add to dict
        aggr_traces[datakey] = fov_traces
        aggr_params[datakey] = fov_params

        if len(aggr_traces.keys())>0:
            save_traces(aggr_traces, aggr_params, experiment,
                        alignment_type=alignment_type, snapshot=snapshot,
                        aggregate_dir=aggregate_dir)

    print("Aggregated pupil traces. Missing %i datasets." % len(missing_traces))
    if verbose:
        for m in missing_traces:
            print(m)

    if return_missing:
        return aggr_traces, aggr_params, missing_traces
    else:
        return aggr_traces, aggr_params


def load_traces(experiment, alignment_type='stimulus', snapshot=391800, 
                return_missing=False,
                traceid='traces001', fov_type='zoom2p0x', state='awake',
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    """
    Load AGGREGATED pupiltraces dict (from all FOVS w dlc)
    keys : datakeys (SESSION_ANIMALID_FOV)
    vals: dataframe of config, trial, pupil metric
    """
    aggr_traces=None
    agg_params=None
    missing_traces=None
    #### Loading existing extracted pupil data
    traces_fname = create_aggr_traces_id(experiment=experiment, 
                            alignment_type=alignment_type, snapshot=snapshot) 
    traces_fpath = os.path.join(aggregate_dir, \
                    'behavior-state', '%s.pkl' % traces_fname)  
    params_fpath = os.path.join(aggregate_dir, \
                    'behavior-state', '%s_params.json' % traces_fname)  
    try: 
        # This is a dict, keys are datakeys
        with open(traces_fpath, 'rb') as f:
            aggr_traces = pkl.load(f, encoding='latin1')
        print(">>>> Loaded aggregated pupil traces.")
        with open(params_fpath, 'r') as f:
            aggr_params = json.load(f)

    except Exception as e:
        traceback.print_exc()
        if not os.path.exists(traces_fpath):
            print( "Aggr. traces not found:\n    %s" % traces_fpath)

     
    if return_missing:
        sdata = aggr.get_aggregate_info(return_cells=False,
                            traceid=traceid, fov_type=fov_type, state=state)
        edata = sdata[(sdata['experiment'] == experiment)]
        missing_traces = [ e for e in edata['datakey'].unique() \
                            if e not in aggr_traces.keys() ]
        return aggr_traces, aggr_params, missing_traces
    else:
        return aggr_traces, aggr_params


def save_traces(aggr_traces, aggr_params, experiment, 
                alignment_type='trial', snapshot=391800,
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    #### Output files
    traces_fname = create_aggr_traces_id(experiment=experiment, 
                            alignment_type=alignment_type, snapshot=snapshot) 
    traces_fpath = os.path.join(aggregate_dir, \
                    'behavior-state', '%s.pkl' % traces_fname)  
    params_fpath = os.path.join(aggregate_dir, \
                    'behavior-state', '%s_params.json' % traces_fname)  
    
    with open(traces_fpath, 'wb') as f:
        pkl.dump(aggr_traces, f, protocol=2) #pkl.HIGHEST_PROTOCOL)
    with open(params_fpath, 'w') as f:
        json.dump(aggr_params, f, indent=4) #pkl.HIGHEST_PROTOCOL)


    return



# AGGREGATE STUFF
def save_dataframes(aggr_dfs, aggr_params, experiment,
                snapshot=391800, trial_epoch='stimulus',
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    fname = create_aggr_metrics_id(experiment, trial_epoch, snapshot)
    df_fpath = os.path.join(aggregate_dir, \
                                    'behavior-state', '%s.pkl' % fname)
    params_fpath = os.path.join(aggregate_dir, \
                                    'behavior-state', '%s_params.json' % fname)

    # Save
    with open(df_fpath, 'wb') as f:
        pkl.dump(aggr_dfs, f, protocol=2)# pkl.HIGHEST_PROTOCOL)
    print("---> Saved aggr dataframes: %s" % df_fpath) 

    with open(params_fpath, 'w') as f:
        json.dump(aggr_params, f, indent=4, sort_keys=True)

    return 


def load_dataframes(experiment, snapshot=391800, trial_epoch='stimulus',
                return_missing=False, traceid='traces001', 
                fov_type='zoom2p0x', state='awake', 
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Load AGGREGATED pupil dataframes (per-trial metric).

    trial_epoch: str
        'pre': Use PRE-stimulus period for response metric.
        'stimulus': Use stimulus period
        'all': Use full trial period
 
    Returns:

    pupildfs: dict
        keys: datakeys (like MEANS dict)
        values: dataframes (pupildf, trial metrics) 
    '''
    aggr_dfs=None; aggr_params=None;
    missing_metrics=[]

    fname = create_aggr_metrics_id(experiment, trial_epoch, snapshot)
    df_fpath = os.path.join(aggregate_dir, 
                                    'behavior-state', '%s.pkl' % fname)
    params_fpath = os.path.join(aggregate_dir, 
                                    'behavior-state', '%s_params.json' % fname)
    try:
        with open(df_fpath, 'rb') as f:
            aggr_dfs = pkl.load(f)
        print(">>>> Loaded aggregate pupil dataframes.")
    except UnicodeDecodeError:
        with open(df_fpath, 'rb') as f:
            aggr_dfs = pkl.load(f, encoding='latin1')
    except Exception as e:
        print('File not found: %s' % df_fpath)
  
    if aggr_dfs is not None:
        with open(params_fpath, 'r') as f:
            aggr_params = json.load(f) 
 
    if return_missing:
        sdata = aggr.get_aggregate_info(return_cells=False, traceid=traceid, 
                                        fov_type=fov_type, state=state)
        edata = sdata[(sdata['experiment'] == experiment)]
        missing_metrics = [ e for e in edata['datakey'].unique() \
                                if e not in aggr_dfs.keys() ]
        return aggr_dfs, aggr_params, missing_metrics
    else:
        return aggr_dfs, aggr_params


def exclude_missing_data(experiment):
    if experiment=='blobs':
        exclude_for_now = ['20190315_JC070_fov1', 
                           '20190506_JC080_fov1', 
                           '20190501_JC076_fov1']
    elif experiment=='gratings':
        exclude_for_now = ['20190627_JC091_fov1',
                           '20190522_JC089_fov1',
                           '20190612_JC099_fov1',
                           '20190517_JC083_fov1']    
    else:
        exclude_for_now=[]

    return exclude_for_now


def aggregate_dataframes(experiment, trial_epoch='stimulus',
                in_rate=20., out_rate=20., iti_pre=1., iti_post=1.,
                create_new=False, realign=False, recombine=False,
                snapshot=391800, alignment_type='trial', traceid='traces001',
                verbose=False, return_missing=False,
                rootdir='/n/coxfs01/2p-data', 
                fov_type='zoom2p0x', state='awake',
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):   
    '''
    Create AGGREGATED dict of pupil dataframes (per-trial metrics) by cycling.
    (prev called load_pupil_data)
    
    Set realign=False and recombine=False to just load fov traces 
    and calculate trial metrics.

    trial_epoch : (str)
        'pre': Use PRE-stimulus period for response metric.
        'stimulus': Use stimulus period
        'all': Use full trial period
 
    create_new: bool
        Set TRUE to recreate aggregated .pkl file with ALL traces.

    Takes pupil traces, resample, then returns dict of pupil dataframes.
    Durations are in seconds.
  
    realign: bool
        Set TRUE to re-align aggregate traces for each FOV.

    recombine: bool
        Set TRUE to re-combine all pupildata across runs for each FOV.

    Returns:

    aggr_dfs (dict)
        keys: datakeys (like MEANS dict)
        values: dataframes (pupildf, trial metrics)
    '''
    print("~~~~~~~~~~~~ Aggregating pupil dataframes. ~~~~~~~~~~~")

    exclude_for_now = exclude_missing_data(experiment)

    missing_dsets={};
    aggr_dfs={}
    aggr_params={}

    if realign or recombine:
        create_new=True

    missing_metrics=[]
    missing_traces=[]
    if (create_new is False):
        print("Re-aggregating")
        sdata = aggr.get_aggregate_info(traceid=traceid, 
                                fov_type=fov_type, state=state, 
                                return_cells=False)
        edata = sdata[sdata['experiment']==experiment].copy()
        dk_list = edata['datakey'].unique()
        for dk in dk_list:
            if dk in exclude_for_now:
                print("    (%s) Excluding %s" % (experiment, dk))
                continue
            try:
                fovdf, fovparams = load_fov_metrics(dk, experiment,
                                     trial_epoch=trial_epoch, snapshot=snapshot)

                assert fovdf is not None
                aggr_dfs[dk] = fovdf
                aggr_params[dk] = fovparams
            except Exception as e:
                missing_metrics.append(dk)
                continue
            
#            aggr_dfs, aggr_params, missing_dsets = load_dataframes(experiment, 
#                                        snapshot=snapshot, 
#                                        trial_epoch=trial_epoch, 
#                                        aggregate_dir=aggregate_dir, 
#                                        return_missing=True)
#            assert len(aggr_dfs)>0, "No aggregated dfs. Creating new."
#        except Exception as e:
#            create_new=True

    if create_new or len(missing_metrics)>0:
        # Load traces
        remake_traces = (realign is True or recombine is True)
        aggr_traces, aggr_traces_params, missing_traces = aggregate_traces(
                                        experiment, 
                                        alignment_type=alignment_type, 
                                        iti_pre=iti_pre, iti_post=iti_post,
                                        snapshot=snapshot, 
                                        traceid=traceid, 
                                        create_new=remake_traces,
                                        realign=realign, recombine=recombine,
                                        return_missing=True)
        # Calculate per-trial metrics 
        print("Calculating trial metrics for missing datasets")
        assert aggr_traces is not None
        for dk, fov_traces in aggr_traces.items():
            if dk not in missing_metrics and dk in aggr_dfs.keys():
                continue
            if dk in exclude_for_now:
                continue
            fov_params = aggr_traces_params[dk].copy()
            df_, params_ = parsed_traces_to_metrics(fov_traces, fov_params, 
                                    trial_epoch=trial_epoch,
                                    in_rate=in_rate, out_rate=out_rate,
                                    iti_pre=iti_pre, iti_post=iti_post)
            # save trial metrics for fov
            save_fov_metrics(dk, experiment, df_, params_,
                    trial_epoch=trial_epoch, snapshot=snapshot)
            aggr_dfs[dk] = df_
            aggr_params[dk] = params_

        # Save aggregate metrics
        save_dataframes(aggr_dfs, aggr_params, experiment,
                            snapshot=snapshot, trial_epoch=trial_epoch,
                            aggregate_dir=aggregate_dir)
        print("Saved aggr. metrics to disk.")

    missing_dsets = {'metrics': np.unique(missing_metrics), 
                     'traces': np.unique(missing_traces)}
    if return_missing:
        return aggr_dfs, aggr_params, missing_dsets
    else:
        return aggr_dfs, aggr_params


def parsed_traces_to_metrics(ptraces, params, in_rate=20., out_rate=20.,
                        trial_epoch='stimulus', iti_pre=1., iti_post=1.):
    '''
    Resample raw (parsed) pupil trace, calculate a single trial metric.

    ptraces: pd.DataFrame
        Raw pupi traces parsed into trials.

    trial_epoch: str
        stimulus:  average over the stimulus period
        pre:  average over iti_pre
        plushalf:  stimulus + stimulus*0.5
        trial: iti_pre thru iti_post

    Returns: pd.DataFrame (also includes pupil_fraction)

    '''
    stim_durs = np.unique([1.0 if round(s/1E3, 1)==1.1 else round(s/1E3, 1) \
                            for s in ptraces['stim_dur_ms'].unique()])
    assert len(stim_durs)==1, "Bad stim durs: %s" % str(stim_durs)
    stim_dur = float(stim_durs[0]) #*1000
    desired_nframes = int((stim_dur + iti_pre + iti_post)*out_rate)
    iti_pre_ms=iti_pre*1000
    new_stim_on = int(round(iti_pre*out_rate))
    nframes_on = int(round(stim_dur*out_rate))

    params.update({'new_stim_on': new_stim_on, 'new_nframes_on': nframes_on, 
                   'trial_epoch': trial_epoch})

    # Resample for exact frame #s
    binned_pupil = resample_pupil_traces(ptraces,
                                in_rate=in_rate, out_rate=out_rate, 
                                min_nframes=desired_nframes, 
                                iti_pre_ms=iti_pre_ms)
    # Calculate single metric per trial
    df_ = calculate_trial_metrics(binned_pupil, trial_epoch=trial_epoch,
                                new_stim_on=new_stim_on, 
                                nframes_on=nframes_on)
#        pupil_r = resample_pupil_traces(ptraces, 
#                                    in_rate=in_rate, 
#                                    out_rate=out_rate, 
#                                    desired_nframes=desired_nframes, 
#                                    feature_name=feature_to_load, #feature_name, 
#                                    iti_pre_ms=iti_pre_ms)
#        pupildf = get_pupil_df(pupil_r, trial_epoch=trial_epoch, 
#                                new_stim_on=new_stim_on, nframes_on=nframes_on)
    if 'pupil_fraction' not in df_.columns:
        pupil_max = df_['pupil_area'].max()
        df_['pupil_fraction'] = df_['pupil_area']/pupil_max
 
    return df_, params


def calculate_trial_metrics(binned_pupil, trial_epoch='pre', new_stim_on=20., nframes_on=20.):
    '''
    Turn resampled pupil traces into reponse vectors
    
    trial_epoch : (str)
        'pre': Use PRE-stimulus period for response metric.
        'stimulus': Use stimulus period
        'all': Use full trial period
    
    new_stim_on: (int)
        Frame index for stimulus start (only needed if trial_epoch is 'pre' or 'stim')
        
    pupil_r : resampled pupil traces (columns are trial, frame, pupil_area, frame_int, frame_ix)
    '''
    if trial_epoch in ['pre', 'plushalf', 'stimulus']:
        if trial_epoch=='pre':
            incl_ixs = np.arange(0, new_stim_on)
        elif trial_epoch=='plushalf': 
            incl_ixs = np.arange(new_stim_on, int(round(new_stim_on+(new_stim_on*0.5))) )
        elif trial_epoch=='stimulus':
            incl_ixs = np.arange(new_stim_on, new_stim_on+nframes_on)
        df_ = pd.concat([g[g['frame_ix'].isin(incl_ixs)].mean() \
                        for t, g in binned_pupil.groupby(['trial'])], axis=1).T
    else:
        df_ = pd.concat([g.mean() \
                    for t, g in binned_pupil.groupby(['trial'])], axis=1).T

    return df_


#def get_pupil_df(pupil_r, trial_epoch='pre', new_stim_on=20., nframes_on=20.):
#    '''
#    Turn resampled pupil traces into reponse vectors
#    
#    trial_epoch : (str)
#        'pre': Use PRE-stimulus period for response metric.
#        'stimulus': Use stimulus period
#        'all': Use full trial period
#    
#    new_stim_on: (int)
#        Frame index for stimulus start (only needed if trial_epoch is 'pre' or 'stim')
#        
#    pupil_r : resampled pupil traces (columns are trial, frame, pupil_area, frame_int, frame_ix)
#    '''
#    if trial_epoch=='pre':
#        pupildf = pd.concat([g[g['frame_ix'].isin(np.arange(0, new_stim_on))].mean(axis=0) \
#                            for t, g in pupil_r.groupby(['trial'])], axis=1).T
#    elif trial_epoch=='stimulus':
#        pupildf = pd.concat([g[g['frame_ix'].isin(np.arange(new_stim_on, new_stim_on+nframes_on))].mean(axis=0) \
#                            for t, g in pupil_r.groupby(['trial'])], axis=1).T
#    else:
#        pupildf = pd.concat([g.mean(axis=0) for t, g in pupil_r.groupby(['trial'])], axis=1).T
#    #print(pupildf.shape)
#
#    return pupildf


def get_dlc_sources(
                    dlc_home_dir='/n/coxfs01/julianarhee/face-tracking',
                    dlc_project='facetracking-jyr-2020-01-25'):

    project_dir = os.path.join(dlc_home_dir, dlc_project)
    video_dir = os.path.join(project_dir, 'videos')
    results_dir = os.path.join(project_dir, 'pose-analysis')

    return results_dir, video_dir


def get_datasets_with_dlc(sdata, dlc_projectid='facetrackingJan25',
                        scorer='DLC_resnet50', iteration=1, shuffle=1, 
                        trainingsetindex=0, snapshot=391800,
                        dlc_home_dir='/n/coxfs01/julianarhee/face-tracking',
                        dlc_project = 'facetracking-jyr-2020-01-25'):
    # This stuff is hard-coded because we only have 1
    #### Set source/dst paths
    #dlc_home_dir = '/n/coxfs01/julianarhee/face-tracking'
    #dlc_project = 'facetracking-jyr-2020-01-25' #'sideface-jyr-2020-01-09'
    dlc_project_dir = os.path.join(dlc_home_dir, dlc_project)

    dlc_video_dir = os.path.join(dlc_project_dir, 'videos')
    dlc_results_dir = os.path.join(dlc_project_dir, 'pose-analysis') # DLC analysis output dir

    #### Training iteration info
    #dlc_projectid = 'facetrackingJan25'
    #scorer='DLC_resnet50'
    #iteration = 1
    #shuffle = 1
    #trainingsetindex=0
    videotype='.mp4'
    #snapshot = 391800 #430200 #20900
    DLCscorer = '%s_%sshuffle%i_%i' % (scorer, dlc_projectid, shuffle, snapshot)
    print("Extracting results from scorer: %s" % DLCscorer)
 
    print("Checking for existing results: %s" % dlc_results_dir)
    dlc_runkeys = list(set([ os.path.split(f)[-1].split('DLC')[0] \
                        for f in glob.glob(os.path.join(dlc_results_dir, '*.h5'))]))
    dlc_analyzed_experiments = ['_'.join(s.split('_')[0:4]) for s in dlc_runkeys]

    # Get sdata indices that have experiments analyzed
    ixs_wth_dlc = [i for i in sdata.index.tolist() 
                    if '%s_%s' % (sdata.loc[i]['datakey'], sdata.loc[i]['experiment']) in dlc_analyzed_experiments]
    dlc_dsets = sdata.iloc[ixs_wth_dlc]

    return dlc_dsets


def get_trialmeta(datakey, experiment, alignment_type='stimulus', 
                   iti_pre=1, iti_post=1, snapshot=391800, create_new=False,
                   rootdir='/n/coxfs01/2p-data',
                   eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'): 
    '''
    Aligns MW trial data to eyetracker frames, wrapper around get_trial_triggers().
    Load previous meta info for alignment, OR create new.
    
    New, saved to:
        ./RUNDIR/facetracker/snapshot-%i_<alignment>.pkl <-- trial metadata
        ./RUNDIR/facetracker/snapshot-%i_<alignment>_params.json <-- params  
    
    ''' 
    trialmeta=None; params=None;
    
    blacklist=['20191018_JC113_fov1_blobs_run5']
    if experiment in ['rfs', 'rfs10']:
        if iti_pre>=0.5 and iti_post>=0.5:
            blacklist.extend(['20190513_JC078_fov1', 
                              '20190511_JC083_fov1',
                              '20190512_JC083_fov1']) # full iti=500ms
    elif experiment=='gratings':
        if iti_pre>=1.0 and iti_post>=1.0:
            blacklist.append('20190517_JC083_fov1')

    print("*******PRE: %.2f, POST: %.2f" % (iti_pre, iti_post))

    if datakey in blacklist:
        return None, None
    
    # load alignment info
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment

    run_dir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum, 
                            'combined_%s_static' % experiment_name))[0]
    dst_dir = os.path.join(run_dir, 'facetracker')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    parse_id = create_parsed_traces_id(alignment_type=alignment_type, 
                                        snapshot=snapshot)
    alignment_fpath = os.path.join(dst_dir, '%s.pkl' % parse_id)
    params_fpath = os.path.join(dst_dir, '%s_params.json' % parse_id)

    # try loading existing
    if not create_new:
        try:
            with open(alignment_fpath, 'rb') as f:
                trialmeta = pkl.load(f)
            with open(params_fpath, 'r') as f:
                params = json.load(f)
            # Check that we have the right params
            assert iti_pre==params['iti_pre'], \
                "Wrong iti_pre (%.1f vs %.1f), rerun" % (iti_pre, params['iti_pre'])
            assert iti_post==params['iti_post'], \
                "Wrong iti_post (%.1f vs %.1f), rerun" % (iti_post, params['iti_post'])
        except Exception as e:
            traceback.print_exc()
            create_new=True

    if create_new:      
        print("... creating trialmeta for pose traces")
        datestr = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S") 
        # Get metadata for facetracker
        trialmeta, missing_dlc = get_trial_triggers(datakey, experiment, 
                                        alignment_type=alignment_type, 
                                        pre_ITI_ms=iti_pre*1000., 
                                        post_ITI_ms=iti_post*1000.,
                                        return_missing=True,
                                        eyetracker_dir=eyetracker_dir)    
        if trialmeta is not None:
            stim_durs = [i for i in np.unique([round(s/1E3, 1) for s \
                        in trialmeta['stim_dur_ms'].unique()]) if i!=1.1]
            assert len(stim_durs)==1, "More than 1 stim dur found: %s" % str(stim_durs)
            params = {'experiment': experiment,
                      'iti_pre': iti_pre, 
                      'iti_post': iti_post,
                      'stim_dur': stim_durs[0],
                      'alignment_type': alignment_type,
                      'dlc_missing_files': missing_dlc, 
                      'snapshot': snapshot,
                      'datetime': datestr, 
                      'raw_src': eyetracker_dir}
            # Save alignment params
            with open(params_fpath, 'w') as f:
                json.dump(params, f, indent=4, sort_keys=True)
            # Save trial meta info
            with open(alignment_fpath, 'wb') as f:
                pkl.dump(trialmeta, f, protocol=2)
        else:
            print("... no metadata (%s)" % datakey)

    return trialmeta, params 


def combine_pose_data(datakey, experiment, feature_list=['pupil'], 
                    snapshot=391800, create_new=False, verbose=False,
                    rootdir='/n/coxfs01/2p-data',
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    '''
    Get pupil traces combined in order across run files *not trials yet.
    TODO:  add other features besides pupil stuff?

    Saves combined pupil traces to:
        ./RUNDIR/facetracker/snapshot-%i_features.pkl 

    Returns:
    
    pupildata: pd.DataFrame
        Values for pupil features with run # and index (combined data).
        Default features are pupil_maj, pupil_min, pupil_area.

    bad_files: list
        No associated DLC data (empty) for a given run.
    ''' 
    bad_files=None; pupildata=None;
    # Set paths 
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    exp_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment

    run_dir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum, 
                            'combined_%s_static' % exp_name))[0]
    dst_dir = os.path.join(run_dir, 'facetracker')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    combined_fpath = os.path.join(dst_dir, 
                            'snapshot-%i_features.pkl' % (snapshot))
    # try loading existing
    if not create_new:
        try:
            with open(combined_fpath, 'rb') as f:
                pupildata = pkl.load(f)
            assert pupildata is not None
        except Exception as e:
            print("... error loading combined pupil traces, creating new.")
            create_new=True

    if create_new:
        # Combined pupil data and calculate metrics
        try:
            pupildata, bad_files = calculate_pose_features(datakey, experiment,
                                            feature_list=feature_list, 
                                            snapshot=snapshot,
                                            verbose=verbose,
                                            eyetracker_dir=eyetracker_dir)
            # Check for bad DLC result files 
            if bad_files is not None and len(bad_files) > 0:
                print("___ there are %i bad files ___" % len(bad_files))
                for b in bad_files:
                    print("    %s" % b)
            # save
            with open(combined_fpath, 'wb') as f:
                pkl.dump(pupildata, f, protocol=2)
                    
        except Exception as e:
            print("ERROR: %s" % datakey)
            traceback.print_exc()

    return pupildata, bad_files


def get_pose_data(datakey, experiment, feature_list=['pupil'],
                   alignment_type='stimulus', 
                   iti_pre=1., iti_post=1., 
                   traceid='traces001', snapshot=391800, 
                   verbose=False, realign=False, recombine=False,
                   rootdir='/n/coxfs01/2p-data',
                   eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
  
    '''
    1. Get alignment info for parsing trials, get_trialmeta()
    2. Load extracted features as combined df across runs, combine_post_data()
    (prev called: load_pose_data())
    
    Args.
    realign: bool
        Set TRUE to re-extra meta info for aligning trials based on triggers.

    recombine: bool
        Set TRUE to re-create/combine pupildata from extracted hdf5 features.

    Returns:
    
    trialmeta (pd.DataFrame)
        labels for each frame and trial

    pupildata (pd.DataFrame)
        all extracted frames and trials
    
    params (dict)
        Params for parsing trials         
    ''' 
    trialmeta=None; pupildata=None; params=None;

    # Create or laod trialmetadata
    trialmeta, params = get_trialmeta(datakey, experiment,
                                   alignment_type=alignment_type, 
                                   iti_pre=iti_pre, iti_post=iti_post,
                                   snapshot=snapshot, create_new=realign,
                                   rootdir=rootdir, eyetracker_dir=eyetracker_dir) 
    if trialmeta is not None:
        # Get pupil data
        pupildata, bad_files = combine_pose_data(datakey, experiment,
                                    feature_list=feature_list, 
                                    snapshot=snapshot, create_new=recombine,
                                    verbose=verbose, 
                                    rootdir=rootdir, 
                                    eyetracker_dir=eyetracker_dir)
        params['dlc_empty_files'] = bad_files

    return trialmeta, pupildata, params

# ===================================================================
# Feature extraction (traces)
# ====================================================================
def align_traces(datakey, experiment, trialmeta, pupildata, params,
                traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    '''
    Align traces (pupildata, combined across runs) from pose data.
    Update params and save results.

    Inputs: from get_pose_data()

    Output:
        ./RUNDIR/facetracker/snapshot-%i_<alignment>_param.json
        ./RUNDIR/facetracker/snapshot-%i_<alignment>_traces.pkl

    '''
    ptraces=None; missing_trials=None;
    # Get labels
    labels = aggr.load_frame_labels(datakey, experiment, 
                                traceid=traceid, rootdir=rootdir)
    # Split traces into trials
    ptraces, missing_trials = traces_to_trials(
                                trialmeta, pupildata, labels,
                                return_missing=True)

    if ptraces is not None:
        # Update params with missing trial info
        params['missing_trials'] = missing_trials
        # Save
        save_traces_and_params(datakey, experiment, ptraces, params)
 
    return ptraces, params

def add_trial_labels(trialmeta, pupildata, labels):

    # Make sure we only take the included runs
    included_run_indices = labels['run_ix'].unique() #0 indexed
    mwmeta_runs = trialmeta['run_num'].unique() # 1 indexed
    pupildata_runs = pupildata['run_num'].unique() # 1 indexed
    
    if 0 in included_run_indices and (1 not in mwmeta_runs): # skipped _run1
        included_run_indices1 = [int(i+2) for i in included_run_indices]
    else:
        included_run_indices1 = [int(i+1) for i in included_run_indices]
    tmpmeta = trialmeta[trialmeta['run_num'].isin(included_run_indices1)]
    tmppupil = pupildata[pupildata['run_num'].isin(included_run_indices1)]

    # Add global trial num and stimulus config info to face metadata
    trial_key = pd.DataFrame({'config': [g['config'].unique()[0] \
                                    for trial, g in labels.groupby(['trial'])],
                              'trial': [int(trial[5:]) \
                                    for trial, g in labels.groupby(['trial'])]})
    trialmeta = pd.concat([tmpmeta, trial_key], axis=1)

    return trialmeta, pupildata


def traces_to_trials(trialmeta, pupildata, labels, 
                    return_missing=False, verbose=False):
    '''
    Combines indices for MW trials (trialmeta) with DLC traces (pupildata)
    and assigns stimulus/condition info w/ labels.
    (Prev called get_pose_traces()).
    Grabs whatever features are saved from calculate_pose_features()
 
    Time stamps are relative to MW and eyetracker camera.
    
    Args:
        trialmeta (pd.DataFrame): 
            Meta info from MW with eyetracker frame indices aligned to trials.
            Triggers indices found in get_trial_triggers().

        pupildata (pd.DataFrame): 
            Extracted traces from DLC analysis, combined across runs.
        
        labels: pd.DataFrame
            Stimulus info for each trial/frame, combined across runs. 
    '''
    print('... splitting traces into trials')

    # Make sure we only take the included runs
    trialmeta, pupildata = add_trial_labels(trialmeta, pupildata, labels)

    # Get pose traces for each valid trial
    feature_cols = [k for k in pupildata.columns if 'run_' not in k]
    missing_trials = []
    p_list = []
    for tix, (trial, g) in enumerate(trialmeta.groupby(['trial'])):
        curr_config = g['config'].unique()[0] 
        # Get run of experiment that current trial is in
        run_label = g['run_label'].unique()[0]
        (e_start, e_end), = g[['start_ix', 'end_ix']].values
        if np.isnan(e_start) or np.isnan(e_end):
            #print("NaN in trial triggers: %s" % (trial))
            missing_trials.append(trial)
            continue
        pdf = pupildata[pupildata['run_label']==run_label]\
                        .iloc[int(e_start):int(e_end)+1]
        if len(pdf)==0 or np.isnan(pdf[feature_cols]).all(axis=None):
            missing_trials.append(trial)
            continue
        pdf['config'] = curr_config
        pdf['trial'] = trial
        pdf['stim_dur_ms'] =  float(g['stim_dur_ms'].unique())
        pdf['pre_iti_ms'] =  float(g['pre_iti_ms'].unique())
        pdf['post_iti_ms'] =  float(g['post_iti_ms'].unique())
        pdf['actual_iti_ms'] =  float(g['actual_iti_ms'].unique())
        p_list.append(pdf)
    pupiltraces = pd.concat(p_list, axis=0) #.fillna(method='pad')  
    print("... missing %i trials total" % (len(missing_trials)))

    if return_missing:
        return pupiltraces, missing_trials
    else:
        return pupiltraces



def parse_pose_data(datakey, experiment, 
                    iti_pre=1., iti_post=1., feature_list=['pupil'], 
                    alignment_type='trial', realign=False, recombine=False, 
                    snapshot=391800, verbose=False,
                    traceid='traces001', rootdir='/n/coxfs01/2p-data',
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    '''
    Load and parse eyetracker data, align to MW trials. 
    Calculates metrics, then saves to combined_run dir in 'facetracker' subdir.

    iti_pre, iti_post: float    
        How much of the trial to use for pre/post, in SEC.

    return_errors: bool
        Set TRUE to return dict of missing/bad file info.

    feature_name: str
        What feature to extract. If "pupil_fraction" this is calculate internally.
    
    alignment_type: str
        'trial' means use pre/post ITI + stimulus duration (read from MW file).
        'stimulus' means just use stimulus duration (ignores pre/post ITI).

    Outputs:
        
    ptraces: pd.DatFrame
        Saved to: .../combined_blobs_static/facetracker/FNAME.pkl

    params: dict
        Saved to: .../combined_blobs_static/facetracker/FNAME_params.json

    '''
    print("[%s] Parsing pose data (%s)." % (datakey, experiment))
    #### Load pupil data
    trialmeta, pupildata, params = get_pose_data(datakey, experiment,
                                        snapshot=snapshot,
                                        feature_list=feature_list,
                                        alignment_type=alignment_type,
                                        iti_pre=iti_pre, iti_post=iti_post, 
                                        verbose=verbose, 
                                        realign=realign, recombine=recombine,
                                        eyetracker_dir=eyetracker_dir)
    if trialmeta is None or pupildata is None:
        return None, None

    #### Parse pupil data into traces and save
    ptraces=None
    ptraces, params = align_traces(datakey, experiment,
                                        trialmeta, pupildata, params,
                                        traceid=traceid, rootdir=rootdir)
    if ptraces is None:
        print("    !!! no data !!!!!")

    return ptraces, params




# ===================================================================
# Calculate metrics (trial stats)
# ====================================================================
def calculate_pose_stats(trialmeta, pupildata, labels, feature='pupil'):
    '''
    Combines indices for MW trials (trialmeta) with pupil traces (pupildata)
    and assigns stimulus/condition info w/ labels.
    
    '''
    # Make sure we only take the included runs
    included_run_indices = labels['run_ix'].unique() #0 indexed
    mwmeta_runs = trialmeta['run_num'].unique() # 1 indexed
    pupildata_runs = pupildata['run_num'].unique() # 1 indexed
    
    #included_run_indices1 = [int(i+1) for i in included_run_indices]
    #included_run_indices1
    
    if 0 in included_run_indices and (1 not in mwmeta_runs): # skipped _run1
        included_run_indices1 = [int(i+2) for i in included_run_indices]
    else:
        included_run_indices1 = [int(i+1) for i in included_run_indices]

    tmpmeta = trialmeta[trialmeta['run_num'].isin(included_run_indices1)]
    tmppupil = pupildata[pupildata['run_num'].isin(included_run_indices1)]

    # Add stimulus config info to face data
    trial_key = pd.DataFrame({'config': [g['config'].unique()[0] \
                             for trial, g in labels.groupby(['trial'])],
                  'trial': [int(trial[5:]) \
                             for trial, g in labels.groupby(['trial'])]})
    trialmeta = pd.concat([tmpmeta, trial_key], axis=1)
    
    # Calculate a pupil metric for each trial
    pupilstats = get_per_trial_metrics(tmppupil, trialmeta, feature_name=feature)
    
    return pupilstats


def get_per_trial_metrics(pupildata, trialmeta, feature_name='pupil_maj', feature_save_name=None):
    
    
    if feature_save_name is None:
        feature_save_name = feature_name
        
    config_names = sorted(trialmeta['config'].unique(), key=hutils.natural_keys)

    #pupilstats_by_config = dict((k, []) for k in config_names)
    pupilstats = []
    #fig, ax = pl.subplots()
    for tix, (trial, g) in enumerate(trialmeta.groupby(['trial'])):

        # Get run of experiment that current trial is in
        run_num = g['run_num'].unique()[0]
        if run_num not in pupildata['run_num'].unique():
            #print(run_num)
            print("--- [trial %i] warning, run %s not found in pupildata. skipping..." % (trial, run_num))
            continue
        
        if feature=='pupil':
            feature_name_tmp = 'pupil_maj'
        elif 'snout' in feature_name:
            feature_name_tmp = 'snout_area'
        else:
            feature_name_tmp = feature_name
        #print("***** getting %s *****" % feature_name_tmp)
        pupil_dists_major = pupildata[pupildata['run_num']==run_num]['%s' % feature_name_tmp]

        # Get start/end indices of current trial in run
        (eye_start, eye_end), = g[['start_ix', 'end_ix']].values
        #print(trial, eye_start, eye_end)

        #eye_tpoints = frames['time_stamp'][eye_start:eye_end+1]
        eye_values = pupil_dists_major[int(eye_start):int(eye_end)+1]
        
        # If all nan, get rid of this trial
        if all(np.isnan(eye_values)):
            continue
            
        curr_config = g['config'].iloc[0]
        #curr_cond = sdf['size'][curr_config]    
        #ax.plot(eye_values.values, color=cond_colors[curr_cond])

        #print(trial, np.nanmean(eye_values))
        #pupilstats_by_config[curr_config].append(np.nanmean(eye_values))

        pdf = pd.DataFrame({'%s' % feature_save_name: np.nan if all(np.isnan(eye_valus)) else np.nanmean(eye_values),
                            'config': curr_config,
                            'trial': trial}, index=[tix])

        pupilstats.append(pdf)

    pupilstats = pd.concat(pupilstats, axis=0)
    
    return pupilstats



# ===================================================================
# Data processing 
# ====================================================================
# Data cleanup
def get_video_metadata_for_run(curr_src):
    metadata=None
    performance_info = os.path.join(curr_src, 'times', 'performance.txt')
    metadata = pd.read_csv(performance_info, sep="\t ", engine='python')
    fps = float(metadata['frame_rate'])
    
    return metadata

def get_frame_triggers_for_run(curr_src):
    '''
    Get frame times and and interpolate missing frames for 1 run
    '''
    frames=None

    try:
        run_num = int(re.search(r"_f\d+_", os.path.split(curr_src)[-1]).group().split('_')[1][1:])
    except Exception as e:
        run_num = int(re.search(r"_f\d+[a-zA-Z]_", os.path.split(curr_src)[-1]).group().split('_')[1][1:])

    # Get meta data for experiment
    metadata=None
    try:
        metadata = get_video_metadata_for_run(curr_src) 
        fps = float(metadata['frame_rate'])
    except Exception as e:
        src_key = os.path.split(curr_src)[-1]
        print('ERROR:\n  Unable to load performance.txt (%s)' % (src_key))
        fps = 20.0

    # Get frame info
    frame_info = os.path.join(curr_src, 'times', 'frame_times.txt')
    try:
        frame_attrs = pd.read_csv(frame_info, sep="\t ", engine='python')
        frames = check_missing_frames(frame_attrs, metadata)
        #print("... adjusted for missing frames:", frames.shape)
        tif_dur_min = frames.iloc[-1]['time_stamp'] / 60.
        print("... Run %i: full dur %.2f min" % (run_num, tif_dur_min) )
    except Exception as e:
        traceback.print_exc()
   
    return frames


def check_missing_frames(frame_attrs, metadata, verbose=False):
    '''
    Find chunks where frames were dropped (based on frame_period, in metadata). 
    Fill with interpolated # of NaNs, then interpolate.

    frame_attrs : pd.DataFrame with columns 
        frame_number
        sync_in1
        sync_in2
        time_stamp
        These are NaNs, and will be interpolated. Matters for time_stamp.
    
    Returns:
        interpdf, the interpolated dataframe
        
    '''
    if verbose:
        print("... checking for missing frames.")
    if metadata is None: # Hardcode values in
        fps = 20.0
        metadata = {'frame_period': (1./fps)}
    tmpdf = frame_attrs.copy()

    # Identify frame indices where we definitely missed a frame or trigger
    missed_ixs = [m-1 for m in np.where(frame_attrs['time_stamp'].diff() > float(metadata['frame_period']*1.5))[0]]
    if len(missed_ixs)>0 and verbose:
        print("... found %i funky frame chunks: %s" \
                % (len(missed_ixs), str(missed_ixs)))

    added_=[]
    for mi in missed_ixs:
        # Identify duration of funky interval, how many missed frames dropped?
        missing_interval = float(frame_attrs['time_stamp'].iloc[mi+1])-float(frame_attrs['time_stamp'].iloc[mi])
        n_missing_frames = int(round(missing_interval/float(metadata['frame_period']), 0) -1)
        if verbose:
            print("... missing %.1f, interpolated %i frames" \
                        % (missing_interval, add_missing.shape[0]))
        # Create empty spaces for missing frames in dataframe
        ixs = np.linspace(mi, mi+1, n_missing_frames+2)[1:-1]
        add_missing = pd.DataFrame({
            'frame_number': [np.nan for i in np.arange(0, n_missing_frames)],
            'sync_in1': [np.nan for _ in np.arange(0, n_missing_frames)],
            'sync_in2': [np.nan for _ in np.arange(0, n_missing_frames)],
            'time_stamp': [np.nan for _ in np.arange(0, n_missing_frames)]},
             index=ixs
        )
        added_.append(add_missing.shape[0])
        df2 = pd.concat([tmpdf.iloc[:mi+1], add_missing, tmpdf.iloc[mi+1:]]) 
        tmpdf = df2.copy() 
    if verbose:
        if 'missingFrames' in metadata.keys():
            print("... missing %i frames, added %i" \
                    % (int(metadata['missingFrames']), np.sum(added_)))
        else:
            print("... added %i" % (np.sum(added_)))

    # Interpolate over the NaNs
    interpdf = tmpdf.interpolate().reset_index(drop=True)
    if verbose:
        print("... frame info shape changed from %i to %i frames" \
                % (frame_attrs.shape[0], interpdf.shape[0]))
    
    return interpdf


def get_raw_experiment_dirs(datakey, experiment=None,
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    '''Glob all subdirs for datakey (experiment=None, if all)'''
    # Eyetracker source files
 
    #print("... finding all movies for dset: %s" % datakey)
    if experiment is None:
        src_dirs = sorted(glob.glob(os.path.join(eyetracker_dir, 
                                    '%s_*' % (datakey))), 
                                    key=hutils.natural_keys)
    else:
        src_dirs = sorted(glob.glob(os.path.join(eyetracker_dir, 
                                    '%s_%s_*' % (datakey, experiment))), 
                                    key=hutils.natural_keys)
    return src_dirs 

def get_trial_triggers(datakey, curr_exp, 
                    alignment_type='stimulus', pre_ITI_ms=1000, post_ITI_ms=1000,
                    rootdir='/n/coxfs01/2p-data',
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp',
                    verbose=False, return_missing=False,
                    blacklist=['20191018_JC113_fov1_blobs_run5',
                               '20190622_JC085_fov1_rfs_run4']):
     
    '''
    Align MW trial events/epochs to eyetracker frames for each trial, 
    Matches eyetracker data to each "run" of a given experiment type. 
    Typically, 1 eyetracker movie for each paradigm file.
    (prev called:  align_trials_to_facedata()) 

    alignment_type (str)
        'trial' : aligned frames to pre/post stimulus period, around stimulus 
        'stimulus': align frames to stimulus ON frames (no pre/post)

    blacklist (list)
        20191018_JC113_fov1_blobs_run5:  paradigm file is f'ed up?
        20190622_JC085_fov1_rfs_run4:  missing para. file 
 
    Returns:
    
    trialmeta (dataframe)
        Start/end indices for each trial across all eyetracker movies in all the runs.
        These indices assign trial labels for pupl traces, in traces_to_trials()
    ''' 
    if curr_exp in ['rfs', 'rfs10']:
        blacklist.append('20190513_JC078_fov1')

    #epoch = 'stimulus_on'
    #pre_ITI_ms = 1000
    #post_ITI_ms = 1000

    trialmeta=None
    # Get all runs for the current experiment
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    exp_name = 'gratings' if (curr_exp in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else curr_exp

    all_runs = sorted(glob.glob(os.path.join(rootdir, animalid, session, 
                                'FOV%i*' % fovnum, '%s_run*' % exp_name)), 
                                key=hutils.natural_keys)
    run_list = [os.path.split(rundir)[-1].split('_run')[-1] \
                                for rundir in all_runs] 
    print("    %s: found runs (%s):" % (curr_exp, exp_name), run_list)
    
    # Eyetracker source files
    facetracker_srcdirs = get_raw_experiment_dirs(datakey, experiment=None,
                    eyetracker_dir=eyetracker_dir)
    if verbose:
        for si, sd in enumerate(facetracker_srcdirs):
            print(si, sd)

    # Align facetracker frames to MW trials based on time stamps
    missing_dlc=[]
    trialmeta_list = []
    for run_num in run_list:
        if verbose:
            print("... File %s:" % run_num)
        run_numeric = int(re.findall('\d+', run_num)[0]) # Get numeric val of run
        if verbose:
            print("... %s (%s), getting MW run: %s (run_%i)" \
                    % (curr_exp, exp_name, run_num, run_numeric)) 
        # Check blacklist
        full_dkey ='%s_%s_run%s' % (datakey, curr_exp, run_numeric) 
        if full_dkey in blacklist or datakey in blacklist:
            continue 
        # Get MW info for this run
        if verbose:
            print("... %i tifs in run (%i trials)" % (n_files, len(trialnames))) 
        mw_file = glob.glob(os.path.join(rootdir, animalid, session, 
                                        'FOV%i*' % fovnum,\
                                        '*%s_run%i' % (exp_name, run_numeric), \
                                        'paradigm', 'trials_*.json'))[0]
        with open(mw_file, 'r') as f:
            mw = json.load(f)
        # How many raw runs go with this MW file?
        n_files = len( glob.glob(os.path.join(rootdir, animalid, 
                                        session, 'FOV%i*' % fovnum,\
                                        '*%s_run%i' % (exp_name, run_numeric), 
                                        'raw*', '*.tif')) ) 
        file_ixs = np.arange(0, n_files)
        trialnames = sorted([t for t, md in mw.items() if md['block_idx'] \
                            in file_ixs \
                            and md['stimuli']['type'] != 'blank'], \
                            key=hutils.natural_keys)
        # Check trial epoch durations
        actual_iti = float(mw[trialnames[0]]['iti_dur_ms'])
        assert pre_ITI_ms + post_ITI_ms <= actual_iti, \
                "Full ITI was %i ms, requested %.1f (pre) + %.1f (post) ITI ms" \
                % (actual_iti, pre_ITI_ms, post_ITI_ms) 
        start_t = mw[trialnames[0]]['start_time_ms'] - mw[trialnames[0]]['iti_dur_ms']
        # end_t = mw[trialnames[0]]['end_time_ms']

        # Get corresponding eyetracker dir for run
        try:
            curr_face_srcdir = [s for s in facetracker_srcdirs \
                                    if '%s_f%s_' % (exp_name, run_num) in s][0]
            if verbose:
                print('... Eyetracker: %s' % os.path.split(curr_face_srcdir)[-1])

            # Get eyetracker metadata
            video_meta = get_frame_triggers_for_run(curr_face_srcdir)
            assert video_meta is not None, "NO meta for run: %s" % curr_face_srcdir

        except Exception as e:
            print("... ERROR loading run (%s, %s): %s" % (datakey, curr_exp, run_num))
            run_key = '%s_%s' % (curr_exp, run_num)
            missing_dlc.append((datakey, run_key))
            continue
 
        for tix, curr_trial in enumerate(sorted(trialnames, key=hutils.natural_keys)): 
            # Get SI triggers for start and end of trial
            if 'retino' in curr_exp:
                trial_num = int(curr_trial)
                curr_trial_triggers = mw[str(curr_trial)]['stiminfo']['trigger_times']
                units = 1E6
            else:
                units = 1E3
                trial_num = int(curr_trial[5:])
                if alignment_type == 'trial':
                    stim_on_ms = mw[curr_trial]['start_time_ms']
                    stim_dur_ms = mw[curr_trial]['stim_dur_ms']
                    start_tstamp = stim_on_ms-pre_ITI_ms
                    end_tstamp = stim_on_ms+stim_dur_ms+post_ITI_ms
                elif alignment_type == 'stimulus':
                    stim_on_ms = mw[curr_trial]['start_time_ms']
                    stim_dur_ms = mw[curr_trial]['stim_dur_ms']
                    start_tstamp = stim_on_ms
                    end_tstamp = stim_on_ms+stim_dur_ms 
                else:
                    start_tstamp = mw[curr_trial]['start_time_ms']
                    end_tstamp = mw[curr_trial]['end_time_ms']
                #curr_trial_triggers = [stim_on_ms, stim_on_ms + stim_dur_ms]
                curr_trial_triggers = [start_tstamp, end_tstamp]
            # Calculate trial duration in secs
            # nsecs_trial = ((curr_trial_triggers[1]-curr_trial_triggers[0])/units ) 
            # Get number of eyetracker frames this corresponds to
            # nframes_trial = nsecs_trial * metadata['frame_rate']

            # Get start time and end time of trial (or tif) relative to start of RUN
            trial_start_sec = (curr_trial_triggers[0] - start_t) / units
            trial_end_sec = (curr_trial_triggers[-1] - start_t) / units

            # Get corresponding eyetracker frame indices for start and end time points
            eye_start = np.where(abs(video_meta['time_stamp']-trial_start_sec) == (abs(video_meta['time_stamp']-trial_start_sec).min()))[0][0]
            eye_end = np.where(abs(video_meta['time_stamp']-trial_end_sec) == (abs(video_meta['time_stamp']-trial_end_sec).min()) )[0][0]
            if verbose:
                print("Eyetracker start/stop frames:", eye_start, eye_end)
            # Make dataframe
            face_movie = '_'.join(os.path.split(curr_face_srcdir)[-1].split('_')[0:-1])
            tmpdf = pd.DataFrame({'start_ix': eye_start,
                                  'end_ix': eye_end,
                                  'trial_in_run': trial_num,
                                  'run_label': run_num,
                                  'run_num': run_numeric,
                                  'alignment_type': alignment_type,
                                  'stim_dur_ms': stim_dur_ms,
                                  'actual_iti_ms': actual_iti,
                                  'pre_iti_ms': pre_ITI_ms,
                                  'post_iti_ms': post_ITI_ms,
                                  'movie': face_movie}, index=[tix])
            trialmeta_list.append(tmpdf)
    if len(trialmeta_list)>0:
        trialmeta = pd.concat(trialmeta_list, axis=0).reset_index(drop=True)

    print("... There were %i missing DLC results." % len(missing_dlc))
    for d in missing_dlc:
        print('    %s' % str(d))
    if return_missing:
        return trialmeta, missing_dlc
    else:
        return trialmeta


def calculate_pose_features(datakey, experiment, feature_list=['pupil'], 
                    snapshot=391800, verbose=False,
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):

    '''
    Load DLC pose analysis results, extract feature for all runs of the experiment.
    Assigns face-data frames to MW trials.
    (Use to be called: parse_pose_data)

    Returns:
    
    pupildata: pd.Ddataframe
        Contains all analyzed (and thresholded) frames for all runs.
        NaNs if no data -- no trials yet, either.

    bad_files: list (str)
        Runs where no pupil data was found, though we expected it.

    '''
    dlc_results_dir, dlc_video_dir = get_dlc_sources()

    print("Calculating pose metrics per trial (%s)" % datakey)
    # DLC outfiles
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    exp_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment

    dlc_outfiles = sorted(glob.glob(os.path.join(dlc_results_dir, 
                        '%s_%s_f*_%i.h5' % (datakey, exp_name, snapshot))), \
                        key=hutils.natural_keys)
    #print(dlc_outfiles)
    if len(dlc_outfiles)==0:
        print("***ERROR: no DLC files (dir:  \n%s)" % dlc_results_dir)
        return None, None

    # Eyetracker source files
    # print("... checking movies for dset: %s" % datakey)
    facetracker_srcdirs = sorted(glob.glob(os.path.join(eyetracker_dir, 
                                '%s_%s_f*' % (datakey, exp_name))), \
                                key=hutils.natural_keys)

    if len(dlc_outfiles) != len(facetracker_srcdirs):
        print("ERROR: Incorrect # dlc output files and found raw input files:")
        print('- DLC OUTFILES (n=%i files):' % len(dlc_outfiles))
        for f in dlc_outfiles:
            print('    %s' % f)

        print('- SRC DIRS (n=%i files):' % len(facetracker_srcdirs))
        for f in facetracker_srcdirs:
            print('    %s' % f)
        
        return None, None
 
    # Check that run num is same for PARA file and DLC results
    for fd, od in zip(facetracker_srcdirs, dlc_outfiles):
        fsub = os.path.split(fd)[-1]
        osub = os.path.split(od)[-1]
        #print('names:', fsub, osub)
        try:
            face_fnum = re.search(r"_f\d+_", fsub).group().split('_')[1][1:]
            dlc_fnum = re.search(r"_f\d+DLC", osub).group().split('_')[1][1:-3]
        except Exception as e:
            #traceback.print_exc()
            face_fnum = re.search(r"_f\d+[a-zA-Z]_", fsub).group().split('_')[1][1:]
            dlc_fnum = re.search(r"_f\d+[a-zA-Z]DLC", osub).group().split('_')[1][1:-3] 
        assert dlc_fnum == face_fnum, "Incorrect match: %s / %s" % (fsub, osub)

    bad_files=[]
    p_list=[]
    pupildata=None
    for dlc_outfile in sorted(dlc_outfiles, key=hutils.natural_keys):
        # Identify run number for current dlc video file
        run_num=None
        try:
            fbase = os.path.split(dlc_outfile)[-1]
            run_num = re.search(r"_f\d+DLC", fbase).group().split('_')[1][1:-3]
        except Exception as e: 
            run_num = re.search(r"_f\d+[a-zA-Z]DLC", fbase).group().split('_')[1][1:-3] 
        assert run_num is not None, "Unable to find run_num for file: %s" % dlc_outfile
        if verbose:
            print("... run %s: %s" % (run_num, os.path.split(dlc_outfile)[-1]))

        # Get corresponding DLC results for movie
        #dlc_outfile = [s for s in dlc_outfiles if '_f%iD' % run_num in s][0]
        
        # Calculate some statistic from pose data
        feature_dict=[] #{}
        for feature in feature_list:
            currdf=None
            if verbose:
                print("... calculating: %s" % feature)
            if 'pupil' in feature:
                currdf = calculate_pupil_metrics(dlc_outfile)
#                if pupil_major is not None and pupil_minor is not None:
#                    pupil_areas = [np.pi*p_maj*p_min for p_maj, p_min in \
#                                        zip(pupil_major, pupil_minor)]
#                    feature_dict.update({'pupil_maj': pupil_major,
#                                         'pupil_min': pupil_minor,
#                                         'pupil_area': pupil_areas})
            elif 'snout' in feature:
                #snout_areas = calculate_snout_metrics(dlc_outfile
                # feature_dict.update({'snout_area': snout_areas})
                currdf = calculate_snout_metrics(dlc_outfile)
            elif 'whisker' in feature:
                currdf = calculate_whisker_metrics(dlc_outfile)

            if currdf is not None:
                feature_dict.append(currdf)
 
        if len(feature_dict)==0:
            bad_files.append(dlc_outfile)
            continue

        pdf = pd.concat(feature_dict, axis=1)
 
        #fkey = list(feature_dict.keys())[0]
        #nsamples = len(feature_dict[fkey]) #pupil_major)
        run_numeric = int(re.findall('\d+', run_num)[0])
        # Create dataframe
        #pdf = pd.DataFrame(feature_dict, index=np.arange(0, nsamples)) 
        pdf['run_label'] = run_num
        pdf['run_num'] = run_numeric
        #pdf.index = np.arange(0, nsamples)
        p_list.append(pdf)

    pupildata = pd.concat(p_list, axis=0)
        
    pupil_max = pupildata['pupil_area'].max()
    pupildata['pupil_fraction'] = pupildata['pupil_area']/pupil_max
 
    print("... done parsing!") 

    return pupildata, bad_files


# body feature extraction
def get_dists_between_bodyparts(bp1, bp2, df): #, DLCscorer=None):

#    if DLCscorer is not None:
#        coords1 = [np.array([x, y]) for x, y, in \
#                    zip(df[DLCscorer][bp1]['x'].values, df[DLCscorer][bp1]['y'].values)]
#        coords2 = [np.array([x, y]) for x, y, in \
#                    zip(df[DLCscorer][bp2]['x'].values, df[DLCscorer][bp2]['y'].values)]
#    else:
    coords1 = [np.array([x, y]) for x, y, in \
                zip(df[bp1]['x'].values, df[bp1]['y'].values)]
    coords2 = [np.array([x, y]) for x, y, in \
                zip(df[bp2]['x'].values, df[bp2]['y'].values)]

    dists = np.array([np.linalg.norm(c1-c2) for c1, c2 in zip(coords1, coords2)])
    
    return dists


def get_relative_position(df, feat='pupilC'):
    '''Get Euclid. distance between each successive row. First entry is Nan'''
    dists = [np.linalg.norm(c1-c2) for c1, c2 in \
                        zip(df[feat][['x', 'y']].values,
                            df[feat][['x', 'y']].shift().values)]
    return dists


def calculate_pupil_centers(tmpdf): #, DLCscorer=None):
    '''Calculate pupil center from line intersection of major/minor axes'''
    newdf=None
#    if DLCscorer is not None:
#        tmpdf = df[DLCscorer].copy()
#    else:
#        tmpdf = df.copy()
    A = [tuple([x,y]) for x, y in tmpdf['pupilT'][['x', 'y']].values]
    B = [tuple([x,y]) for x, y in tmpdf['pupilB'][['x', 'y']].values]

    C = [tuple([x,y]) for x, y in tmpdf['pupilL'][['x', 'y']].values]
    D = [tuple([x,y]) for x, y in tmpdf['pupilR'][['x', 'y']].values]

    # Calculate centers from intersection of 2 lines 
    ctrs = [line_intersection((a, b), (c, d)) for a, b, c, d in zip(A, B, C, D)]
    # Save as dataframe coords    
    newdf = pd.DataFrame(ctrs, columns={'x', 'y'})

    # Add likelihoods from MIN of combined:
    feature_list = ['pupilT', 'pupilB', 'pupilL', 'pupilR']
    calc_cols = [(x, 'likelihood') for x in feature_list]
    min_likelihoods = tmpdf[calc_cols].min(axis=1)

    newdf['likelihood'] = min_likelihoods

    return newdf


def add_feature_to_df(df, newdf, new_features=[], ix_levels=['x', 'y', 'likelihood']):
    '''
    Add proper MultiIndex columns to dataframe to be added to main DLC results.
    Main results are loaded from hdf5, and assumed to have:
    Level 0: 'bodyparts', 'coords'
    Level 1: 'x', 'y', 'likelihood'

    Returns combined df (df - orig, newdf - added df).
    '''
    #lowest_ix_levels = ['x', 'y']
    #new_features = ['pupilC']
    new_labels = [[f]*len(ix_levels) for f in new_features]
    new_labels.append(np.tile(ix_levels, len(new_features)))
    new_ix = pd.MultiIndex.from_arrays(new_labels, names=('bodyparts', 'coords'))
    newdf.columns = new_ix

    tmpdf = pd.concat([df, newdf], axis=1)

    return tmpdf


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def filter_dlc_scores(df, threshold=0.99, bodyparts=[], filtered=False):
    '''Filter DLC scores before or after caluclating metrics
    Returns inputs for calculating metrics.

    '''
    DLCscorer = df.columns.get_level_values(level=0).unique()[0]
    filtered_df = df[DLCscorer][bodyparts][df[DLCscorer][bodyparts] >= threshold].dropna()
    kept_ixs = filtered_df.index.tolist()

    if filtered:    
        finaldf = filtered_df.copy()
        dlc_scorer = None
        replace_ixs = []
    else:
        finaldf = df.copy()
        dlc_scorer = DLCscorer
        # Save indices to replace with NaNs if not using filtered df above
        replace_ixs = np.array([i for i in np.arange(0, df.shape[0]) \
                                if i not in kept_ixs])

    if dlc_scorer is not None:
        tmpdf = finaldf[dlc_scorer].copy()
    else:
        tmpdf = finaldf.copy()

    return tmpdf, dlc_scorer, replace_ixs


def calculate_pupil_metrics(dlc_outfile, filtered=False, threshold=0.99):
    '''
    Read specific dlc outfile (hdf5). Extract bodyparts, and do calculations.

    For pupil, bodyparts = ['pupilT', 'pupilB', 'pupilL', 'pupilR', 'cornealR']

    dlc_outfile: str
        Path to output file (.h5) saved by DLC extraction notebook (collab).

    filtered: bool
        Set TRUE in order to filter by specified threshold value.

    threshold: float
        Drops values for which DLC is not sure (higher = more certain).

    '''
    bodyparts = ['pupilT', 'pupilB', 'pupilL', 'pupilR', 'cornealR']

    df = pd.read_hdf(dlc_outfile)
    if df.shape[0] < 5: # sth wrong
        return None
   
    tmpdf, dlc_scorer, replace_ixs = filter_dlc_scores(df, threshold=threshold, 
                                                filtered=filtered, bodyparts=bodyparts)
    # Add additional columns
    df1 = calculate_pupil_centers(tmpdf) # Add 'pupilC' as intersection of lines
    finaldf = add_feature_to_df(tmpdf, df1, new_features=['pupilC'])

    # Calculate metrics
    pupil_major = get_dists_between_bodyparts(
                            'pupilT', 'pupilB', finaldf) #DLCscorer=dlc_scorer)
    pupil_minor = get_dists_between_bodyparts(
                            'pupilL', 'pupilR', finaldf) #DLCscorer=dlc_scorer)

    #TODO:  CR as dist between intersection of pupil_maj and pupil_min? 
    cr_dist = get_dists_between_bodyparts('pupilC', 'cornealR', finaldf) #, DLCscorer=dlc_scorer)
    rel_pupil_pos = get_relative_position(finaldf, feat='pupilC')

    # Cal area
    pupil_areas = [np.pi*p_maj*p_min for p_maj, p_min in zip(pupil_major, pupil_minor)]
 
    df_ = pd.DataFrame({'pupil_maj': pupil_major,  
                        'pupil_min': pupil_minor,
                        'pupil_area': pupil_areas,
                        'cr_dist': cr_dist,
                        'pupil_dist': rel_pupil_pos}, index=finaldf.index)

    #print("Replacing bad vals")
    if len(replace_ixs) > 0:
        df_.loc[replace_ixs] = np.nan
        #pupil_major[replace_ixs] = np.nan
        #pupil_minor[replace_ixs] = np.nan
        #cr_dist[replace_ixs] = np.nan

    return df_ #pupil_major, pupil_minor


def calculate_whisker_metrics(dlc_outfile, filtered=False, threshold=.99999999):
    from shapely import geometry
    bodyparts = ['whiskerP', 'whiskerP1', 'whiskerP2', 'whiskerP3',
               'whiskerAU', 'whiskerAU1', 'whiskerAU2', 'whiskerAU3',
               'whiskerAL', 'whiskerAL1', 'whiskerAL2', 'whiskerAL3']

    df = pd.read_hdf(dlc_outfile)
    if df.shape[0] < 5: # sth wrong
        return None

    finaldf, dlc_scorer, replace_ixs = filter_dlc_scores(df, threshold=threshold, 
                                                filtered=filtered, bodyparts=bodyparts)

    feat_={}
    for bp in bodyparts: 
        rel_pos = get_relative_position(finaldf, feat=bp)
        feat_.update({bp: rel_pos})
 
    df_ = pd.DataFrame(feat_, index=finaldf.index)

    #print("Replacing bad vals")
    if len(replace_ixs) > 0:
        df_.loc[replace_ixs] = np.nan   

    return df_ #snout_areas


def calculate_snout_metrics(dlc_outfile, filtered=False, threshold=.99999999):
    from shapely import geometry

    #bodyparts = ['snoutA', 'snoutL2', 'snoutL1', 'whiskerAL', 'whiskerP', 'whiskerAU', 'snoutU1', 'snoutU2']
    bodyparts = ['snoutA', 'snoutL2', 'snoutL1', 'whiskerAL2', 'whiskerP2', 'whiskerAU2', 'snoutU1', 'snoutU2']

    df = pd.read_hdf(dlc_outfile)
    if df.shape[0] < 5: # sth wrong
        return None

    finaldf, dlc_scorer, replace_ixs = filter_dlc_scores(df, threshold=threshold, 
                                                filtered=filtered, bodyparts=bodyparts)
   
#    if filtered:
#        xcoords = filtdf[bodyparts].xs(('x'), level=('coords'), axis=1)
#        ycoords = filtdf[bodyparts].xs(('y'), level=('coords'), axis=1)
#    else:
#        xcoords = df[DLCscorer][bodyparts].xs(('x'), level=('coords'), axis=1)
#        ycoords = df[DLCscorer][bodyparts].xs(('y'), level=('coords'), axis=1)
#
    xcoords = finaldf[bodyparts].xs(('x'), level=('coords'), axis=1)
    ycoords = finaldf[bodyparts].xs(('y'), level=('coords'), axis=1)
   
    nsamples = xcoords.shape[0]
    snout_areas = np.array([poly_area(xcoords.iloc[i,:], ycoords.iloc[i,:]) \
                            for i in np.arange(0, nsamples)])
    df_ = pd.DataFrame({'snout_area': snout_areas}, index=finaldf.index)

    #print("Replacing bad vals")
    if len(replace_ixs) > 0:
        df_.loc[replace_ixs] = np.nan   

    return df_ #snout_areas


def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class Struct():
    pass

def subtract_condition_mean(neuraldata, labels, included_trials):
    
    # Remove excluded trials and Calculate neural residuals
    trial_configs = pd.DataFrame(np.vstack([g['config'].iloc[0]\
                                        for trial, g in labels.groupby(['trial']) \
                                           if int(trial[5:]) in included_trials]), columns=['config']) # trials should be 1-indexed
    trial_configs = trial_configs.loc[included_trial_ixs]
    
    # Do mean subtraction for neural data
    residuals_neural = neuraldata.copy()
    for c, g in trial_configs.groupby(['config']):
        residuals_neural.loc[g.index] = neuraldata.loc[g.index] - neuraldata.loc[g.index].mean(axis=0)

    return residuals_neural


# ===================================================================
# Trace processing 
# ====================================================================
from scipy import interpolate

def resample_traces(samples, in_rate=44.65, out_rate=20.0):

    n_in_samples= len(samples)
    in_samples = samples.copy()
    in_tpoints = np.arange(0, n_in_samples)
    flinear = interpolate.interp1d(in_tpoints, in_samples, axis=0)
    
    n_out_samples = round(n_in_samples * out_rate/in_rate)
    #print("N out samples: %i" % n_out_samples)
    out_tpoints = np.linspace(in_tpoints[0], in_tpoints[-1], n_out_samples)
    out_samples = flinear(out_tpoints)
    #print("Out samples:", out_samples.shape)
    
    return out_tpoints, out_samples

def resample_traces2(samples, in_rate=44.65, out_rate=20.0):

    n_in_samples= len(samples)
    in_samples = samples.copy()
    in_tpoints = np.arange(0, n_in_samples)
    flinear = interpolate.interp1d(in_tpoints, in_samples, axis=0)
    
    n_out_samples = round(n_in_samples * out_rate/in_rate)
    #print("N out samples: %i" % n_out_samples)
    out_tpoints = np.linspace(in_tpoints[0], in_tpoints[-1], n_out_samples)
    out_samples = flinear(out_tpoints)
    #print("Out samples:", out_samples.shape)
 
    return out_samples


def pad_traces(values, npad=0, mode='edge'):
    vals = np.pad(values, pad_width=((0, npad)), mode=mode)
    return vals


def resample_pupil_traces(pupiltraces, in_rate=20.0, out_rate=22.325, 
                        min_nframes=None, iti_pre_ms=1000):

    if min_nframes is None:
        min_nframes = int(round(np.mean([len(g) for p, g in pupiltraces.groupby(['trial'])])))

    new_stim_on = (iti_pre_ms/1E3)*out_rate 

    fcols = [k for k in pupiltraces.columns if k!='config']
    p_ = []
    for trial, g in pupiltraces.groupby(['trial']):
        if g.shape[0] < min_nframes:
            npad = min_nframes - g.shape[0]
            gpad = g.apply(pad_traces, npad=npad) 
        else:
            gpad = g.iloc[0:min_nframes].copy()
       
        currconfig = g['config'].unique()[0]
 
        pdf = gpad[fcols].apply(resample_traces2, in_rate=in_rate, out_rate=out_rate)
        pdf['stim_on'] = new_stim_on
        pdf['config'] = currconfig 
        pdf['trial'] = trial
        pdf['frame_ix'] = np.arange(0, min_nframes)

        p_.append(pdf)

    pupildf = pd.concat(p_, axis=0).reset_index(drop=True)

    return pupildf


#def bin_pupil_traces(pupiltraces, feature_name='pupil',in_rate=20.0, out_rate=22.325, 
#                          min_nframes=None, iti_pre_ms=1000):
#    pupildfs = []
#    if min_nframes is None:
#        min_nframes = int(round(np.mean([len(g) for p, g in pupiltraces.groupby(['trial'])])))
#
#    fcols = [k for k in pupiltraces.columns if k!='config']
#
#    for trial, g in pupiltraces.groupby(['trial']):
#        if len(g[feature_name]) < min_nframes:
#            npad = min_nframes - len(g[feature_name])
#            vals = np.pad(g[feature_name].values, pad_width=((0, npad)), mode='edge')
#        else:
#            vals = g[feature_name].values[0:min_nframes]
#        #print(len(vals))
#        out_ixs, out_s = resample_traces(vals, in_rate=in_rate, out_rate=out_rate)
#        currconfig = g['config'].unique()[0]
#        new_stim_on = (iti_pre_ms/1E3)*out_rate 
#        pupildfs.append(pd.DataFrame({feature_name: out_s, 
#                                       'stim_on': [new_stim_on for _ in np.arange(0, len(out_s))],
#                                       'config': [currconfig for _ in np.arange(0, len(out_s))],
#                                       'trial': [trial for _ in np.arange(0, len(out_s))]} ))
#    pupildfs = pd.concat(pupildfs, axis=0).reset_index(drop=True)
#    return pupildfs
#

def zscore_array(v):
    return (v-v.mean())/v.std()


#def resample_pupil_traces(pupiltraces, in_rate=20., out_rate=20., iti_pre_ms=1000, 
#                    desired_nframes=60, feature_name='pupil_area'):
#    '''
#    resample pupil traces to make sure we have exactly the right # of frames to match neural data
#    '''
#    binned_pupil = bin_pupil_traces(pupiltraces, #feature_name=feature_name,
#                                    in_rate=in_rate, out_rate=out_rate, 
#                                    min_nframes=desired_nframes, iti_pre_ms=iti_pre_ms)
#    trials_ = sorted(pupiltraces['trial'].unique())
#    frames_ = np.arange(0, desired_nframes)
#
#    pupil_trialmat = pd.DataFrame(np.vstack([p[feature_name].values for trial, p in binned_pupil.groupby(['trial'])]),
#                                  index=trials_, columns=frames_)
#    pupil_r = pupil_trialmat.T.unstack().reset_index().rename(columns={'level_0': 'trial', 
#                                                                       'level_1': 'frame',
#                                                                       0: feature_name})
#    pupil_r['frame_int'] = [int(round(f)) for f in pupil_r['frame']]
#    interp_frame_ixs = list(sorted(pupil_r['frame'].unique()))
#    pupil_r['frame_ix'] = [interp_frame_ixs.index(f) for f in pupil_r['frame']]
#
#    return pupil_r
#    
def match_trials(neuraldf, pupiltraces, labels_all):
    '''
    make sure neural data trials = pupil data trials
    '''
    trials_with_pupil = list(pupiltraces['trial'].unique())
    trials_with_neural = list(labels_all['trial_num'].unique())
    n_pupil_trials = len(trials_with_pupil)
    n_neural_trials = len(trials_with_neural)

    labels = labels_all[labels_all['trial_num'].isin(trials_with_pupil)].copy()
    if n_pupil_trials > n_neural_trials:
        pupiltraces = pupiltraces[pupiltraces['trial'].isin(trials_with_neural)]
    elif n_pupil_trials < n_neural_trials:    
        print(labels.shape, labels_all.shape)
        neuraldf = neuraldf.loc[trials_with_pupil]
    
    return neuraldf, pupiltraces

def match_neural_and_pupil_trials(neuraldf, pupildf, equalize_conditions=False):
    '''
    make sure neural data trials = pupil data trials
    Former name:  match_trials_df
    '''
    #from pipeline.python.classifications.aggregate_data_stats import equal_counts_df
    trials_with_pupil = list(pupildf['trial'].unique())
    trials_with_neural = neuraldf.index.tolist()
    n_pupil_trials = len(trials_with_pupil)
    n_neural_trials = len(trials_with_neural)

    if 'trial' in neuraldf.columns:
        neuraldf.index = neuraldf['trial']
    else:
        neuraldf['trial'] = neuraldf.index.tolist()
    if n_pupil_trials > n_neural_trials:
        pupildf0 = pupildf[pupildf['trial'].isin(trials_with_neural)]
        neuraldf0 = neuraldf.copy()
    elif n_pupil_trials < n_neural_trials:
        neuraldf0 = neuraldf.loc[trials_with_pupil]
        pupildf0 = pupildf.copy()
    else:
        neuraldf0 = neuraldf.copy()
        pupildf0 = pupildf.copy() 
    # Equalize trial numbers after all neural and pupil trials matched 
    if equalize_conditions:
        neuraldf1 = aggr.equal_counts_df(neuraldf0)
        new_trials_neural = neuraldf1.index.tolist()
        new_trials_pupil = pupildf0['trial'].unique()
        if len(new_trials_neural) < len(new_trials_pupil):
            pupildf1 = pupildf0[pupildf0['trial'].isin(new_trials_neural)]
        else:
            pupildf1 = pupildf0.copy()
    else:
        neuraldf1 = neuraldf0.copy()
        pupildf1 = pupildf0.copy()
           
    return neuraldf1, pupildf1

def neural_trials_from_pupil_trials(neuraldf, pupildf):
    '''
    Given pupildf, with trial numbers (trial is column in pupildf),
    return the corresponding neuraldf trials.
    Also return subset of pupil df as needed.

    '''
    return None

def split_pupil_range(pupildf, feature_name='pupil_area', n_cuts=3, return_bins=False):
    ''' 
    Split pupil into "high" and "low" (Returns pd.DataFrame for each).

    n_cuts (int)
        4: use quartiles (0.25,  0.5 ,  0.75)
        3: use H/M/L (0.33, 0.66)
    Returns LOW, HIGH
    '''

    bins = np.linspace(0, 1, n_cuts+1)[1:-1]
    low_bin = bins[0]
    high_bin = bins[-1]
    pupil_quantiles = pupildf[feature_name].quantile(bins)
    low_pupil_thr = pupil_quantiles[low_bin]
    high_pupil_thr = pupil_quantiles[high_bin]
    pupil_low = pupildf[pupildf[feature_name]<=low_pupil_thr].copy()
    pupil_high = pupildf[pupildf[feature_name]>=high_pupil_thr].copy()
    # Can also bin into low, mid, high
    #pupildf['quantile'] = pd.qcut(pupildf[face_feature], n_cuts, labels=False)
    
    if return_bins:
        return bins, pupil_low, pupil_high
    else:
        return pupil_low, pupil_high


def get_train_configs(sdf, class_name='morphlevel', class_a=0, class_b=106,
                                train_transform_name=None, train_transform_value=None):

    # Get train configs
    if train_transform_name is not None and train_transform_value is not None:
        if not isinstance(train_transform_value, (list, np.array)):
            train_transform_value = [train_transform_value]
        train_configs = sdf[sdf[class_name].isin([class_a, class_b]) &
                           sdf[train_transform_name].isin(train_transform_value)].index.tolist()
    else:
        train_configs = sdf[sdf[class_name].isin([class_a, class_b])].index.tolist()
    # train_trials = sorted(ndata[ndata['config'].isin(train_configs)]['trial'].unique())
    
    return train_configs


def add_configs_to_pupildf(ndata, pdata, sdf):
    '''
    Assign config to each trial in pupildf    
    # If no match, might be some incorrect alignment 
    '''
    ntrials_total, ncols = ndata.shape
    ntrials_no_pupil = pdata.dropna().shape[0]
    # Make sure pupil trials are same as neural trials:
    ntrials_misaligned=0
    if sorted(ndata.index.tolist())!=sorted(pdata['trial'].unique()):
        ndata, pdata = match_neural_and_pupil_trials(ndata, pdata, equalize_conditions=False)
        ntrials_misaligned = ntrials_total - ndata.shape[0]
      
    # Addd config info to pupildata
    if 'trial' not in ndata.columns:
        ndata['trial'] = ndata.index.tolist()
    pdata['config'] = [ndata[ndata['trial']==t]['config'].unique()[0] for t in pdata['trial']]
    pdata['n_trials_total'] = ntrials_total
    pdata['n_trials_nodata'] = ntrials_total - ntrials_no_pupil
    pdata['n_trials_misaligned'] = ntrials_misaligned 

    # Add some meta info
    pdata['size'] = [sdf['size'][c] for c in pdata['config']]
    pdata['morphlevel'] = [sdf['morphlevel'][c] for c in pdata['config']]

    pdata = pdata.drop(['frame', 'frame_int', 'frame_ix'], axis=1)
    return ndata, pdata


def get_valid_neuraldata_and_pupildata(pupildata, MEANS, SDF, verbose=False, return_valid_only=True,
                            class_name='morphlevel', class_a=0, class_b=106, 
                            train_transform_name=None, train_transform_value=None, 
                            experiment='blobs', traceid='traces001'):
    '''
    pupildata (dict):  keys are datakeys, values are dataframe of pupil info (all trials)
    MEANS (dict):  keys are datakeys, cells not split by area here, just need the trial nums
    SDF (dict): stim config dfs for each datakey
    
    # If no match, might be some incorrect alignment 
    # formerly:  add_stimuli_to_pupildf()

    '''
    stim_datakeys = pupildata.keys()
    _, renamed_configs = aggr.check_sdfs(stim_datakeys, experiment=experiment, 
                                    traceid=traceid, images_only=True, rename=True, return_incorrect=True) 
    bad_alignment=[] 
    for datakey, pdata0 in pupildata.items():
        pdata = pdata0.copy()
        if datakey not in MEANS.keys():
            print("Missing <%s> from MEANS dict. Skipping (dropping from pupil dict)." % datakey)
            pupildata.pop(datakey, None)
            continue
        ndata = MEANS[datakey].copy()
        ntrials_total, ncols = ndata.shape
        sdf = SDF[datakey].copy()

        # Add configs
        ndata, pdata = add_configs_to_pupildf(ndata, pdata, sdf)
        n_bad = float(pdata['n_trials_misaligned'].unique())
        if n_bad > 0:
            print("Warning: %i misaligned trials (%s)" % (n_bad, datakey))
            bad_alignment.append((datakey, n_bad))

        # Count train configs
        train_configs = get_train_configs(sdf, class_name=class_name, 
                                    class_a=class_a, class_b=class_b,
                                    train_transform_name=train_transform_name, 
                                    train_transform_value=train_transform_value)
        # Get counts
        n_train_trials = pdata[pdata.config.isin(train_configs)].shape[0]
        n_train_trials_incl = pdata.dropna()[pdata.config.isin(train_configs)].shape[0]
        pdata['n_train_trials'] = n_train_trials
        pdata['n_train_trials_dropped'] = n_train_trials - n_train_trials_incl

        # Remove neural trials that don't have valid pupil data 
        ndata_match, pdata_match = match_neural_and_pupil_trials(ndata, pdata.dropna(), equalize_conditions=False)  
        ntrials_dropped = ntrials_total - ndata_match.shape[0]
        
        # Add some meta info
        pdata['n_trials_dropped'] = ntrials_dropped
        pdata['datakey'] = datakey
       
        if verbose and (ntrials_total != ndata.shape[0]):
            print('... %s: Dropping %i of %i trials' \
                        % (datakey, ntrials_dropped, ntrials_total))
            
        if return_valid_only:
            pupil_ = pdata.loc[pdata_match.index]
            neural_ = ndata.loc[ndata_match.index]
        else:
            pupil_ = pdata.copy()
            neural_ = ndata.copy()

        MEANS[datakey] = neural_
        pupildata[datakey] = pupil_
        
    return pupildata, MEANS, bad_alignment


# ===================================================================
# Neural trace processing (should prob go somewhere else)
# ====================================================================
def resample_neural_traces(roi_traces, labels=None, in_rate=44.65, out_rate=20.0, 
                           zscore=True, return_labels=True):

    # Create trial mat, downsampled: shape = (ntrials, nframes_per_trial)
    trialmat = pd.DataFrame(np.vstack([roi_traces[tg.index] for trial, tg in labels.groupby(['trial'])]),\
                            index=[int(trial[5:]) for trial, tg in labels.groupby(['trial'])])

    #### Bin traces - Each tbin is a column, each row is a sample 
    sample_data = trialmat.fillna(method='pad').copy()
    ntrials, nframes_per_trial = sample_data.shape

    #### Get resampled indices of trial epochs
    #print("%i frames/trial" % nframes_per_trial)
    out_tpoints, out_ixs = resample_traces(np.arange(0, nframes_per_trial), 
                                           in_rate=in_rate, out_rate=out_rate)
    
    #### Bin traces - Each tbin is a column, each row is a sample 
    df = trialmat.fillna(method='pad').copy().T
    xdf = df.reindex(df.index.union(out_ixs)).interpolate('values').loc[out_ixs]
    binned_trialmat = xdf.T
    n_tbins = binned_trialmat.shape[1]

    #### Zscore traces 
    if zscore:
        traces_r = binned_trialmat / binned_trialmat.values.ravel().std()
    else:
        traces_r = binned_trialmat.copy()
        
    # Reshape roi traces
    curr_roi_traces = traces_r.T.unstack().reset_index() # level_0=trial number, level_1=frame number
    curr_roi_traces.rename(columns={0: roi_traces.name}, inplace=True)
    
    if return_labels:
        configs_on_included_trials = [tg['config'].unique()[0] for trial, tg in labels.groupby(['trial'])]
        included_trials = [trial for trial, tg in labels.groupby(['trial'])]
        cfg_list = np.hstack([[c for _ in np.arange(0, n_tbins)] for c in configs_on_included_trials])
        curr_roi_traces.rename(columns={'level_0': 'trial', 'level_1': 'frame_interp'}, inplace=True)
        curr_roi_traces['config'] = cfg_list
        return curr_roi_traces
    else:
        return curr_roi_traces[roi_traces.name]

def resample_labels(labels, in_rate=44.65, out_rate=20):
    # Create trial mat, downsampled: shape = (ntrials, nframes_per_trial)
    trialmat = pd.DataFrame(np.vstack([tg.index for trial, tg in labels.groupby(['trial'])]),\
                            index=[int(trial[5:]) for trial, tg in labels.groupby(['trial'])])
    configs_on_included_trials = [tg['config'].unique()[0] for trial, tg in labels.groupby(['trial'])]
    included_trials = [trial for trial, tg in labels.groupby(['trial'])]

    #### Bin traces - Each tbin is a column, each row is a sample 
    sample_data = trialmat.fillna(method='pad').copy()
    ntrials, nframes_per_trial = sample_data.shape
    

    #### Get resampled indices of trial epochs
    print("%i frames/trial" % nframes_per_trial)
    out_tpoints, out_ixs = resample_traces(np.arange(0, nframes_per_trial), 
                                           in_rate=in_rate, out_rate=out_rate)
    
    #### Bin traces - Each tbin is a column, each row is a sample 
    df = trialmat.fillna(method='pad').copy().T
    xdf = df.reindex(df.index.union(out_ixs)).interpolate('values').loc[out_ixs]
    binned_trialmat = xdf.T
    n_tbins = binned_trialmat.shape[1]

    # Reshape roi traces
    curr_roi_traces = binned_trialmat.T.unstack().reset_index() # level_0=trial number, level_1=frame number
    curr_roi_traces.rename(columns={0: 'index'}, inplace=True)
    

    cfg_list = np.hstack([[c for _ in np.arange(0, n_tbins)] for c in configs_on_included_trials])
    curr_roi_traces.rename(columns={'level_0': 'trial', 'level_1': 'frame_interp'}, inplace=True)
    curr_roi_traces['config'] = cfg_list
    
    return curr_roi_traces

def roi_traces_to_trialmat(curr_roi_traces, trial_ixs):
    '''Assumes that label info in curr_roi_traces dataframe (return_labels=True, for resample_neural_traces())
    '''
    rid = [i for i in curr_roi_traces.columns if hutils.isnumber(i)][0]
    
    curr_ntrials = len(trial_ixs)
    curr_nframes = curr_roi_traces[curr_roi_traces['trial'].isin(trial_ixs)][rid].shape[0]/curr_ntrials
    trial_tmat = curr_roi_traces[curr_roi_traces['trial'].isin(trial_ixs)][rid].reshape((curr_ntrials,curr_nframes))
    
    return trial_tmat

import multiprocessing as mp
from functools import partial 

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_


def apply_to_columns(df, labels, in_rate=44.65, out_rate=20, zscore=True):
    print("is MP")
    df = df.T
    curr_rois = df.columns
    
    newdf = pd.concat([resample_neural_traces(df[x], labels, in_rate=framerate, out_rate=face_framerate, 
                                             zscore=zscore, return_labels=False) for x in curr_rois])
    return newdf
    
def resample_roi_traces_mp(df, labels, in_rate=44.65, out_rate=20., zscore=True, n_processes=4):
    #cart_x=None, cart_y=None, sphr_th=None, sphr_ph=None,
    #                      row_vals=None, col_vals=None, resolution=None, n_processes=4):
    results = []
    terminating = mp.Event()
    
    df_split = np.array_split(df.T, n_processes)
    pool = mp.Pool(processes=n_processes, initializer=initializer, initargs=(terminating,))
    try:
        results = pool.map(partial(apply_to_columns, labels=labels,
                                   in_rate=in_rate, out_rate=out_rate, zscore=zscore), df_split)
        print("done!")
    except KeyboardInterrupt:
        pool.terminate()
        print("terminating")
    finally:
        pool.close()
        pool.join()
  
    print(results[0].shape)
    df = pd.concat(results, axis=1)
    print(df.shape)
    return df #results

def resample_all_roi_traces(traces, labels, in_rate=44.65, out_rate=20.):
    roi_list = traces.columns.tolist()
    configs_on_included_trials = [tg['config'].unique()[0] for trial, tg in labels.groupby(['trial'])]
    included_trials = [trial for trial, tg in labels.groupby(['trial'])]

    r_list=[]

    for ri, rid in enumerate(roi_list):
        if ri%20==0:
            print("... %i of %i cells" % (int(ri+1), len(roi_list)))
            
        # Create trial mat, downsampled: shape = (ntrials, nframes_per_trial)
        trialmat = pd.DataFrame(np.vstack([traces[rid][tg.index] for trial, tg in labels.groupby(['trial'])]),\
                                index=[int(trial[5:]) for trial, tg in labels.groupby(['trial'])])

        #### Bin traces - Each tbin is a column, each row is a sample 
        sample_data = trialmat.fillna(method='pad').copy()
        ntrials, nframes_per_trial = sample_data.shape

        #### Get resampled indices of trial epochs
        out_tpoints, out_ixs = resample_traces(np.arange(0, nframes_per_trial), 
                                               in_rate=in_rate, out_rate=out_rate)

        #### Bin traces - Each tbin is a column, each row is a sample 
        df = sample_data.T
        xdf = df.reindex(df.index.union(out_ixs)).interpolate('values').loc[out_ixs] # Interpolate resampled values
        binned_trialmat = xdf.T # should be Ntrials # Nframes
        n_trials, n_tbins = binned_trialmat.shape

        #### Zscore traces 
        zscored_neural = binned_trialmat / binned_trialmat.values.ravel().std()

        # Reshape roi traces
        cfg_list = np.hstack([[c for _ in np.arange(0, n_tbins)] for c in configs_on_included_trials])
        curr_roi_traces = zscored_neural.T.unstack().reset_index() # level_0=trial number, level_1=frame number
        curr_roi_traces.rename(columns={'level_0': 'trial', 'level_1': 'frame_ix', 0: rid}, inplace=True)
        r_list.append(curr_roi_traces)

    # Combine all traces into 1 dataframe (all frames x nrois)
    traces_r = pd.concat(r_list, axis=1)
    traces_r['config'] = cfg_list

    _, dii = np.unique(traces_r.columns, return_index=True)
    traces_r = traces_r.iloc[:, dii]
    
    return traces_r



