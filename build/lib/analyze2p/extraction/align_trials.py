#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  21 12:17:55 2020

@author: julianarhee
"""
#%%
import matplotlib as mpl
mpl.use('agg')
import h5py
import glob
import os
import json
import copy
import traceback
import re
import optparse
import sys
import operator
import shutil
import pandas as pd
import numpy as np
from functools import reduce

import analyze2p.utils as hutils
import analyze2p.extraction.traces as traceutils
import analyze2p.extraction.paradigm as putils

#from pipeline.python.paradigm.plot_responses import make_clean_psths
#%%
def extract_options(options):
    parser = optparse.OptionParser()
    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--datakey', action='store', dest='datakey', 
                      default='', help='YYYYMMDD_JCxx_fovi')
#    parser.add_option('-S', '--session', action='store', dest='session', default='', 
#                      help='Session (format: YYYYMMDD)')
#    # Set specific session/run for current animal:
#    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', 
#                      help="fov name (default: FOV1_zoom2p0x)")
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") #: FOV1_zoom2p0x)")
    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    parser.add_option('-p', '--iti-pre', action='store', dest='iti_pre', default=1.0, 
                      help="pre-stim amount in sec (default: 1.0)")
    parser.add_option('-P', '--iti-post', action='store', dest='iti_post', default=1.0, 
                      help="post-stim amount in sec (default: 1.0)")

#    parser.add_option('--plot', action='store_true', dest='plot_psth', default=False, 
#                      help="set flat to plot psths (specify row, col, hue)")
#    parser.add_option('-r', '--rows', action='store', dest='rows',
#                          default=None, help='Transform to plot along ROWS (only relevant if >2 trans_types)')
#    parser.add_option('-c', '--columns', action='store', dest='columns',
#                          default=None, help='Transform to plot along COLUMNS')
#    parser.add_option('-H', '--hue', action='store', dest='subplot_hue',
#                          default=None, help='Transform to plot by HUE within each subplot')
#    parser.add_option('-d', '--response', action='store', dest='response_type',
#                          default='dff', help='Traces to plot (default: dff)')
#    parser.add_option('-f', '--filetype', action='store', dest='filetype',
#                          default='svg', help='File type for images [default: svg]')
#    parser.add_option('--resp-test', action='store', dest='responsive_test',
#                          default='nstds', help='Responsive test or plotting rois [default: nstds]')
#    parser.add_option('--resp-thr', action='store', dest='responsive_thr',
#                          default=10, help='Responsive test or plotting rois [default: 10]')
#
    (options, args) = parser.parse_args(options)

    return options

datakey='20190522_JC084_fov1'
experiment='gratings'

def aggregate_experiment_runs(datakey, experiment, 
                        traceid='traces001', 
                        rootdir='/n/coxfs01/2p-data'):
#%%  
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, 
                                    'FOV%i_*' % fovn))[0]
    if int(session) < 20190511 and experiment=='rfs':
        print("This is actually a RFs, but was previously called 'gratings'")
        experiment = 'gratings'
    rawfns = sorted(glob.glob(os.path.join(fovdir, '*%s_*' % experiment, \
                        'traces/%s*' % traceid, 'files', '*.hdf5')), \
                        key=hutils.natural_keys)
    print("[%s]: Found %i raw file arrays." % (experiment, len(rawfns))) 
    #%
    runpaths = sorted(glob.glob(os.path.join(fovdir,
                        '*%s_*' % experiment,
                        'traces/%s*' % traceid, 'files')), \
                            key=hutils.natural_keys)
    assert len(runpaths) > 0, "No extracted traces for run %s (%s)" \
                        % (experiment, traceid) 
    # Get .tif file list with corresponding aux file (i.e., run) index:
    rawfns = [(run_ix, file_ix, fn) for run_ix, rpath \
                in enumerate(sorted(runpaths, key=hutils.natural_keys))\
              for file_ix, fn in enumerate(sorted(glob.glob(os.path.join(rpath, '*.hdf5')), key=hutils.natural_keys))] 
    
    # Check if this run has any excluded tifs
    rundirs = sorted([d for d in glob.glob(os.path.join(fovdir, 
                    '%s_*' % experiment)) if 'combined' not in d \
                    and os.path.isdir(d)], key=hutils.natural_keys)
#%%
    #%% Cycle through all tifs, detrend, then get aligned frames
    dfs = {}
    frame_times=[]; trial_ids=[]; config_ids=[]; sdf_list=[]; 
    run_ids=[]; file_ids=[];
    frame_indices = []
    for total_ix, (run_ix, file_ix, fpath) in enumerate(rawfns):
        print("**** File %i of %i *****" % (int(total_ix+1), len(rawfns)))
        try:
            rfile = h5py.File(fpath, 'r')
            fdata = rfile['Slice01']
            trace_types = list(fdata['traces'].keys())
            for trace_type in trace_types:
                if not any([trace_type in k for k in dfs.keys()]):
                    dfs['%s-detrended' % trace_type] = []
                    dfs['%s-F0' % trace_type] = []
            frames_to_select = pd.DataFrame(fdata['frames_indices'][:])        
            #%
            rundir = rundirs[run_ix]
            tid_fpath = glob.glob(os.path.join(rundir, 'traces', '*.json'))[0]
            with open(tid_fpath, 'r') as f:
                tids = json.load(f)
            excluded_tifs = tids[traceid]['PARAMS']['excluded_tiffs']
            print("*** Excluding:", excluded_tifs)
            currfile = str(re.search(r"File\d{3}", fpath).group())
            if currfile in excluded_tifs:
                print("... skipping...")
                continue 
            # Set output dir
            basedir = os.path.split(os.path.split(fpath)[0])[0] 
            data_array_dir = os.path.join(basedir, 'data_arrays')
            if not os.path.exists(data_array_dir):
                os.makedirs(data_array_dir)          
            #% # Get SCAN IMAGE info for run:
            run_name = os.path.split(rundir)[-1]
            si = get_frame_info(rundir) 
            #% # Load MW info to get stimulus details:
            mw_fpath = glob.glob(os.path.join(rundir, 'paradigm', \
                                'trials_*.json'))[0] # 
            with open(mw_fpath,'r') as m:
                mwinfo = json.load(m)
            trial_keys = list(mwinfo.keys())
            pre_iti_sec = round(mwinfo[trial_keys[0]]['iti_dur_ms']/1E3) 
            nframes_iti_full = int(round(pre_iti_sec * si['volumerate']))
            
            # Load current run's stim configs 
            sconfig_fpath = os.path.join(rundir,'paradigm','stimulus_configs.json')
            with open(sconfig_fpath, 'r') as s:
                stimconfigs = json.load(s)
            cfg_list = list(stimconfigs.keys())
            stim_params = list(stimconfigs[cfg_list[0]].keys())
            if 'frequency' in stim_params:
                stimtype = 'gratings'
            elif 'fps' in stim_params:
                stimtype = 'movie'
            else:
                stimtype = 'image' 
            # Get all trials contained in current .tif file:
            tmp_trials_in_block = sorted([t for t, mdict in mwinfo.items() \
                                    if mdict['block_idx']==file_ix], \
                                    key=hutils.natural_keys)
            # 20181016 BUG: ignore trials that are BLANKS:
            trials_in_block = sorted([t for t in tmp_trials_in_block \
                                    if mwinfo[t]['stimuli']['type']!='blank'],\
                                     key=hutils.natural_keys) 
            frame_shift = 0 if 'block_frame_offset' \
                            not in mwinfo[trials_in_block[0]].keys() \
                            else mwinfo[trials_in_block[0]]['block_frame_offset']
            parsed_frames_fpath = glob.glob(os.path.join(rundir, \
                                    'paradigm', 'parsed_frames_*.hdf5'))[0] 
            frame_ixs = np.array(frames_to_select[0].values) 
            # Assign frames to trials 
            trial_frames_to_vols, relative_tsecs = frames_to_trials(
                                                    parsed_frames_fpath, 
                                                    trials_in_block, 
                                                    file_ix, si, 
                                                    frame_shift=frame_shift, 
                                                    frame_ixs=frame_ixs) 
        #%
            # Get stimulus info for each trial:        
            excluded_params = [k for k in mwinfo[trials_in_block[0]]['stimuli'].keys() if k not in stim_params]
            print("Excluding:", excluded_params)
            curr_stimconfigs = dict((trial, dict((k,v) for k,v \
                                        in mwinfo[trial]['stimuli'].items() \
                                        if k not in excluded_params)) \
                                        for trial in trials_in_block)
            for k, v in curr_stimconfigs.items():
                if v['scale'][0] is not None:
                    curr_stimconfigs[k]['scale'] = [round(v['scale'][0], 1), round(v['scale'][1], 1)]
            for k, v in stimconfigs.items():
                if v['scale'][0] is not None:
                    stimconfigs[k]['scale'] = [round(v['scale'][0], 1), round(v['scale'][1], 1)]
            if stimtype=='image' and 'filepath' in mwinfo['trial00001']['stimuli'].keys():
                for t, v in curr_stimconfigs.items():
                    if 'filename' not in v.keys():
                        curr_stimconfigs[t].update(\
                            {'filename': os.path.split(mwinfo[t]['stimuli']['filepath'])[-1]
                             }
                        )
                        
             # Add stim_dur if included in stim params:          
            varying_stim_dur = False
            if 'stim_dur' in stim_params:
                varying_stim_dur = True
                for ti, trial in enumerate(sorted(trials_in_block, key=hutils.natural_keys)):
                    curr_trial_stimconfigs[trial]['stim_dur'] = round(mwinfo[trial]['stim_dur_ms']/1E3, 1)         
            trial_configs=[]
            for trial, sparams in curr_stimconfigs.items():
                config_name = [k for k, v in stimconfigs.items() \
                                if v==sparams]
                assert len(config_name) == 1, "Bad configs - %s" % trial
                sparams['position'] = tuple(sparams['position'])
                sparams['scale'] = sparams['scale'][0] if not isinstance(sparams['scale'], int) else sparams['scale']
                sparams['config'] = config_name[0]
                trial_configs.append(pd.Series(sparams, name=trial))                
            trial_configs = pd.concat(trial_configs, axis=1).T
            trial_configs = trial_configs.sort_index() #(by=)
 
            # Get corresponding stimulus/trial labels 
            # for each frame in each trial:
            tlength = trial_frames_to_vols.shape[0]
            config_labels = np.hstack([np.tile(trial_configs.T[trial]['config'], (tlength, ))\
                                       for trial in trials_in_block]) 
            trial_labels = np.hstack([np.tile(trial, (tlength,)) \
                                      for trial in trials_in_block]) 
            # Get relevant timecourse points
            frames_in_trials = trial_frames_to_vols.T.values.ravel()            
            #%
            window_size_sec = 30.
            framerate = si['framerate']
            quantile= 0.10
            windowsize = window_size_sec*framerate
                
            for trace_type in trace_types:
                print("... processing trace type: %s" % trace_type)
                # Load raw traces and detrend within .tif file
                df = pd.DataFrame(fdata['traces'][trace_type][:])
               # np_subtracted traces are created in traces/get_traces.py, without offset added.
               # Remove rolling baseline, return detrended traces with offset added back in? 
                detrended_df, F0_df = traceutils.get_rolling_baseline(df, windowsize, quantile=quantile)
                print("Showing initial drift correction (quantile: %.2f)" % quantile)
                print("Min value for all ROIs:", np.min(np.min(detrended_df, axis=0)))
                currdf = detrended_df.loc[frames_in_trials]
                currdf['ix'] = [total_ix for _ in range(currdf.shape[0])]
                dfs['%s-detrended' % trace_type].append(currdf)
                
                currf0 = F0_df.loc[frames_in_trials]
                currf0['ix'] = [total_ix for _ in range(currdf.shape[0])]
                dfs['%s-F0' % trace_type].append(currf0)
                 
            frame_indices.append(frames_in_trials) # added 2019-05-21 
            frame_times.append(relative_tsecs)
            trial_ids.append(trial_labels)
            config_ids.append(config_labels)
            run_ids.append([run_ix for _ in range(len(trial_labels))])
            file_ids.append([file_ix for _ in range(len(trial_labels))])
            
            stimdict = putils.format_stimconfigs(stimconfigs) 
            sdf = pd.DataFrame(stimdict).T
            #sdf = putils.stimdict_to_df(stimconfigs, experiment)
            sdf_list.append(sdf)
        except Exception as e:
            traceback.print_exc()
            print(e)
        finally:
            rfile.close()
#%%
    #% Concatenate all runs into 1 giant dataframe
    trial_list = sorted(mwinfo.keys(), key=hutils.natural_keys)    
    # Make combined stimconfigs
    sdfcombined = pd.concat(sdf_list, axis=0)
    #sdfcombined = sdfcombined.drop('position', 1)     
    if 'position' in sdfcombined.columns:
        sdfcombined['position'] = [tuple(s) for s \
                                    in sdfcombined['position'].values] 
    sdf = sdfcombined.drop_duplicates()
    all_param_names = sorted(sdf.columns.tolist())
    param_names = [p for p in all_param_names if p not in ['position']] 
    sdf = sdf.sort_values(by=sorted(param_names))
    sdf.index = ['config%03d' % int(ci+1) for ci in range(sdf.shape[0])] 
    
    # Rename each run's configs according to combined sconfigs
    new_config_ids=[]
    for ci, (orig_cfgs, orig_sdf) in enumerate(zip(config_ids, sdf_list)):
        if 'position' in orig_sdf.columns:
            orig_sdf['position'] = [tuple(s) for s in orig_sdf['position'].values]
        try:
            merged_ = orig_sdf.reset_index().merge(sdf.reset_index(), 
                                                   on=all_param_names, how='left')
            cfg_lut = dict((k, v) for k, v in merged_[['index_x', 'index_y']].values)            
            # for old_cfg_name in orig_cfgs:
            #     new_cfg_name = sdf[sdf.eq(orig_sdf.loc[old_cfg_name], 
            #                               axis=1).all(axis=1)].index[0]
            #     cfg_cipher[old_cfg_name] = new_cfg_name
            new_config_ids.append([cfg_lut[c] for c in orig_cfgs])
        except Exception as e:
            print(ci)
            raise(e)
    configs = np.hstack(new_config_ids)
    
#%% 
    # Reindex trial numbers in order
    trials = np.hstack(trial_ids)  # Need to reindex trials
    run_ids = np.hstack(run_ids)
    last_trial_num = 0
    for run_id in sorted(np.unique(run_ids)):
        next_run_ixs = np.where(run_ids==run_id)[0]
        old_trial_names = trials[next_run_ixs]
        new_trial_names = ['trial%05d' % int(int(ti[-5:])+last_trial_num) for ti in old_trial_names]
        trials[next_run_ixs] = new_trial_names
        last_trial_num = int(sorted(trials[next_run_ixs], key=hutils.natural_keys)[-1][-5:])
        
    # Check for stim durations
    if 'stim_dur' in stim_params:
        stim_durs = np.array([stimconfigs[c]['stim_dur'] for c in configs])
    else:
        stim_durs = list(set([round(mwinfo[t]['stim_dur_ms']/1e3, 1) for t in trial_list]))
    nframes_on = np.array([int(round(dur*si['volumerate'])) for dur in stim_durs])
    print("Nframes on:", nframes_on)
    print("stim_durs (sec):", stim_durs) 
    # Also collate relevant frame info (i.e., labels):
    tstamps = np.hstack(frame_times)
    f_indices = np.hstack(frame_indices) 
  
#%% 
    #HERE.
    # Get concatenated df for indexing meta info
    trace_keys = list(dfs.keys() )
    roi_list = np.array([r for r in dfs[trace_keys[0]][0].columns.tolist() if hutils.isnumber(r)])
    xdata_df = pd.concat([d[roi_list] for d in dfs[trace_keys[0]]], axis=0).reset_index(drop=True) #drop=True)
    print("XDATA concatenated: %s" % str(xdata_df.shape)) 
     # Turn paradigm info into dataframe: 
    labels_df = pd.DataFrame({'tsec': tstamps, 
                              'frame': f_indices,
                              'config': configs,
                              'trial': trials,
                              'stim_dur': stim_durs #np.tile(stim_dur, trials.shape)
                              }, index=xdata_df.index)
    try:
        ons = [int(np.where(np.array(g['tsec'])==0)[0]) for t, g in labels_df.groupby('trial')]
        assert len(list(set(ons))) == 1
        stim_on_frame = list(set(ons))[0]
    except Exception as e: 
        all_ons = [np.where(np.array(t)==0)[0] for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
        all_ons = np.concatenate(all_ons).ravel()
        unique_ons = np.unique(all_ons)
        print("**** WARNING: multiple stim onset idxs found - %s" % str(list(set(unique_ons))))
        stim_on_frame = int(round( np.mean(unique_ons) ))
        print("--- assigning stim on frame: %i" % stim_on_frame)     
    labels_df['stim_on_frame'] = np.tile(stim_on_frame, (len(tstamps),))
    labels_df['nframes_on'] = np.tile(int(nframes_on), (len(tstamps),))
    labels_df['run_ix'] = run_ids
    labels_df['file_ix'] = np.hstack(file_ids)
    print("*** LABELS:", labels_df.shape)
     
    sconfigs = sdf.T.to_dict()
    
    run_info = get_run_summary(xdata_df, labels_df, stimconfigs, si)
#%% 
    # #########################################################################
    #% Combine all data trace types and save
    # #########################################################################
    # Get combo dir
    existing_combined = glob.glob(os.path.join(fovdir, 'combined_%s_static' % experiment, 
                                               'traces', '%s*' % traceid))
    if len(existing_combined) > 0:
        combined_dir = os.path.join(existing_combined[0], 'data_arrays')
    else:
        combined_traceids = '_'.join([os.path.split(f)[-1] \
                                  for f in [glob.glob(os.path.join(rundir, 'traces', '%s*' % traceid))[0] \
                                            for rundir in rundirs]])
    
        combined_dir = os.path.join(fovdir, 'combined_%s_static' % experiment, 
                                    'traces', combined_traceids, 'data_arrays')
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
         
    labels_fpath = os.path.join(combined_dir, 'labels.npz')
    print("Saving labels data...", labels_fpath)
    ddict = {'sconfigs': sconfigs, 'labels_data': labels_df, 
             'labels_columns': labels_df.columns.tolist(), 'run_info': run_info}
    np.savez(labels_fpath, **ddict, protocol=2)
            
    # Save all the dtypes
    for trace_type in trace_types:
        print(trace_type)
        xdata_df = pd.concat(dfs['%s-detrended' % trace_type], axis=0).reset_index() 
        f0_df = pd.concat(dfs['%s-F0' % trace_type], axis=0).reset_index() 
        roidata = [c for c in xdata_df.columns if hutils.isnumber(c)] #c != 'ix']
        
        data_fpath = os.path.join(combined_dir, '%s.npz' % trace_type)
        print("Saving labels data...", data_fpath)
        np.savez(data_fpath, 
                 data=xdata_df[roidata].values,
                 f0=f0_df[roidata].values,
                 file_ixs=xdata_df['ix'].values,
                 sconfigs=sconfigs,
                 labels_data=labels_df,
                 labels_columns=labels_df.columns.tolist(),
                 run_info=run_info)
        
        del f0_df
        del xdata_df
 
#%% --------------------------------------------------------------------
def load_parsed_trials(parsed_trials_path):
    with open(parsed_trials_path, 'r') as f:
        trialdict = json.load(f)
    return trialdict

def get_frame_info(run_dir):
    '''
    Get acquisition info for all tifs in specified run_dir (1 block).
    '''
    si_info = {}
    run = os.path.split(run_dir)[-1]
    runinfo_path = os.path.join(run_dir, '%s.json' % run)
    with open(runinfo_path, 'r') as fr:
        runinfo = json.load(fr)
    nfiles = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], 
                        key=hutils.natural_keys)
    # Get frame_idxs
    # These are FRAME indices in the current .tif file, i.e.,
    # removed flyback frames and discard frames at the top and 
    # bottom of the volume should not be included in the indices...
    frame_idxs = runinfo['frame_idxs']
    if len(frame_idxs) > 0:
        print("Found %i frames from flyback correction." % len(frame_idxs))
    else:
        frame_idxs = np.arange(0, runinfo['nvolumes'] * len(runinfo['slices']))

    ntiffs = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(ntiffs)], \
                        key=hutils.natural_keys)
    volumerate = runinfo['volume_rate']
    framerate = runinfo['frame_rate']
    nvolumes = runinfo['nvolumes']
    nslices = int(len(runinfo['slices']))
    nchannels = runinfo['nchannels']
    nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
    nframes_per_file = nslices_full * nvolumes
    
    # Get VOLUME indices to assign frame numbers to volumes:
    vol_idxs_file = np.empty((nvolumes*nslices_full,))
    vcounter = 0
    for v in range(nvolumes):
        vol_idxs_file[vcounter:vcounter+nslices_full] = np.ones((nslices_full,))*v
        vcounter += nslices_full
    vol_idxs_file = [int(v) for v in vol_idxs_file]
    vol_idxs = []
    vol_idxs.extend(np.array(vol_idxs_file)+nvolumes*tiffnum \
                    for tiffnum in range(nfiles))
    vol_idxs = np.array(sorted(np.concatenate(vol_idxs).ravel()))

    # Aggregate info
    si_info['nslices_full'] = nslices_full
    si_info['nframes_per_file'] = nframes_per_file
    si_info['vol_idxs'] = vol_idxs
    si_info['volumerate'] = volumerate
    si_info['framerate'] = framerate
    si_info['nslices'] = nslices
    si_info['nchannels'] = nchannels
    si_info['ntiffs'] = ntiffs
    si_info['nvolumes'] = nvolumes
    all_frames_tsecs = runinfo['frame_tstamps_sec']
    if nchannels==2:
        all_frames_tsecs = np.array(all_frames_tsecs[0::2])
    si_info['frames_tsec'] = all_frames_tsecs #runinfo['frame_tstamps_sec']

    return si_info

def get_run_summary(xdata_df, labels_df, stimconfigs, si, verbose=False):
    
    run_info = {}
    transform_dict, object_transformations = putils.get_transforms(stimconfigs)
    trans_types = list(object_transformations.keys())

    conditions = sorted(list(set(labels_df['config'])), key=hutils.natural_keys)
 
    # Get trun info:
    roi_list = sorted(list(set([r for r in xdata_df.columns.tolist() if hutils.isnumber(r)]))) #not r=='index'])))
    ntrials_total = len(sorted(list(set(labels_df['trial'])), key=hutils.natural_keys))
    trial_counts = labels_df.groupby(['config'])['trial'].apply(set)
    ntrials_by_cond = tuple((k, len(trial_counts[i])) for i,k in enumerate(trial_counts.index.tolist()))
    nframes_per_trial = list(set(labels_df.groupby(['trial'])['stim_on_frame'].count())) #[0]
    nframes_on = list(set(labels_df['stim_dur']))
    nframes_on = [int(round(si['framerate'])) * n for n in nframes_on]

    try:
        ons = [int(np.where(np.array(t)==0)[0]) for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
        assert len(list(set(ons))) == 1
        stim_on_frame = list(set(ons))[0]
    except Exception as e: 
        all_ons = [np.where(np.array(t)==0)[0] for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
        all_ons = np.concatenate(all_ons).ravel()
        print("N stim onsets:", len(all_ons))
        unique_ons = np.unique(all_ons)
        print("**** WARNING: multiple stim onset idxs found - %s" % str(list(set(unique_ons))))
        stim_on_frame = int(round( np.mean(unique_ons) ))
        print("--- assigning stim on frame: %i" % stim_on_frame)  
        
    #ons = [int(np.where(t==0)[0]) for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
    #assert len(list(set(ons)))==1, "More than one unique stim ON idx found!"
    #stim_on_frame = list(set(ons))[0]
    if verbose:
        print("-------------------------------------------")
        print("Run summary:")
        print("-------------------------------------------")
        print("N rois:", len(roi_list))
        print("N trials:", ntrials_total)
        print("N frames per trial:", nframes_per_trial)
        print("N trials per stimulus:", ntrials_by_cond)
        print("-------------------------------------------")

    run_info['roi_list'] = roi_list
    run_info['ntrials_total'] = ntrials_total
    run_info['nframes_per_trial'] = nframes_per_trial
    run_info['ntrials_by_cond'] = ntrials_by_cond
    run_info['condition_list'] = conditions
    run_info['stim_on_frame'] = stim_on_frame
    run_info['nframes_on'] = nframes_on
    #run_info['trace_type'] = trace_type
    #run_info['transforms'] = object_transformations
    #run_info['datakey'] = datakey
    run_info['trans_types'] = trans_types
    run_info['framerate'] = si['framerate']
    run_info['nfiles'] = len(labels_df['file_ix'].unique())

    return run_info

def frames_to_trials(parsed_frames_fpath, trials_in_block, file_ix, si, 
                frame_shift=0, frame_ixs=None):
    '''Load parsed_frames.hdf5 and align frames to trials'''
    
    all_frames_tsecs = np.array(si['frames_tsec'])
    nslices_full = int(len(all_frames_tsecs) / si['nvolumes'])
    if si['nchannels']==2:
        all_frames_tsecs = np.array(all_frames_tsecs[0::2])
    print("N tsecs:", len(all_frames_tsecs))
    
    # Get volume indices to assign frame numbers to volumes:
    vol_ixs_tif = np.empty((int(si['nvolumes'])*nslices_full,))
    vcounter = 0
    for v in range(si['nvolumes']):
        vol_ixs_tif[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
        vcounter += nslices_full
    vol_ixs_tif = np.array([int(v) for v in vol_ixs_tif])
    vol_ixs = []
    vol_ixs.extend(np.array(vol_ixs_tif) + si['nvolumes']*tiffnum for tiffnum in range(si['ntiffs']))
    vol_ixs = np.array(sorted(np.concatenate(vol_ixs).ravel()))
 
    try:
        parsed_frames = h5py.File(parsed_frames_fpath, 'r') 
        trial_list = sorted(parsed_frames.keys(), key=hutils.natural_keys)
        print("There are %i total trials across all .tif files." % len(trial_list))     
        # Check if frame indices are indexed relative to full run 
        # (all .tif files) or relative to within-tif (i.e., a "block")
        block_indexed = True
        if all([all(parsed_frames[t]['frames_in_run'][:] == parsed_frames[t]['frames_in_file'][:]) for t in trial_list]):
            block_indexed = False
            print("Frame indices are NOT block indexed") 
        # Assumes all trials have same structure
        min_frame_interval = 1 #list(set(np.diff(frames_to_select['Slice01'].values)))  # 1 if not slices
        nframes_pre = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['baseline_dur_sec'] * si['volumerate']))
        nframes_post = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['iti_dur_sec'] * si['volumerate']))
        nframes_on = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * si['volumerate']))
        nframes_per_trial = nframes_pre + nframes_on + nframes_post 
    
        # Get ALL frames corresponding to trial epochs:
        # -----------------------------------------------------
        # Get all frame indices for trial epochs 
        # (if there are overlapping frame indices, there will be repeats)    
        all_frames_in_trials = np.hstack([np.array(parsed_frames[t]['frames_in_file']) \
                                   for t in trials_in_block])
        print("... N frames to align:", len(all_frames_in_trials))
        print("... N unique frames:", len(np.unique(all_frames_in_trials)))
        stim_onset_idxs = np.array([parsed_frames[t]['frames_in_file'].attrs['stim_on_idx'] \
                                    for t in trials_in_block])
    
        # Since we are cycling thru FILES readjust frame indices 
        # to match within-file, rather than across-files.
        # block_frame_offset set in extract_paradigm_events (MW parsing) 
        # - supposedly set this up to deal with skipped tifs?
        # -------------------------------------------------------
        if block_indexed is False:
            all_frames_in_trials = all_frames_in_trials - len(all_frames_tsecs)*file_ix - frame_shift  
            if all_frames_in_trials[-1] >= len(all_frames_tsecs):
                print('... File: %i (has %i frames)' % (file_ix, len(all_frames_tsecs)))
                print("... asking for %i extra frames..." % (all_frames_in_trials[-1] - len(all_frames_tsecs)))
            stim_onset_idxs = stim_onset_idxs - len(all_frames_tsecs)*file_ix - frame_shift 
        print("... Last frame to align: %i (N frames total, %i)" % (all_frames_in_trials[-1], len(all_frames_tsecs)))
    
        stim_onset_idxs_adjusted = vol_ixs_tif[stim_onset_idxs]
        stim_onset_idxs = copy.copy(stim_onset_idxs_adjusted)
        varying_stim_dur=False
        trial_frames_to_vols = dict((t, []) for t in trials_in_block)
        for t in trials_in_block: 
            frames_to_vols = parsed_frames[t]['frames_in_file'][:] 
            frames_to_vols = frames_to_vols - len(all_frames_tsecs)*file_ix - frame_shift  
            actual_frames_in_trial = [i for i in frames_to_vols if i < len(vol_ixs_tif)]
            trial_vol_ixs = np.empty(frames_to_vols.shape, dtype=int)
            trial_vol_ixs[0:len(actual_frames_in_trial)] = vol_ixs_tif[actual_frames_in_trial]
            if varying_stim_dur is False:
                trial_vol_ixs = trial_vol_ixs[0:nframes_per_trial]
            trial_frames_to_vols[t] = np.array(trial_vol_ixs)
    except Exception as e:
        traceback.print_exc()
    finally:
        parsed_frames.close()            
    #%
    # Convert frame- to volume-reference
    # select 1st frame for each volume. (Only relevant for multi-plane)
    # -------------------------------------------------------
    # Don't take unique values, since stim period of trial N can be ITI of trial N-1
    actual_frames = [i for i in all_frames_in_trials if i < len(vol_ixs_tif)]
    frames_in_trials = vol_ixs_tif[actual_frames]
    
    # Turn frame_tsecs into RELATIVE tstamps (to stim onset):
    # ------------------------------------------------
    first_plane_tstamps = all_frames_tsecs[np.array(frame_ixs)]
    print("... N tstamps:", len(first_plane_tstamps))
    trial_tstamps = first_plane_tstamps[frames_in_trials[0:len(actual_frames)]]  
    # Check whether asking for more frames than unique, and pad array if so
    if len(trial_tstamps) < len(all_frames_in_trials): #len(frames_in_trials):
        print("... padding trial tstamps array... (should be %i)" % len(all_frames_in_trials))
        trial_tstamps = np.pad(trial_tstamps, (0, len(all_frames_in_trials)-len(trial_tstamps)), mode='constant', constant_values=np.nan)
        frames_in_trials = np.pad(frames_in_trials, (0, len(all_frames_in_trials)-len(frames_in_trials)), mode='constant', constant_values=np.nan)

    # All trials have the same structure:
    reformat_tstamps = False
    print("N frames per trial:", nframes_per_trial)
    print("N tstamps:", len(trial_tstamps))
    print("N trials in block:", len(trials_in_block))
    try: 
        tsec_mat = np.reshape(trial_tstamps, (len(trials_in_block), nframes_per_trial))
        # Subtract stim_on tstamp from each frame of each trial (relative tstamp)
        tsec_mat -= np.tile(all_frames_tsecs[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T        
    except Exception as e:
        traceback.print_exc()

    x, y = np.where(tsec_mat==0)
    assert len(list(set(y)))==1, \
            "Incorrect stim onset alignment: %s" % str(list(set(y)))
       
    relative_tsecs = np.reshape(tsec_mat, (len(trials_in_block)*nframes_per_trial, ))
    trial_frames_to_vols = pd.DataFrame(trial_frames_to_vols)

    return trial_frames_to_vols, relative_tsecs

# 
 

#%% ALIGN.
def get_alignment_specs(paradigm_dir, si_info, iti_pre=1.0, iti_post=None, \
                        same_order=False):
    trial_epoch_info = {}
    run = os.path.split(os.path.split(paradigm_dir)[0])[-1] #options.run

    stimorder_fns = None
    first_stimulus_volume_num = None
    vols_per_trial = None
    # Get parsed trials info
    try:
        trial_fn = glob.glob(os.path.join(paradigm_dir, 'trials_*.json'))
        assert len(trial_fn)==1, "Unable to find trials .json in %s" % paradigm_dir
        trial_fn = trial_fn[0]
        parsed_trials_path = os.path.join(paradigm_dir, trial_fn)
        trialdict = load_parsed_trials(parsed_trials_path)
        trial_list = sorted(trialdict.keys(), key=hutils.natural_keys)
        stimtype = trialdict[trial_list[0]]['stimuli']['type']

        # Get presentation info (should be constant across trials and files):
        trial_list = sorted(trialdict.keys(), key=hutils.natural_keys)
        stim_durs = [round((float(trialdict[t]['stim_dur_ms'])/1E3), 1) \
                        for t in trial_list]
        print('Found STIM durs:', list(set(stim_durs)))
        if all([i >= 1.0 for i in stim_durs]): 
            stim_durs = [round(t, 0) for t in stim_durs]
        if len(list(set(stim_durs))) > 1:
            print("more than 1 stim_dur found:", list(set(stim_durs)))
            stim_on_sec =dict((t,round(float(trialdict[t]['stim_dur_ms'])/1E3,1)) \
                            for t in trial_list)
        else:
            stim_on_sec = list(set(stim_durs))[0]
        print('Found STIM durs:', list(set(stim_durs)))
      
        iti_durs = [round(np.floor(float(trialdict[t]['iti_dur_ms'])/1E3), 1) \
                            for t in trial_list]
        print('Found ITI durs:', list(set(iti_durs)))
        if len(list(set(iti_durs))) > 1:
            iti_jitter = round(max(iti_durs) - min(iti_durs)) #1.0 # TMP TMP 
            replace_max = max(list(set(iti_durs))) - iti_jitter
            iti_durs_tmp = list(set(iti_durs))
            max_ix = iti_durs_tmp.index(max(iti_durs))
            iti_durs_tmp[max_ix] = replace_max
            iti_durs_unique = list(set(iti_durs_tmp))
        else:
            iti_durs_unique = list(set(iti_durs))
        print("Unique itis (minus jitter from max):", iti_durs_unique)
        assert len(iti_durs_unique) == 1, "More than 1 iti_dur found..."
        iti_full = iti_durs_unique[0]
        if iti_post is None:
            iti_post = iti_full - iti_pre
        print("ITI POST:", iti_post)
        assert (stim_on_sec + iti_pre + iti_post) <= (stim_on_sec + iti_full), \
                "Requested ITI pre/post+stim_dur too big (iti_full=%.1f, stim_dur=%.1f)" % (iti_full, stim_on_sec)

        # Check whether acquisition method is one-to-one 
        # (1 aux file per SI tif) or single-to-many:
        sample_trial = trial_list[0]
        one_to_one = trialdict[sample_trial]['ntiffs_per_auxfile']==1

    except Exception as e:
        print("Could not find unique trial-file for current run %s..." % run)
        print("Aborting with error:")
        print("--------------------------------------------------------")
        traceback.print_exc()
        print("--------------------------------------------------------")
        return None
    try:
        nframes_iti_pre = int(round(iti_pre * si_info['volumerate'])) 
        nframes_iti_post = int(round(iti_post*si_info['volumerate'])) 
        nframes_iti_full = int(round(iti_full * si_info['volumerate'])) 
        if isinstance(stim_on_sec, dict):
            nframes_on = dict((t,int(round(stim_on_sec[t]*si_info['volumerate']))) \
                            for t in sorted(stim_on_sec.keys(), \
                            key=hutils.natural_keys))
            nframes_post_onset = dict((t, nframes_on+nframes_iti_post) \
                            for t in sorted(stim_on_sec.keys(), \
                                key=hutils.natural_keys))
            vols_per_trial = dict((t, nframes_iti_pre+nframes_on+nframes_iti_post) \
                            for t in sorted(stim_on_sec.keys(), \
                                key=hutils.natural_keys))
        else:
            nframes_on = int(round(stim_on_sec * si_info['framerate'])) 
            nframes_post_onset = nframes_on + nframes_iti_post
            vols_per_trial = nframes_iti_pre + nframes_on + nframes_iti_post
    except Exception as e:
        print("Problem calcuating nframes for trial epochs...")
        traceback.print_exc()
        return None
    trial_epoch_info['stim_on_sec'] = stim_on_sec
    trial_epoch_info['iti_full'] = iti_full
    trial_epoch_info['iti_post'] = iti_post
    trial_epoch_info['iti_pre'] = iti_pre
    trial_epoch_info['one_to_one'] = one_to_one
    trial_epoch_info['nframes_on'] = nframes_on
    trial_epoch_info['nframes_iti_pre'] = nframes_iti_pre
    trial_epoch_info['nframes_iti_post'] = nframes_iti_post
    trial_epoch_info['nframes_iti_full'] = nframes_iti_full
    trial_epoch_info['nframes_post_onset'] = nframes_post_onset
    trial_epoch_info['parsed_trials_source'] = parsed_trials_path
    trial_epoch_info['stimorder_source'] = stimorder_fns
    trial_epoch_info['framerate'] = si_info['framerate']
    trial_epoch_info['volumerate'] = si_info['volumerate']
    trial_epoch_info['first_stimulus_volume_num'] = first_stimulus_volume_num
    trial_epoch_info['vols_per_trial'] = vols_per_trial
    trial_epoch_info['stimtype'] = stimtype
    trial_epoch_info['custom_mw'] = False

    return trial_epoch_info


def assign_frames_to_trials(si_info, trial_info, paradigm_dir, create_new=False):
    '''
    Given scan/acquisition info (si_info) + desired alignment (trial_info)
    assign frames to trial epochs.
    Creates parsed_frames.hdf5 in paradigm_dir.
    '''
    run = os.path.split(os.path.split(paradigm_dir)[0])[-1]
    # First check if parsed frame file already exists:
    found_paths = sorted(glob.glob(os.path.join(paradigm_dir, \
                            'parsed_frames_*.hdf5')), key=hutils.natural_keys)
    # Sort by date modified
    found_paths.sort(key=lambda x: os.stat(x).st_mtime) 
    if len(found_paths) > 0 and create_new is False:
        parsed_frames_fpath = found_paths[-1] # Get most recently modified file
        print("---> Got existing parsed-frames file:", parsed_frames_fpath)
        return parsed_frames_fpath

    print("---> Creating NEW parsed-frames file...")
    parsed_frames_fpath = os.path.join(paradigm_dir, 'parsed_frames.hdf5')

    # 1. Create HDF5 file to store trials in run with stimulus info and frame info:
    parsed_frames = h5py.File(parsed_frames_fpath, 'w')
    parsed_frames.attrs['framerate'] = si_info['framerate'] #framerate
    parsed_frames.attrs['volumerate'] = si_info['volumerate'] #volumerate
    parsed_frames.attrs['baseline_dur'] = trial_info['iti_pre'] #iti_pre
    #run_grp.attrs['creation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if trial_info['custom_mw'] is False:
        trialdict = load_parsed_trials(trial_info['parsed_trials_source'])
        # Get presentation info (should be constant across trials and files):
        trial_list = sorted(trialdict.keys(), key=hutils.natural_keys)

    # 1. Get stimulus presentation order for each TIFF found:
    try:
        trial_counter = 0
        for tiffnum in range(si_info['ntiffs']): #ntiffs):
            currfile= "File%03d" % int(tiffnum+1)
            # Get current trials in block:
            trials_in_file = sorted([t for t in trial_list \
                                if trialdict[t]['block_idx'] == tiffnum], \
                                key=hutils.natural_keys)
            # Get stimulus order:
            stimorder = [trialdict[t]['stimuli'] for t in trials_in_file]
            #print("... %s, %i" % (currfile, len(trials_in_file))) 
            for trialidx,currtrial_in_run in enumerate(trials_in_file):
                currtrial_in_file = 'trial%03d' % int(trialidx+1)
                # Trial in run might not be ntrials-per-file * nfiles
                # Trials with missing pix are ignored.
                if trial_info['custom_mw'] is True:
                    if trialidx==0:
                        first_frame_on = si_info['first_stimulus_volume_num'] 
                    else:
                        first_frame_on += si_info['vols_per_trial']  
                else:
                    no_frame_match = False
                    first_frame_on = int(trialdict[currtrial_in_run]['frame_stim_on'])
                # Get baseline, stim, and post frame indices:
                pre_iti_frame = int(first_frame_on-trial_info['nframes_iti_pre'])
                preframes = list(np.arange(pre_iti_frame, first_frame_on, 1))
                post_iti_start = int(first_frame_on + 1)
                if isinstance(trial_info['nframes_post_onset'], dict):
                    post_iti_stop = int(round(first_frame_on+trial_info['nframes_post_onset'][currtrial_in_run]))
                else:
                    post_iti_stop = int(round(first_frame_on + trial_info['nframes_post_onset']))

                postframes = list(np.arange( post_iti_start, post_iti_stop))
                   
                framenums = [preframes, [first_frame_on], postframes]
                framenums = reduce(operator.add, framenums)
                #print "POST FRAMES:", len(framenums)
                diffs = np.diff(framenums)
                consec = [i for i in np.diff(diffs) if not i==0]
                assert len(consec)==0, \
                        "Bad frame parsing in %s, %s, frames: %s " \
                        % (currtrial_in_run, currfile, str(framenums))
                # Create dataset for current trial with frame indices:
                fridxs_in_file = parsed_frames.create_dataset('/'.join((currtrial_in_run, 'frames_in_file')), np.array(framenums).shape, np.array(framenums).dtype)
                fridxs_in_file[...] = np.array(framenums)
                fridxs_in_file.attrs['trial'] = currtrial_in_run
                fridxs_in_file.attrs['trial_idx_in_file'] = trialidx
                fridxs_in_file.attrs['aux_file_idx'] = tiffnum
                fridxs_in_file.attrs['stim_on_idx'] = first_frame_on

                if trial_info['one_to_one'] is True:
                    framenums_in_run = np.array(framenums) + (si_info['nframes_per_file']*tiffnum)
                    abs_stim_on_idx = first_frame_on + (si_info['nframes_per_file']*tiffnum)
                else:
                    framenums_in_run = np.array(framenums)
                    abs_stim_on_idx = first_frame_on

                fridxs = parsed_frames.create_dataset('/'.join((currtrial_in_run, 'frames_in_run')), np.array(framenums_in_run).shape, np.array(framenums_in_run).dtype)
                fridxs[...] = np.array(framenums_in_run)
                fridxs.attrs['trial'] = currtrial_in_run
                fridxs.attrs['aux_file_idx'] = tiffnum
                fridxs.attrs['stim_on_idx'] = abs_stim_on_idx
                if isinstance(trial_info['stim_on_sec'], dict):
                    fridxs.attrs['stim_dur_sec'] = trial_info['stim_on_sec'][currtrial_in_run]
                else: 
                    fridxs.attrs['stim_dur_sec'] = trial_info['stim_on_sec']
                fridxs.attrs['iti_dur_sec'] = trial_info['iti_post']
                fridxs.attrs['baseline_dur_sec'] = trial_info['iti_pre']
    except Exception as e:
        print(e)
        print("Error parsing frames into trials: current file - %s" % currfile)
        print("%s in tiff file %s (%i trial out of total in run)." % (currtrial_in_file, currfile, trial_counter))
        traceback.print_exc()
        print("---------------------------------------------------------------")
    finally:
        parsed_frames.close()

    # Get unique hash for current PARSED FRAMES file:
    parsed_frames_hash = hutils.hash_file(parsed_frames_fpath, hashtype='sha1')

    # Check existing files:
    outdir = os.path.split(parsed_frames_fpath)[0]
    existing_files = [f for f in os.listdir(outdir) if 'parsed_frames_' \
                        in f and f.endswith('hdf5') and parsed_frames_hash not in f]
    if len(existing_files) > 0:
        old = os.path.join(os.path.split(outdir)[0], 'paradigm', 'old')
        if not os.path.exists(old):
            os.makedirs(old)

        for f in existing_files:
            shutil.move(os.path.join(outdir, f), os.path.join(old, f))

    if parsed_frames_hash not in parsed_frames_fpath:
        parsed_frames_filepath = hutils.hash_file_read_only(parsed_frames_fpath)

    print("Finished assigning frames across all tiffs to trials in run %s." % run)
    print("Saved parsed frame info to file:", parsed_frames_fpath)
    print("---------------------------------------------------------------------")

    return parsed_frames_filepath


def parse_trial_epochs(datakey, experiment, traceid, 
                        iti_pre=1.0, iti_post=1.0, 
                        rootdir='/n/coxfs01/2p-data'):
    '''
    For each block or run of EXPERIMENT, parse frames to trials.
    Load desired alignment specs. Assign frames to trial epochs. 
    
    Creates parse_frames.h5py
    Creates/updates event_alignment.json and extraction_params.json
    '''
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    fovdir = glob.glob(os.path.join(rootdir, animalid, 
                                    session, 'FOV%i_*' % fovn))[0]
    rundirs = [os.path.split(p)[0] for p in glob.glob(os.path.join(fovdir, \
                                    '%s_run*' % experiment, 'paradigm'))]
    for rundir in rundirs:
        si_info = get_frame_info(rundir) 
        paradigm_dir = os.path.join(rundir, 'paradigm')  
        trial_info = get_alignment_specs(paradigm_dir, si_info, \
                                iti_pre=iti_pre, iti_post=iti_post) 
        # Save alignment info
        traceid_dir = glob.glob(os.path.join(rundir, 'traces', '%s*' % traceid))
        assert len(traceid_dir)==1, "More than 1 tracedid path found..."
        traceid_dir = traceid_dir[0]
        alignment_info_filepath = os.path.join(traceid_dir, 'event_alignment.json')
        with open(alignment_info_filepath, 'w') as f:
            json.dump(trial_info, f, sort_keys=True, indent=4)       
        # Update extraction_params.json
        extraction_info_fpath = os.path.join(traceid_dir, \
                                                'extraction_params.json')
        if os.path.exists(extraction_info_fpath):
            with open(extraction_info_fpath, 'r') as f:
                eparams = json.load(f)
            for k, v in trial_info.items():
                if k in eparams:
                    eparams[k] = v
        else:
            eparams = trial_info
        with open(extraction_info_fpath, 'w') as f:
            json.dump(eparams, f, sort_keys=True, indent=4)       

        # Get parsed frame indices
        parsed_frames_filepath = assign_frames_to_trials(
                                        si_info, 
                                        trial_info, paradigm_dir, 
                                        create_new=True)

    print("Done!")
  
    return
 
def remake_dataframes(datakey, experiment, traceid, 
                      rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(datakey) 
    experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment
    soma_fpath = traceutils.get_data_fpath(datakey, 
                                experiment_name=experiment_name, 
                                traceid=traceid, 
                                trace_type='np_subtracted', rootdir=rootdir)

    tr, lb, rinfo, sdf = traceutils.load_dataset(soma_fpath, 
                                    trace_type='dff', 
                                    add_offset=True, 
                                    make_equal=False, create_new=True)
    
    print("Remade all the other dataframes.")
    return

#%%
def main(options):
    opts = extract_options(options)

    experiment = opts.experiment
    traceid = opts.traceid
    iti_pre = float(opts.iti_pre)
    iti_post = float(opts.iti_post)
    rootdir = opts.rootdir

    #plot_psth = opts.plot_psth    
  
    datakey = opts.datakey

    print("1. Parsing") 
    parse_trial_epochs(datakey, experiment, traceid, 
                        iti_pre=iti_pre, iti_post=iti_post)

    print("2. Aligning - %s" % experiment)
    aggregate_experiment_runs(datakey, experiment, traceid=traceid)
    remake_dataframes(datakey, experiment, traceid, rootdir=rootdir)
    print("Aligned traces!") 
   
#    if plot_psth:
#        print("3. Plotting")
#        row_str = opts.rows
#        col_str = opts.columns
#        hue_str = opts.subplot_hue
#        response_type = opts.response_type
#        file_type = opts.filetype
#        responsive_test=opts.responsive_test
#        responsive_thr=opts.responsive_thr
#
#        plot_opts = ['-i', datakey, '-t', traceid, 
#                     '-R', 'combined_%s_static' % experiment, 
#                     '--shade', '-r', row_str, '-c', col_str, '-H', hue_str, '-d', response_type, '-f', file_type,
#                    '--responsive', '--test', responsive_test, '--thr', responsive_thr]
#        #make_clean_psths(plot_opts) 
#   
   
#%% 
if __name__ == '__main__':
    main(sys.argv[1:])
    



# %%
