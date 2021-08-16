#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:01:55 2019

@author: julianarhee
"""
#%%
import h5py
import glob
import os
import json
import copy
import traceback
import re
import optparse
import sys

import pandas as pd
import numpy as np
#from pipeline.python.utils import natural_keys, get_frame_info, isnumber
#from pipeline.python.paradigm import utils as putils

import analyze2p.utils as hutils
import analyze2p.extraction.paradigm as putils

#%%
#%%
#%%


rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084'
session = '20190522'
fov = 'FOV1_zoom2p0x'

experiment = 'rfs'
traceid = 'traces001'





def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
                      help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', 
                      help='Session (format: YYYYMMDD)')
    # Set specific session/run for current animal:
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', 
                      help="fov name (default: FOV1_zoom2p0x)")
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") #: FOV1_zoom2p0x)")
    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")

    (options, args) = parser.parse_args(options)

    return options

#%%

def aggregate_experiment_runs(animalid, session, fov, experiment, traceid='traces001'):
#%%    
    fovdir = os.path.join(rootdir, animalid, session, fov)
    if int(session) < 20190511 and experiment=='rfs':
        print("This is actually a RFs, but was previously called 'gratings'")
        experiment = 'gratings'

    rawfns = sorted(glob.glob(os.path.join(fovdir, '*%s_*' % experiment, 'traces', '%s*' % traceid, 'files', '*.hdf5')), key=hutils.natural_keys)
    print("[%s]: Found %i raw file arrays." % (experiment, len(rawfns)))
    
    #%
    runpaths = sorted(glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s_*' % experiment,
                              'traces', '%s*' % traceid, 'files')), key=hutils.natural_keys)
    assert len(runpaths) > 0, "No extracted traces for run %s (%s)" % (experiment, traceid)
    
    # Get .tif file list with corresponding aux file (i.e., run) index:
    rawfns = [(run_ix, file_ix, fn) for run_ix, rpath in enumerate(sorted(runpaths, key=hutils.natural_keys))\
              for file_ix, fn in enumerate(sorted(glob.glob(os.path.join(rpath, '*.hdf5')), key=hutils.natural_keys))]
    
    
    # Check if this run has any excluded tifs
    rundirs = sorted([d for d in glob.glob(os.path.join(rootdir, animalid, session, fov, '%s_*' % experiment))\
              if 'combined' not in d and os.path.isdir(d)], key=hutils.natural_keys)

#%%
    # #########################################################################
    #% Cycle through all tifs, detrend, then get aligned frames
    # #########################################################################
    dfs = {}
    frame_times=[]; trial_ids=[]; config_ids=[]; sdf_list=[]; run_ids=[]; file_ids=[];
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
            print "*** Excluding:", excluded_tifs
            currfile = str(re.search(r"File\d{3}", fpath).group())
            if currfile in excluded_tifs:
                print "... skipping..."
                continue
            
            basedir = os.path.split(os.path.split(fpath)[0])[0]
            
            # Set output dir
            data_array_dir = os.path.join(basedir, 'data_arrays')
            if not os.path.exists(data_array_dir):
                os.makedirs(data_array_dir)
                    
            #% # Get SCAN IMAGE info for run:
            run_name = os.path.split(rundir)[-1]
            si = get_frame_info(rundir)
            
            #% # Load MW info to get stimulus details:
            mw_fpath = glob.glob(os.path.join(rundir, 'paradigm', 'trials_*.json'))[0] # 
            with open(mw_fpath,'r') as m:
                mwinfo = json.load(m)
            pre_iti_sec = round(mwinfo[mwinfo.keys()[0]]['iti_dur_ms']/1E3) 
            nframes_iti_full = int(round(pre_iti_sec * si['volumerate']))
            
            with open(os.path.join(rundir, 'paradigm', 'stimulus_configs.json'), 'r') as s:
                stimconfigs = json.load(s)
            if 'frequency' in stimconfigs[stimconfigs.keys()[0]].keys():
                stimtype = 'gratings'
            elif 'fps' in stimconfigs[stimconfigs.keys()[0]].keys():
                stimtype = 'movie'
            else:
                stimtype = 'image'
       
            # Get all trials contained in current .tif file:
            tmp_trials_in_block = sorted([t for t, mdict in mwinfo.items() if mdict['block_idx']==file_ix], key=hutils.natural_keys)
            # 20181016 BUG: ignore trials that are BLANKS:
            trials_in_block = sorted([t for t in tmp_trials_in_block if mwinfo[t]['stimuli']['type'] != 'blank'], key=hutils.natural_keys)
        
            frame_shift = 0 if 'block_frame_offset' not in mwinfo[trials_in_block[0]].keys() else mwinfo[trials_in_block[0]]['block_frame_offset']
            parsed_frames_fpath = glob.glob(os.path.join(rundir, 'paradigm', 'parsed_frames_*.hdf5'))[0] #' in pfn][0]
            frame_ixs = np.array(frames_to_select[0].values)
            
            # Assign frames to trials 
            trial_frames_to_vols, relative_tsecs = frames_to_trials(parsed_frames_fpath, trials_in_block, file_ix,
                                                                    si, frame_shift=frame_shift, frame_ixs=frame_ixs)
        
        #%
            # Get stimulus info for each trial:        
            # -----------------------------------------------------
            excluded_params = [k for k in mwinfo[trials_in_block[0]]['stimuli'].keys() if k not in stimconfigs['config001'].keys()]
            print("Excluding:", excluded_params)
            #if 'filename' in stimconfigs['config001'].keys() and stimtype=='image':
            #    excluded_params.append('filename')
            curr_trial_stimconfigs = dict((trial, dict((k,v) for k,v in mwinfo[trial]['stimuli'].items() \
                                           if k not in excluded_params)) for trial in trials_in_block)
            for k, v in curr_trial_stimconfigs.items():
                if v['scale'][0] is not None:
                    curr_trial_stimconfigs[k]['scale'] = [round(v['scale'][0], 1), round(v['scale'][1], 1)]
            for k, v in stimconfigs.items():
                if v['scale'][0] is not None:
                    stimconfigs[k]['scale'] = [round(v['scale'][0], 1), round(v['scale'][1], 1)]
            if stimtype=='image' and 'filepath' in mwinfo['trial00001']['stimuli'].keys():
                for t, v in curr_trial_stimconfigs.items():
                    if 'filename' not in v.keys():
                        curr_trial_stimconfigs[t].update({'filename': os.path.split(mwinfo[t]['stimuli']['filepath'])[-1]})
            
            varying_stim_dur = False
            # Add stim_dur if included in stim params:
            if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
                varying_stim_dur = True
                for ti, trial in enumerate(sorted(trials_in_block, key=hutils.natural_keys)):
                    curr_trial_stimconfigs[trial]['stim_dur'] = round(mwinfo[trial]['stim_dur_ms']/1E3, 1)
        
            trial_configs=[]
            for trial, sparams in curr_trial_stimconfigs.items():
                #print sparams.keys()
                #print "-------"
                config_name = [k for k, v in stimconfigs.items() if v==sparams]
                #print v.keys()
                assert len(config_name) == 1, "Bad configs - %s" % trial
                #config_name = config_name[0]
                sparams['position'] = tuple(sparams['position'])
                sparams['scale'] = sparams['scale'][0] if not isinstance(sparams['scale'], int) else sparams['scale']
                sparams['config'] = config_name[0]
                trial_configs.append(pd.Series(sparams, name=trial))
                
            trial_configs = pd.concat(trial_configs, axis=1).T
            trial_configs = trial_configs.sort_index() #(by=)
            
            # Get corresponding stimulus/trial labels for each frame in each trial:
            # --------------------------------------------------------------        
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

                # If trace_type is np_subtracted, need to add original offset back in, first.
                # np_subtracted traces are created in traces/get_traces.py, without offset added.
                #if trace_type == 'np_subtracted':
                #    print "Adding offset for np_subtracted traces"
                #    orig_offset = pd.DataFrame(fdata['traces']['raw'][:]).mean.mean()
                #    df = df + orig_offset

                # Remove rolling baseline, return detrended traces with offset added back in? 
                detrended_df, F0_df = traceutils.get_rolling_baseline(df, windowsize, quantile=quantile)
                print "Showing initial drift correction (quantile: %.2f)" % quantile
                print "Min value for all ROIs:", np.min(np.min(detrended_df, axis=0))
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
            
            sdf = pd.DataFrame(putils.format_stimconfigs(stimconfigs)).T
            sdf_list.append(sdf)
        except Exception as e:
            traceback.print_exc()
            print(e)
        finally:
            rfile.close()

#%%
    # #########################################################################
    #% Concatenate all runs into 1 giant dataframe
    # #########################################################################
    trial_list = sorted(mwinfo.keys(), key=hutils.natural_keys)
    
    # Make combined stimconfigs
    sdfcombined = pd.concat(sdf_list, axis=0)
    if 'position' in sdfcombined.columns:
        sdfcombined['position'] = [tuple(s) for s in sdfcombined['position'].values]
    sdf = sdfcombined.drop_duplicates()
    param_names = sorted(sdf.columns.tolist())
    sdf = sdf.sort_values(by=sorted(param_names))
    sdf.index = ['config%03d' % int(ci+1) for ci in range(sdf.shape[0])]
    
    # Rename each run's configs according to combined sconfigs
    new_config_ids=[]
    for orig_cfgs, orig_sdf in zip(config_ids, sdf_list):
        if 'position' in orig_sdf.columns:
            orig_sdf['position'] = [tuple(s) for s in orig_sdf['position'].values]
        cfg_cipher= {}
        for old_cfg_name in orig_cfgs:
            new_cfg_name = sdf[sdf.eq(orig_sdf.loc[old_cfg_name], axis=1).all(axis=1)].index[0]
            cfg_cipher[old_cfg_name] = new_cfg_name
        new_config_ids.append([cfg_cipher[c] for c in orig_cfgs])
    configs = np.hstack(new_config_ids)
            
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
    if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
        stim_durs = np.array([stimconfigs[c]['stim_dur'] for c in configs])
    else:
        stim_durs = list(set([round(mwinfo[t]['stim_dur_ms']/1e3, 1) for t in trial_list]))
    nframes_on = np.array([int(round(dur*si['volumerate'])) for dur in stim_durs])
    print "Nframes on:", nframes_on
    print "stim_durs (sec):", stim_durs
    
    # Also collate relevant frame info (i.e., labels):
    tstamps = np.hstack(frame_times)
    f_indices = np.hstack(frame_indices) 
  
#%% 
    #HERE.
    # Get concatenated df for indexing meta info
    roi_list = np.array([r for r in dfs[dfs.keys()[0]][0].columns.tolist() if hutils.isnumber(r)])
    xdata_df = pd.concat([d[roi_list] for d in dfs[dfs.keys()[0]]], axis=0).reset_index(drop=True) #drop=True)
    print "XDATA concatenated: %s" % str(xdata_df.shape)
    
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
    print "Saving labels data...", labels_fpath
    np.savez(labels_fpath, 
             sconfigs = sconfigs,
             labels_data=labels_df,
             labels_columns=labels_df.columns.tolist(),
             run_info=run_info)
    
    # Save all the dtypes
    for trace_type in trace_types:
        print trace_type
        xdata_df = pd.concat(dfs['%s-detrended' % trace_type], axis=0).reset_index() 
        f0_df = pd.concat(dfs['%s-F0' % trace_type], axis=0).reset_index() 
        roidata = [c for c in xdata_df.columns if hutils.isnumber(c)] #c != 'ix']
        
        data_fpath = os.path.join(combined_dir, '%s.npz' % trace_type)
        print "Saving labels data...", data_fpath
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


#%%

def main(options):
    opts = extract_options(options)
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    experiment = opts.experiment
    traceid = opts.traceid
    
    aggregate_experiment_runs(animalid, session, fov, experiment, traceid=traceid)


    
if __name__ == '__main__':
    main(sys.argv[1:])
    



