#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:23:06 2018

@author: juliana
"""

import os
import glob
import traceback
import json
import h5py
import random
import math

import tifffile as tf
import numpy as np
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)

import scipy.stats as spstats
import analyze2p.utils as hutils
import _pickle as pkl

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def get_traceid_dir(datakey, experiment, traceid='traces001',
                    rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    experiment_name = 'gratings' if (experiment in ['rfs', 'rfs10'] \
                        and int(session)<20190511) else experiment
    try:
        traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                            'FOV%i_*' % fovnum, 
                            'combined_%s_static' % experiment_name,
                            'traces', '%s*' % traceid))[0]
    except Exception as e:
        print(e)
        print("%s: no traceid (%s/%s)!" \
                % (datakey, experiment, experiment_name))
        return None

    return traceid_dir

def get_data_fpath(datakey, experiment_name='rfs10', traceid='traces001', 
                    trace_type='corrected',
                    rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    run_str = 'combined_%s_' % experiment_name if experiment_name!='retino' \
                else experiment_name
    data_fpath=None
    try:
        # Get traceid dir
        data_dir = glob.glob(os.path.join(rootdir, animalid, session, \
                        'FOV%i_*' % fovnum, '%s*' % run_str, 
                        'traces/%s*' % traceid, 'data_arrays'))[0]

        data_fpath = os.path.join(data_dir, '%s.npz' % trace_type)
        assert os.path.exists(data_fpath), 'No fpath for [%s]' % trace_type

    except AssertionError as e:
        found_fpaths = glob.glob(os.path.join(data_dir, '*.npz'))
        print("Found the following files in dir\n    %s" % data_dir)
        for f in found_fpaths:
            print("    %s" % os.path.split(f)[-1])
    except IndexError as e:
        print("No data dir: %s, %s, %s" % (datakey, experiment_name, traceid))
    
    except Exception as e:
        traceback.print_exc()

    return data_fpath 


def get_trial_alignment(datakey, curr_exp, traceid='traces001',
        rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    try:
        extraction_files = sorted(glob.glob(os.path.join(rootdir, 
                                animalid, session, 'FOV%i*' % fovn, 
                                '*%s*' % curr_exp, 'traces', 
                                '%s*' % traceid,'event_alignment.json')), 
                                key=hutils.natural_keys) 
        assert len(extraction_files) > 0, \
            "(%s, %s) No extraction info found..." % (datakey, curr_exp)
    except AssertionError:
        return None
    
    for i, ifile in enumerate(extraction_files):
        with open(ifile, 'r') as f:
            info = json.load(f)
        if i==0:
            infodict = dict((k, [v]) for k, v in info.items() if hutils.isnumber(v)) 
        else:
            for k, v in info.items():
                if hutils.isnumber(v): 
                    infodict[k].append(v)
    try: 
        for k, v in infodict.items():
            nvs = np.unique(v)
            assert len(nvs)==1, "%s: more than 1 value found: (%s, %s)" \
                                    % (datakey, k, str(nvs))
            infodict[k] = np.unique(v)[0]
    except AssertionError:
        return -1

    return infodict


def load_RID(session_dir, roi_id):

    roi_dir = os.path.join(session_dir, 'ROIs')
    roidict_path = glob.glob(os.path.join(roi_dir, 'rids_*.json'))[0]
    with open(roidict_path, 'r') as f:
        roidict = json.load(f)
    RID = roidict[roi_id]

    return RID

def load_AID(run_dir, traceid):
    run = os.path.split(run_dir)[-1]
    trace_dir = os.path.join(run_dir, 'retino_analysis')
    tracedict_path = os.path.join(trace_dir, 'analysisids_%s.json' % run)
    with open(tracedict_path, 'r') as f:
        tracedict = json.load(f)

    if 'traces' in traceid:
        fovdir = os.path.split(run_dir)[0]
        tmp_tdictpath = glob.glob(os.path.join(fovdir, '*run*', 'traces', 'traceids*.json'))[0]
        with open(tmp_tdictpath, 'r') as f:
            tmptids = json.load(f)
        roi_id = tmptids[traceid]['PARAMS']['roi_id']
        analysis_id = [t for t, v in tracedict.items() if v['PARAMS']['roi_type']=='manual2D_circle' and v['PARAMS']['roi_id'] == roi_id][0]
        print("Corresponding ANALYSIS ID (for %s with %s) is: %s" % (traceid, roi_id, analysis_id))

    else:
        analysis_id = traceid 
    TID = tracedict[analysis_id]
    pp.pprint(TID)
    return TID

def load_TID(run_dir, trace_id, auto=False):
    run = os.path.split(run_dir)[-1]
    trace_dir = os.path.join(run_dir, 'traces')
    tmp_tid_dir = os.path.join(trace_dir, 'tmp_tids')
    tracedict_path = os.path.join(trace_dir, 'traceids_%s.json' % run)
    try:
        print("Loading params for TRACE SET, id %s" % trace_id)
        with open(tracedict_path, 'r') as f:
            tracedict = json.load(f)
        TID = tracedict[trace_id]
        pp.pprint(TID)
    except Exception as e:
        print("No TRACE SET entry exists for specified id: %s" % trace_id)
        print("TRACE DIR:", tracedict_path)
        try:
            print("Checking tmp trace-id dir...")
            if auto is False:
                while True:
                    tmpfns = [t for t in os.listdir(tmp_tid_dir) \
                                    if t.endswith('json')]
                    for tidx, tidfn in enumerate(tmpfns):
                        print(tidx, tidfn)
                    userchoice = input("Select IDX of found tmp trace-id to view: ")
                    with open(os.path.join(tmp_tid_dir, tmpfns[int(userchoice)]), 'r') as f:
                        tmpTID = json.load(f)
                    print("Showing tid: %s, %s" % (tmpTID['trace_id'], tmpTID['trace_hash']))
                    pp.pprint(tmpTID)
                    userconfirm = input('Press <Y> to use this trace ID, or <q> to abort: ')
                    if userconfirm == 'Y':
                        TID = tmpTID
                        break
                    elif userconfirm == 'q':
                        break
        except Exception as e:
            traceback.print_exc()
            print("--------------------------------------------------------------")
            print("No tmp trace-ids found either... ABORTING with error:")
            print(e)
            print("--------------------------------------------------------------")

    return TID

 # ---------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------
def check_counts_per_condition(raw_traces, labels):
    # Check trial counts / condn:
    #print("Checking counts / condition...")
    min_n = labels.groupby(['config'])['trial'].unique().apply(len).min()
    conds_to_downsample = np.where( labels.groupby(['config'])['trial'].unique().apply(len) != min_n)[0]
    if len(conds_to_downsample) > 0:
        print("... adjusting for equal reps / condn...")
        d_cfgs = [sorted(labels.groupby(['config']).groups.keys())[i]\
                  for i in conds_to_downsample]
        trials_kept = []
        for cfg in labels['config'].unique():
            c_trialnames = labels[labels['config']==cfg]['trial'].unique()
            if cfg in d_cfgs:
                #ntrials_remove = len(c_trialnames) - min_n
                #print("... removing %i trials" % ntrials_remove)
    
                # In-place shuffle
                random.shuffle(c_trialnames)
    
                # Take the first 2 elements of the now randomized array
                trials_kept.extend(c_trialnames[0:min_n])
            else:
                trials_kept.extend(c_trialnames)
    
        ixs_kept = labels[labels['trial'].isin(trials_kept)].index.tolist()
        
        tmp_traces = raw_traces.loc[ixs_kept].reset_index(drop=True)
        tmp_labels = labels[labels['trial'].isin(trials_kept)].reset_index(drop=True)
        return tmp_traces, tmp_labels

    else:
        return raw_traces, labels
   
def reformat_morph_values(sdf, verbose=False):
    '''
    Rounds values for stimulus parameters
    Checks to make sure true aspect ratio is used.
    '''
    sdf = sdf.sort_index()
    aspect_ratio=1.75
    control_ixs = sdf[sdf['morphlevel']==-1].index.tolist()
    if len(control_ixs)==0: # Old dataset
        if 17.5 in sdf['size'].values:
            sizevals = sdf['size'].divide(aspect_ratio).astype(float).round(0)
            #np.array([round(s/aspect_ratio,0) for s in sdf['size'].values])
            sdf['size'] = sizevals
    else:  
        sizevals = np.array([round(s, 1) for s in sdf['size'].unique() \
                            if s not in ['None', None] and not np.isnan(s)])
        sdf.loc[control_ixs, 'size'] = pd.Series(sizevals, index=control_ixs).astype(float)
        sdf['size'] = sdf['size'].astype(float).round(decimals=1)
        #[round(s, 1) for s in sdf['size'].values]

    xpos = [x for x in sdf['xpos'].unique() if x is not None]
    ypos =  [x for x in sdf['ypos'].unique() if x is not None]
    #assert len(xpos)==1 and len(ypos)==1, "More than 1 pos? x: %s, y: %s" % (str(xpos), str(ypos))
    if verbose and (len(xpos)>1 or len(ypos)>1):
        print("warning: More than 1 pos? x: %s, y: %s" % (str(xpos), str(ypos)))
    sdf.loc[control_ixs, 'xpos'] = xpos[0]
    sdf.loc[control_ixs, 'ypos'] = ypos[0]

    return sdf


def process_and_save_traces(trace_type='dff', add_offset=True, save=True,
                            animalid=None, session=None, fov=None, 
                            experiment=None, traceid='traces001',
                            soma_fpath=None,
                            rootdir='/n/coxfs01/2p-data'):
    '''Process raw traces (SOMA ONLY), and calculate dff'''

    if save:
        print("... processing + saving data arrays (%s)." % trace_type)
    else:
        print("... processing data arrays, no save (%s)." % trace_type)

    assert (animalid is None and soma_fpath is not None) or (soma_fpath is None and animalid is not None), "Must specify either dataset params (animalid, session, etc.) OR soma_fpath to data arrays."

    if soma_fpath is None:
        # Load default data_array path
        search_str = '' if 'combined' in experiment else '_'
        soma_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov,
                                '*%s%s*' % (experiment, search_str), 
                                'traces', '%s*' % traceid, 
                                'data_arrays', 'np_subtracted.npz'))[0]
    dset = np.load(soma_fpath, allow_pickle=True, encoding='latin1')
    
    # Stimulus / condition info
    labels = pd.DataFrame(data=dset['labels_data'], 
                          columns=dset['labels_columns'])
    try: 
        labels = hutils.convert_columns_byte_to_str(labels)
    except (UnicodeDecodeError, AttributeError):
        pass

    #
    sdf = pd.DataFrame(dset['sconfigs'][()]).T
    if 'blobs' in soma_fpath: #self.experiment_type:
        sdf = reformat_morph_values(sdf)
    run_info = dset['run_info'][()]

    xdata_df = pd.DataFrame(dset['data'][:]) # neuropil-subtracted & detrended
    F0 = pd.DataFrame(dset['f0'][:]).mean().mean() # detrended offset
    #add_offsets = list(np.nanmean(dset['f0'][:], axis=0))
 
    if add_offset:
        #% Add baseline offset back into raw traces:
#        neuropil_fpath = soma_fpath.replace('np_subtracted', 'neuropil')
#        npdata = np.load(neuropil_fpath, allow_pickle=True)
#        neuropil_f0 = np.nanmean(np.nanmean(pd.DataFrame(npdata['f0'][:])))
#        neuropil_df = pd.DataFrame(npdata['data'][:]) 
#        add_offsets = list(np.nanmean(neuropil_df, axis=0)) 
#        print("    adding NP offset (NP f0 offset: %.2f)" % neuropil_f0)

        # # Also add raw 
        raw_fpath = soma_fpath.replace('np_subtracted', 'raw')
        rawdata = np.load(raw_fpath, allow_pickle=True)
        raw_f0 = np.nanmean(np.nanmean(pd.DataFrame(rawdata['f0'][:])))
        raw_df = pd.DataFrame(rawdata['data'][:])
        add_offsets = list(np.nanmean(raw_df, axis=0))
        print("    adding raw offset (raw f0 offset: %.2f)" % raw_f0)

        raw_traces = xdata_df + add_offsets + F0 # + raw_f0 
        #+ neuropil_f0 + raw_f0 # list(np.nanmean(raw_df, axis=0)) #.T + F0

        #raw_traces = xdata_df + F0 #add_offsets
        print("    adding raw offset (raw f0 offset: %.2f)" % F0)
   
    else:
        raw_traces = xdata_df.copy()

    if save:
        # SAVE
        data_dir = os.path.split(soma_fpath)[0]
        data_fpath = os.path.join(data_dir, 'corrected.npz')
        print("... Saving corrected data (%s)" %  os.path.split(data_fpath)[-1])
        np.savez(data_fpath, data=raw_traces.values)
  
    # Process dff/df/etc.
    stim_on_frame = labels['stim_on_frame'].unique()[0]
    tmp_df = []
    tmp_dff = []
    for k, g in labels.groupby(['trial']):
        tmat = raw_traces.loc[g.index]
        bas_mean = np.nanmean(tmat[0:stim_on_frame], axis=0)
        #if trace_type == 'dff':
        tmat_dff = (tmat - bas_mean) / bas_mean
        tmp_dff.append(tmat_dff)
        #elif trace_type == 'df':
        tmat_df = (tmat - bas_mean)
        tmp_df.append(tmat_df)
    dff_traces = pd.concat(tmp_dff, axis=0) 
    df_traces = pd.concat(tmp_df, axis=0) 

    if save:
        data_fpath = os.path.join(data_dir, 'dff.npz')
        print("... Saving dff data (%s)" %  os.path.split(data_fpath)[-1])
        np.savez(data_fpath, data=dff_traces.values)

        data_fpath = os.path.join(data_dir, 'df.npz')
        print("... Saving df data (%s)" %  os.path.split(data_fpath)[-1])
        np.savez(data_fpath, data=df_traces.values)

    if trace_type=='dff':
        return dff_traces, labels, sdf, run_info
    elif trace_type == 'df':
        return df_traces, labels, sdf, run_info
    else:
        return raw_traces, labels, sdf, run_info


def load_dataset(soma_fpath, trace_type='dff', is_neuropil=False,
                add_offset=True, make_equal=False, create_new=False, save=True):
    '''
    Loads all the roi traces and labels.
    If want to load corrected NP traces, set flag is_neuropil.
    To load raw NP traces, set trace_type='neuropil' and is_neuropil=False.

    Returns:  traces, labels, sdf, run_info

    '''
    traces=None
    labels=None
    sdf=None
    run_info=None
    try:
        data_fpath = soma_fpath.replace('np_subtracted', trace_type)
        if not os.path.exists(data_fpath) or create_new is True:
            # Process data and save
            traces, labels, sdf, run_info = process_and_save_traces(
                                                    trace_type=trace_type,
                                                    soma_fpath=soma_fpath,
                                                    add_offset=add_offset, save=save
            )
        else:
            if is_neuropil:
                np_fpath = data_fpath.replace(trace_type, 'neuropil')
                traces = load_corrected_neuropil_traces(np_fpath)
            else:
                #print("... loading saved data array (%s)." % trace_type)
                traces_dset = np.load(data_fpath, allow_pickle=True)
                traces = pd.DataFrame(traces_dset['data'][:]) 

            # Stimulus / condition info
            labels_fpath = data_fpath.replace(trace_type, 'labels')
            labels_dset = np.load(labels_fpath, allow_pickle=True, 
                                  encoding='latin1') 
            labels = pd.DataFrame(data=labels_dset['labels_data'], 
                                  columns=labels_dset['labels_columns'])
            try: 
                labels = hutils.convert_columns_byte_to_str(labels)
            except (UnicodeDecodeError, AttributeError):
                pass
            sdf = pd.DataFrame(labels_dset['sconfigs'][()]).T.sort_index()
            if 'blobs' in data_fpath: 
                sdf = reformat_morph_values(sdf)
            # Format condition info:
            if 'image' in sdf['stimtype']:
                aspect_ratio = sdf['aspect'].unique()[0]
                sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]
            # Get run info 
            run_info = labels_dset['run_info'][()]
        if make_equal:
            print("... making equal")
            traces, labels = check_counts_per_condition(traces, labels)      
    except Exception as e:
        traceback.print_exc()
        print("ERROR LOADING DATA")

    return traces, labels, sdf, run_info


def load_corrected_neuropil_traces(neuropil_fpath):
    npdata = np.load(neuropil_fpath, allow_pickle=True)
    #print(npdata.keys())
    neuropil_f0 = np.nanmean(np.nanmean(pd.DataFrame(npdata['f0'][:])))
    neuropil_df = pd.DataFrame(npdata['data'][:]).copy()

    add_np_offsets = list(np.nanmean(neuropil_df, axis=0))
    xdata_np = neuropil_df + add_np_offsets + neuropil_f0
    return xdata_np
 
# 
# --------------------------------------------------------------------
# Data processing
# --------------------------------------------------------------------
def get_rolling_baseline(Xdf, window_size, quantile=0.08):
        
    #window_size_sec = (nframes_trial/framerate) * 2 # decay_constant * 40
    #decay_frames = window_size_sec * framerate # decay_constant in frames
    #window_size = int(round(decay_frames))
    #quantile = 0.08
    window_size = int(round(window_size))
    Fsmooth = Xdf.apply(rolling_quantile, args=(window_size, quantile))
    offset = Fsmooth.mean().mean()
    print("drift offset:", offset)
    Xdata = (Xdf - Fsmooth) #+ offset
    #Xdata = np.array(Xdata_tmp)
    
    return Xdata, Fsmooth


# Use cnvlib.smoothing functions to deal get mirrored edges on rolling quantile:
def rolling_quantile(x, width, quantile):
    """Rolling quantile (0--1) with mirrored edges."""
    x, wing, signal = check_inputs(x, width)
    rolled = signal.rolling(2 * wing + 1, 2, center=True).quantile(quantile)
    return np.asfarray(rolled[wing:-wing])


def check_inputs(x, width, as_series=True):
    """Transform width into a half-window size.

    `width` is either a fraction of the length of `x` or an integer size of the
    whole window. The output half-window size is truncated to the length of `x`
    if needed.
    """
    x = np.asfarray(x)
    wing = _width2wing(width, x)
    signal = _pad_array(x, wing)
    if as_series:
        signal = pd.Series(signal)
    return x, wing, signal


def _width2wing(width, x, min_wing=3):
    """Convert a fractional or absolute width to integer half-width ("wing").
    """
    if 0 < width < 1:
        wing = int(math.ceil(len(x) * width * 0.5))
    elif width >= 2 and int(width) == width:
        wing = int(width // 2)
    else:
        raise ValueError("width must be either a fraction between 0 and 1 "
                         "or an integer greater than 1 (got %s)" % width)
    wing = max(wing, min_wing)
    wing = min(wing, len(x) - 1)
    assert wing >= 1, "Wing must be at least 1 (got %s)" % wing
    return wing


def _pad_array(x, wing):
    """Pad the edges of the input array with mirror copies."""
    return np.concatenate((x[wing-1::-1],
                           x,
                           x[:-wing-1:-1]))


# Visualization
def smooth_timecourse(in_trace, win_size=41):
    #smooth trace
    win_half = int(round(win_size/2))
    trace_pad = np.pad(in_trace, ((win_half, win_half)), 'reflect') # 'symmetric') #'edge')

    smooth_trace = np.convolve(trace_pad, np.ones((win_size,))*(1/float(win_size)),'valid')
    
    return smooth_trace

def smooth_traces_trial(gg, win_size=5, colname='trial'):
    smoothed_ = smooth_timecourse(gg, win_size=win_size)
    return pd.Series(smoothed_)


# --------------------------------------------------------------------
# Neuropil Calculations
# --------------------------------------------------------------------
def append_neuropil_subtraction(maskdict_path, cfactor, 
                        filetraces_dir, datakey, create_new=False, rootdir=''):
    '''
    Loads RAW traces hdf5 from <filetraces_dir>, adds:
    - neuropil : neuropil masks applied to movie tifs
    - np_subtracted: raw - (cfactor)*neuropil
    
    '''
    MASKS = h5py.File(maskdict_path, 'r')
    filetraces_fpaths = sorted([os.path.join(filetraces_dir, t) for t in os.listdir(filetraces_dir) if t.endswith('hdf5')], \
                                key=hutils.natural_keys)
    #
    print("Appending subtrated NP traces to %i files." % len(filetraces_fpaths))
    for tfpath in filetraces_fpaths:
        traces_currfile = h5py.File(tfpath, 'r+')
        fidx = int(os.path.split(tfpath)[-1].split('_')[0][4:]) - 1
        curr_file = "File%03d" % int(fidx+1)
        print("CFACTOR -- Appending neurpil corrected traces: %s" % curr_file)
        try:
            for curr_slice in traces_currfile.keys():
                # Load raw traces
                tracemat = np.array(traces_currfile[curr_slice]['traces']['raw'])
                # First check that neuropil traces don't already exist:
                if 'neuropil' in traces_currfile[curr_slice]['traces'].keys() and create_new is False:
                    np_tracemat = np.array(traces_currfile[curr_slice]['traces']['neuropil'])
                    overwrite_neuropil = False
                else:
                    overwrite_neuropil = True
                if 'np_subtracted' in traces_currfile[curr_slice]['traces'].keys() and create_new is False:
                    np_correctedmat = np.array(traces_currfile[curr_slice]['traces']['np_subtracted'])
                    if np.mean(np_correctedmat) == 0:
                        overwrite_correctedmat = True
                    else:
                        overwrite_correctedmat = False
                        
                if overwrite_neuropil is True:
                    overwrite_correctedmat = True # always overwrite tracemat if new neuropil
                    # Load tiff:
                    tiffpath = traces_currfile.attrs['source_file']
                    print("Calculating neuropil from src: %s" % tiffpath)
                    if rootdir not in tiffpath:
                        session, animalid, fovn = hutils.split_datakey_str(datakey)
                        tiffpath = hutils.replace_root(tiffpath, rootdir, animalid, session)
                    tiff = tf.imread(tiffpath)
                    T, d1, d2 = tiff.shape
                    d = d1*d2
                    orig_mat_shape = traces_currfile[curr_slice]['traces']['raw'].shape
                    nchannels = T/orig_mat_shape[0]
                    signal_channel_idx = int(traces_currfile.attrs['signal_channel']) - 1

                    tiffR = np.reshape(tiff, (T, d), order='C'); del tiff
                    tiffslice = tiffR[signal_channel_idx::nchannels,:]
                    print("SLICE shape is:", tiffslice.shape)

                    np_maskarray = MASKS[curr_file][curr_slice]['np_maskarray'][:]
                    np_tracemat = tiffslice.dot(np_maskarray)

                if overwrite_correctedmat is True:
                    np_correctedmat = tracemat - (cfactor * np_tracemat)

                if 'neuropil' not in traces_currfile[curr_slice]['traces'].keys():
                    np_traces = traces_currfile.create_dataset('/'.join([curr_slice, 'traces', 'neuropil']), np_tracemat.shape, np_tracemat.dtype)
                else:
                    np_traces = traces_currfile[curr_slice]['traces']['neuropil']
                np_traces[...] = np_tracemat
                if 'np_subtracted' not in traces_currfile[curr_slice]['traces'].keys():
                    np_corrected = traces_currfile.create_dataset('/'.join([curr_slice, 'traces', 'np_subtracted']), np_correctedmat.shape, np_correctedmat.dtype)
                else:
                    np_corrected = traces_currfile[curr_slice]['traces']['np_subtracted']
                np_corrected[...] = np_correctedmat
                np_corrected.attrs['correction_factor'] = cfactor
        except Exception as e:
            print("** ERROR appending NP-subtracted traces: %s" % traces_currfile)
            print(traceback.print_exc())
        finally:
            traces_currfile.close()

    return filetraces_dir



# --------------------------------------------------------------------
# Data grouping and calculations
# --------------------------------------------------------------------

def get_mean_and_std_traces(roi, traces, labels, curr_cfgs, stimdf, return_stacked=False,
                            smooth=False, win_size=5):
    import scipy.stats as spstats

    cfg_groups = labels[labels['config'].isin(curr_cfgs)].groupby(['config'])
    tested_thetas = sorted(stimdf['ori'].unique())

    mean_traces = np.array([np.nanmean(np.array([traces[roi][trials.index]\
                for rep, trials in cfg_df.groupby(['trial'])]), axis=0) \
                for cfg, cfg_df in \
                sorted(cfg_groups, key=lambda x: stimdf['ori'][x[0]])])

    std_traces = np.array([spstats.sem(np.array([traces[roi][trials.index]\
                for rep, trials \
                in cfg_df.groupby(['trial'])]), axis=0, nan_policy='omit') \
                for cfg, cfg_df \
                in sorted(cfg_groups, key=lambda x: stimdf['ori'][x[0]])])

    tpoints = np.array([np.array([trials['tsec'] for rep, trials \
                in cfg_df.groupby(['trial'])]).mean(axis=0) \
                for cfg, cfg_df \
                in sorted(cfg_groups, key=lambda x: stimdf['ori'][x[0]])]).mean(axis=0).astype(float)

    if smooth:
        meandf = pd.DataFrame(mean_traces.T, columns=tested_thetas)
        smoothdf = meandf.apply(smooth_timecourse, win_size=win_size)
        mean_traces = smoothdf.T.values

        semdf = pd.DataFrame(std_traces.T, columns=tested_thetas)
        smoothdf2 = semdf.apply(smooth_timecourse, win_size=win_size)
        std_traces = smoothdf2.T.values
       

    if return_stacked:
        tested_thetas = sorted(stimdf['ori'].unique())
        trace_df = pd.DataFrame(mean_traces.T, columns=tested_thetas)
        trace_df['time'] = tpoints
        tdf_mean = trace_df.melt(id_vars=['time'], var_name='ori', value_name='mean')

        err_df = pd.DataFrame(std_traces.T, columns=tested_thetas)
        err_df['time'] = tpoints
        tdf_sem = err_df.melt(id_vars=['time'], var_name='ori', value_name='sem')
        tdf = pd.merge(tdf_mean, tdf_sem, on=['time', 'ori'])

        return tdf
    else:
        return mean_traces, std_traces, tpoints


def group_roidata_stimresponse(roidata, labels, roi_list=None, 
                            return_grouped=True, nframes_post=0): #None):
    '''
    roidata: array of shape nframes_total x nrois
    labels:  dataframe of corresponding nframes_total with trial/config info
    
    Returns:
        grouped dataframe, where each group is a cell's dataframe of shape ntrials x (various trial metrics and trial/config info)
    '''   
    if isinstance(roidata, pd.DataFrame):
        roidata = roidata.values
    if roi_list is not None:
        roi_list = np.array(roi_list)
        roidata0 = roidata.copy()
        roidata = roidata0[:, roi_list]
        print("... selecting %i of %i rois" \
                    % (roidata0.shape[1], roidata.shape[1]))    
    try:
        stimdur_vary = False
        assert len(labels['nframes_on'].unique())==1, \
            "More than 1 idx found for nframes on... %s" \
                % str(list(set(labels['nframes_on'])))
        assert len(labels['stim_on_frame'].unique())==1, \
            "More than 1 idx found for first frame on... %s" \
                % str(list(set(labels['stim_on_frame'])))
        nframes_on = int(round(labels['nframes_on'].unique()[0]))
        stim_on_frame =  int(round(labels['stim_on_frame'].unique()[0]))
    except Exception as e:
        stimdur_vary = True
        
    groupby_list = ['config', 'trial']
    config_groups = labels.groupby(groupby_list)
    
    if roi_list is None:
        roi_list = np.arange(0, roidata.shape[-1])

    df_list = []
    for (config, trial), trial_ixs in config_groups:
        if stimdur_vary:
            # Get stim duration info for this config:
            assert len(labels[labels['config']==config]['nframes_on'].unique())==1, \
                "Something went wrong! More than 1 unique stim dur for config: %s" % config
            assert len(labels[labels['config']==config]['stim_on_frame'].unique())==1, \
                "Something went wrong! More than 1 unique stim ON frame for config: %s" % config
            nframes_on = labels[labels['config']==config]['nframes_on'].unique()[0]
            stim_on_frame = labels[labels['config']==config]['stim_on_frame'].unique()[0]
             
        trial_frames = roidata[trial_ixs.index.tolist(), :]
        
        nframes_stim = int(round(nframes_on + nframes_post)) #int(round(nframes_on*1.5))

        nrois = trial_frames.shape[-1]
        base_mean = np.nanmean(trial_frames[0:stim_on_frame, :], axis=0)
        base_std = np.nanstd(trial_frames[0:stim_on_frame, :], axis=0)
        stim_mean = np.nanmean(trial_frames[stim_on_frame:stim_on_frame+nframes_stim, :], axis=0)
        
        df_trace = (trial_frames - base_mean) / base_mean
        bas_mean_df = np.nanmean(df_trace[0:stim_on_frame, :], axis=0)
        bas_std_df = np.nanstd(df_trace[0:stim_on_frame, :], axis=0)
        stim_mean_df = np.nanmean(df_trace[stim_on_frame:stim_on_frame+nframes_stim, :], axis=0)
        
        zscore = (stim_mean - base_mean) / base_std
        #zscore = (stim_mean) / base_std
        dff = (stim_mean - base_mean) / base_mean
        dF = stim_mean - base_mean
        snr = stim_mean / base_mean
        df_list.append(pd.DataFrame(
                    {'config': np.tile(config, (nrois,)),
                     'trial': np.tile(trial, (nrois,)), 
                     'stim_mean': stim_mean, # called meanstim ...
                     'zscore': zscore,
                     'dff': dff,
                     'df': dF, 
                     'snr': snr,
                     'base_std': base_std,
                     'base_mean': base_mean,
                     
                     'stim_mean_df': stim_mean_df,
                     'base_mean_df': bas_mean_df,
                     'base_std_df': bas_std_df}, index=roi_list)
        )

    df = pd.concat(df_list, axis=0) # size:  ntrials * 2 * nrois\
    df['cell'] = df.index.tolist()
    if return_grouped:    
        df_by_rois = df.groupby(df.index)
        return df_by_rois
    else:
        return df



