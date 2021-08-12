
import os
import glob
import traceback

import numpy as np
import pandas as pd

import scipy.stats as spstats
import analyze2p.utils as hutils
import _pickle as pkl

# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------

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
                                key=natural_keys) 
        assert len(extraction_files) > 0, \
            "(%s, %s) No extraction info found..." % (datakey, curr_exp)
    except AssertionError:
        return None
    
    for i, ifile in enumerate(extraction_files):
        with open(ifile, 'r') as f:
            info = json.load(f)
        if i==0:
            infodict = dict((k, [v]) for k, v in info.items() if isnumber(v)) 
        else:
            for k, v in info.items():
                if isnumber(v): 
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

 
# 
# --------------------------------------------------------------------
# Data processing
# --------------------------------------------------------------------
def load_corrected_neuropil_traces(neuropil_fpath):
    npdata = np.load(neuropil_fpath, allow_pickle=True)
    #print(npdata.keys())
    neuropil_f0 = np.nanmean(np.nanmean(pd.DataFrame(npdata['f0'][:])))
    neuropil_df = pd.DataFrame(npdata['data'][:]).copy()

    add_np_offsets = list(np.nanmean(neuropil_df, axis=0))
    xdata_np = neuropil_df + add_np_offsets + neuropil_f0
    return xdata_np


# --------------------------------------------------------------------
# Data grouping and calculations
# --------------------------------------------------------------------

def get_mean_and_std_traces(roi, traces, labels, curr_cfgs, stimdf):
    import scipy.stats as spstats

    cfg_groups = labels[labels['config'].isin(curr_cfgs)].groupby(['config'])

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

    return mean_traces, std_traces, tpoints


def group_roidata_stimresponse(roidata, labels_df, roi_list=None, 
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



