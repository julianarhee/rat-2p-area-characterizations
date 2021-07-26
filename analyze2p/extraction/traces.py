

def get_mean_and_std_traces(roi, raw_traces, labels, curr_cfgs, stimdf):
    cfg_groups = labels[labels['config'].isin(curr_cfgs)].groupby(['config'])

    mean_traces = np.array([np.nanmean(np.array([traces[roi][trials.index]\
                for rep, trials in cfg_df.groupby(['trial'])]), axis=0) \
                for cfg, config_df in \
                sorted(cfg_groups, key=lambda x: stimdf['ori'][x[0]])])

    std_traces = np.array([stats.sem(np.array([traces[roi][trials.index]\
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



