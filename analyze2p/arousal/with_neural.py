#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  Apr 24 19:56:32 2020

@author: julianarhee
"""
import numpy as np
import pandas as pd

import analyze2p.aggregate_datasets as aggr
import analyze2p.arousal.dlc_utils as dlcutils

def balance_pupil_split(pupildf, feature_name='pupil_fraction', n_cuts=3,
                        match_cond=True, match_cond_name='size', equalize_after_split=True, 
                        equalize_by='config', common_labels=None, verbose=False, shuffle_labels=False):
    '''
    Split pupildf (trials) by "low" and "high" states with balanced samples.

    ARGS

    match_cond (bool): 
        Split distribution within condition (e.g., match for each SIZE separately).

    match_cond_name (str):
        The condition to split by, if match_cond==True 
        *(Should be a column in pupildf)

    equalize_after_split (bool):
        Equalize across stimulus configs AFTER doing pupil split.

    equalize_by (str):
        Condition to match across, e.g., config label,  morphlevel, etc. 
        *(Should be column in pupildf)
    '''

    if match_cond:
        assert match_cond_name in pupildf.columns, \
                "Requested split within <%s>, but not found." % match_cond_name
        low_=[]; high_=[];
        for sz, sd in pupildf.groupby([match_cond_name]):
            p_low, p_high = dlcutils.split_pupil_range(sd, \
                                        feature_name=feature_name, n_cuts=n_cuts)
            # if shuffle, shuffle labels, then equalize conds
            if shuffle_labels:
                n_low = p_low.shape[0]
                n_high = p_high.shape[0]
                p_all = pd.concat([p_low, p_high], axis=0)
                p_all_shuffled = p_all.sample(frac=1).reset_index(drop=True)
                p_low = p_all_shuffled.sample(n=n_low)
                unused_ixs = [i for i in p_all_shuffled['trial'].values \
                                if i not in p_low['trial'].values]
                p_high = p_all_shuffled[p_all_shuffled['trial'].isin(unused_ixs)] 
            if equalize_after_split:
                p_low_eq, p_high_eq = balance_samples_by_config(p_low, p_high, 
                                            config_label=equalize_by, 
                                            common_labels=common_labels)
                if verbose:
                    print("... sz %i, pre/post balance. Low: %i/%i | High: %i/%i" \
                      % (sz, p_low.shape[0], p_low_eq.shape[0], \
                            p_high.shape[0], p_high_eq.shape[0]))
                low_.append(p_low_eq)
                high_.append(p_high_eq)
            else:
                low_.append(p_low)
                high_.append(p_high)
        high = pd.concat(high_)
        low = pd.concat(low_)
    else:
        low, high = dlcutils.split_pupil_range(pupildf, \
                                feature_name=pupil_feature, n_cuts=n_cuts)

    return low, high


def balance_samples_by_config(df1, df2, config_label='config', common_labels=None):
    '''
    Given 2 dataframes with config labels on each trial, match reps per config.

    ARGS
    
    config_label: (str)
        If 'config', matches across all configs found (config001, config002, etc.)
        For ex., can also use "morphlevel" to just match counts of 
        morphlevel (ignoring unique size, etc.).

    '''
    assert config_label in df1.columns, "Label <%s> not in df1 columns" % config_label
    assert config_label in df2.columns, "Label <%s> not in df2 columns" % config_label

    if common_labels is None:
        common_labels = np.intersect1d(df1[config_label].unique(), \
                                       df2[config_label].unique())
        config_subset = False
    else:
        config_subset = True

    # Get equal counts for specified labels 
    df1_eq = aggr.equal_counts_df(df1[df1[config_label].isin(common_labels)], \
                                    equalize_by=config_label)
    df2_eq = aggr.equal_counts_df(df2[df2[config_label].isin(common_labels)], \
                                    equalize_by=config_label) 
    # Check that each partition has the same # of labels 
    # (i.e., if missing 1 label, will ahve diff #s)
    #df1_labels = df1_eq[df1_eq[config_label].isin(common_labels)][config_label].unique()
    #df2_labels = df2_eq[df2_eq[config_label].isin(common_labels)][config_label].unique()

 
    # Get min N reps per condition (based on N reps per train condition)
    min_reps_per = min([df1_eq[config_label].value_counts()[0], \
                        df2_eq[config_label].value_counts()[0]])
    
    # Draw min_reps_per samples without replacement
    if config_subset:
        # we set sample #s by train configs, and include the other trials for test
        df1_resampled = pd.concat([g.sample(n=min_reps_per, replace=False) \
                                if g.shape[0]>min_reps_per \
                                    else g for c, g in df1.groupby([config_label])])
        df2_resampled = pd.concat([g.sample(n=min_reps_per, replace=False) \
                                if g.shape[0]>min_reps_per \
                                    else g for c, g in df2.groupby([config_label])])

    else:
        df2_resampled = pd.concat([g.sample(n=min_reps_per, replace=False) \
                                    for c, g in df2_eq.groupby([config_label])])
        df1_resampled = pd.concat([g.sample(n=min_reps_per, replace=False) \
                                    for c, g in df1_eq.groupby([config_label])])

    return df1_resampled, df2_resampled



def split_by_arousal(nmetrics, pmetrics, n_cuts=3, 
                     feature_name='pupil_fraction', match_cond_name='config'):
    '''
    Split trialmetrics for neuraldata (nmetrics) and pupil data (pmetrics)
    by "low" and "high" arousal states. Does zscore for neural before split.
     
    
    ARGS
    n_cuts (int)
        3=Split into Low, Med, High
        4=Split quartiles
    '''
    ndf, pdf = match_trials(nmetrics, pmetrics)
    pdf = add_stimulus_info(ndf, pdf)
    p_low, p_high = balance_pupil_split(pdf, feature_name=feature_name,
                                        match_cond_name=match_cond_name, 
                                        n_cuts=n_cuts)
    p_low['arousal'] = 'low'
    p_high['arousal'] = 'high'
    splitpupil = pd.concat([p_low, p_high], axis=0)

    # Now, Zscore responses, and then split neural
    ndf = ndf.reset_index(drop=True)
    zscored_ = aggr.get_zscored_from_ndf(ndf)
    ndf_z = aggr.unstacked_neuraldf_to_stacked(zscored_)
    add_cols= [k for k in ndf.columns if k not in ndf_z.columns]
    ndf_z[add_cols] = ndf[add_cols].copy()      
    # Add arousal info
    n_list=[]
    for cond, split_p in splitpupil.groupby('arousal'):
        split_n = ndf_z[ndf_z.trial.isin(split_p['trial'].values)].copy()
        split_n['arousal'] = cond
        n_list.append(split_n)
    splitneural = pd.concat(n_list)

    return splitneural, splitpupil



# --------------------------------------------------------------------
# Stimulus info
# --------------------------------------------------------------------
def add_stimulus_info(ndf, pdf):
    '''
    Args: 
    
    ndf (pd.DataFrame) 
        Neuraldata (trial metrics) for 1 FOV. 
        Should have 'trials' and 'config' as cols.

    pdf (pd.DataFrame)
        Pupil metrics for all trials. Also should have 'trials' as col.

    '''
    # Add stimulus info to pupil df
    trial_lut = ndf[['trial', 'config']].drop_duplicates().sort_values(by='trial')
    trial_lut.index = trial_lut['trial'].values
    trial_lut = trial_lut['config'].copy()
    pdf['config'] = trial_lut.loc[pdf['trial'].values].values

    return pdf

def match_trials(ndf, pdf):
    trials_ = np.intersect1d(ndf['trial'].unique(), pdf.dropna()['trial'].unique())
    ndf_ = ndf[ndf['trial'].isin(trials_)].copy()
    pdf_ = pdf[pdf['trial'].isin(trials_)].copy()

    return ndf_, pdf_
 

