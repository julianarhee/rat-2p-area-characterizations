#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  Apr 24 19:56:32 2020

@author: julianarhee
"""
import os
import sys
import json
import glob
import copy
import copy
import random
import optparse
import itertools
import datetime
import time
import math
import traceback

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import statsmodels as sm
import _pickle as pkl
import multiprocessing as mp
from functools import partial

from scipy import stats as spstats


from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold
from sklearn import preprocessing
import sklearn.svm as svm

from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmetrics

import analyze2p.aggregate_datasets as aggr
import analyze2p.utils as hutils
import analyze2p.plotting as pplot
import psignifit as ps


from inspect import currentframe, getframeinfo
from pandas.core.common import SettingWithCopyError
pd.options.mode.chained_assignment='warn' #'raise' # 'warn'


# ======================================================================
# RF/retino funcs
# ======================================================================
import analyze2p.receptive_fields.utils as rfutils
import analyze2p.objects.sim_utils as su

def get_cells_with_overlap(cells0, sdata, overlap_thr=0.5, greater_than=False,
                response_type='dff', do_spherical_correction=False):

    rfdf = get_rfdf(cells0, sdata, response_type=response_type,
                    do_spherical_correction=do_spherical_correction)
    cells_RF = get_cells_with_rfs(cells0, rfdf)

    fit_desc = rfutils.get_fit_desc(response_type=response_type,
                            do_spherical_correction=do_spherical_correction)
    rfpolys, _ = su.load_rfpolys(fit_desc)
 
    cells_pass = calculate_overlaps(cells_RF, rfpolys, 
                            overlap_thr=overlap_thr, greater_than=greater_than)

#    cells_pass['global_ix'] = [int(cells0[(cells0.visual_area==va)
#                             & (cells0.datakey==dk) 
#                             & (cells0['cell']==rid)]['global_ix']\
#                             .unique()) for va, dk, rid \
#                             in cells_pass[['visual_area', 'datakey', 'cell']]\
#                             .values]
#
    return cells_pass

def calculate_overlaps(rfdf, rfpolys, overlap_thr=0.5, greater_than=False,
                        experiment='blobs', 
                        overlap_metric='relative_overlap', 
                        resolution=[1920, 1080]):
    '''
    cycle thru all rf datasets, calculate overlaps.
    returns MIN overlap value for each cell.
    '''
    m_=[]
    for (va, dk), rfs_ in rfdf.groupby(['visual_area', 'datakey']):
        curr_rfs = rfs_.copy()
        curr_rfs.index = rfs_['cell'].values
        curr_rfs['poly'] = None
        rois_ = curr_rfs['cell'].unique()
        try:
            curr_polys = rfpolys[(rfpolys.datakey==dk) 
                               & (rfpolys['cell'].isin(rois_))]
            if len(curr_polys)>0:
                curr_polys.index=curr_polys['cell'].values
                curr_rfs.loc[curr_polys.index, 'poly'] = curr_polys['poly']

            overlaps_ = su.calculate_overlaps_fov(dk, curr_rfs, 
                                experiment=experiment, 
                               resolution=resolution)
            min_ = overlaps_.groupby('cell')[overlap_metric].min()
            if greater_than:
                pass_ = min_[min_>overlap_thr].index.to_numpy()
            else:
                pass_ = min_[min_>=overlap_thr].index.to_numpy()
            pass_rfs = curr_rfs[curr_rfs['cell'].isin(pass_)].copy()
            m_.append(pass_rfs)
        except Exception as e:
            print("ERROR w/ overlaps: %s, %s" % (va, dk))
            raise e 
        #min_['visual_area'] = va
        #min_['datakey'] = dk
    min_overlaps = pd.concat(m_, axis=0)
    
    return min_overlaps


def get_cells_with_matched_rfs(cells0, sdata, 
                rf_lim='percentile', rf_metric='fwhm_avg',
                response_type='dff', do_spherical_correction=False):
    '''
    Load RF fits for current cells. 
    Currently, only selects rf-5 for V1/LM, rf-10 for Li. 
    Return cells with RF sizes within specified range

    Args:
    rf_lim: (str or tuple/list/array)
        'maxmin': calculate max and min ranges
        'percentile':  use whisker extents (1.5 x IQR range above/below)
        tuple/array:  use specified (lower, upper) bounds 
    '''
 
    rfdf = get_rfdf(cells0, sdata, response_type=response_type,
                    do_spherical_correction=do_spherical_correction)
    cells_RF = get_cells_with_rfs(cells0, rfdf)
    cells_lim, limits = limit_cells_by_rf(cells_RF, rf_lim=rf_lim,
                                rf_metric=rf_metric)
    # Resample matched RFs to match 1-to-1
    cells_matched = match_each_cell_by_area(cells_lim, n_samples=None)

    return cells_matched

def find_closest(A, B, C):
    p=len(A)
    q=len(B)
    r=len(C)
    # Initialize min diff
    diff = sys.maxsize
    res_i=0; res_j=0; res_k=0;
    # Travesre Array
    i=0; j=0; k=0;
    while(i < p and j < q and k < r):
        # Find minimum and maximum of current three elements
        minimum = min(A[i], min(B[j], C[k]))
        maximum = max(A[i], max(B[j], C[k]));
        # Update result if current diff is less than the min diff so far
        if maximum-minimum < diff:
            res_i = i
            res_j = j
            res_k = k
            diff = maximum - minimum;
        # We can 't get less than 0 as values are absolute
        if diff == 0:
            break
        # Increment index of array with smallest value
        if A[i] == minimum:
            i = i+1
        elif B[j] == minimum:
            j = j+1
        else:
            k = k+1
    return res_i, res_j, res_k

def match_each_cell_by_area(cellsdf, n_samples=None):
    cellsdf = cellsdf.reset_index(drop=True)
    if n_samples is None:
        n_samples = cellsdf['visual_area'].value_counts().min() 
    A = cellsdf[cellsdf.visual_area=='V1'].copy()
    B = cellsdf[cellsdf.visual_area=='Lm'].copy()
    C = cellsdf[cellsdf.visual_area=='Li'].copy()
    p_list = []
    for i in range(n_samples):
        c1 = np.asarray(A['fwhm_avg'].values)
        c2 = np.asarray(B['fwhm_avg'].values)
        c3 = np.asarray(C['fwhm_avg'].values)
        pos = find_closest(c1, c2, c3)
        p_list.append(A.iloc[pos[0]])
        p_list.append(B.iloc[pos[1]])       
        p_list.append(C.iloc[pos[2]])
        A.drop(A.iloc[pos[0]].name, inplace=True)
        B.drop(B.iloc[pos[1]].name, inplace=True)
        C.drop(C.iloc[pos[2]].name, inplace=True)
    df_ = pd.concat(p_list, axis=1).T
    #df_['n_samples'] = n_samples
    return df_

def limit_cells_by_rf(cells_RF, rf_lim='percentile', rf_metric='fwhm_avg'):

    if rf_lim=='maxmin':
        rf_upper = cells_RF.groupby('visual_area')[rf_metric].max().min()
        rf_lower = cells_RF.groupby('visual_area')[rf_metric].min().max()
    elif rf_lim=='percentile':
        lims = pd.concat([pd.DataFrame(\
                    get_quartile_limits(cg, rf_metric=rf_metric, whis=1.5),
                    columns=[va], index=['lower', 'upper'])\
                    for va, cg in cells_RF.groupby('visual_area')], axis=1)
        rf_lower, rf_upper = lims.loc['lower'].max(), lims.loc['upper'].min()
    else:
        rf_lower, rf_upper= rf_lim  # (6.9, 16.6)

    cells_lim = cells_RF[(cells_RF[rf_metric]<=rf_upper) 
                    & (cells_RF[rf_metric]>=rf_lower)].copy()

    return cells_lim, lims

def get_quartile_limits(cells_RF, rf_metric='fwhm_avg', whis=1.5):
    q1 = cells_RF[rf_metric].quantile(0.25)
    q3 = cells_RF[rf_metric].quantile(0.75)
    iqr = q3 - q1
    limit_lower = q1 - whis*iqr
    limit_upper = q3 + whis*iqr
    return limit_lower, limit_upper


def get_rfdf(cells0, sdata, response_type='dff', do_spherical_correction=False):
    '''Combines all RF fit params, returns AVERAGE of >1 experiment'''
    # Get cells and metadata
    assigned_cells, rf_meta = aggr.select_assigned_cells(cells0, sdata, 
                                                    experiments=['rfs', 'rfs10']) 
    # Load RF fit data
    rf_fit_desc = rfutils.get_fit_desc(response_type=response_type, 
                                do_spherical_correction=do_spherical_correction)
    rfdata = rfutils.aggregate_rfdata(rf_meta, assigned_cells, 
                                fit_desc=rf_fit_desc,
                                reliable_only=False)
    # Combined rfs5/rfs10
    rfdf = rfutils.average_rfs(rfdata, keep_experiment=False) 

    return rfdf

def get_cells_with_rfs(cells0, rfdf):
    '''
    CELLS should be assigned + responsive cells (from NDATA)
    rfdf should only have 1 value per cell
    '''
    cells_RF = pd.concat([rfdf[(rfdf.visual_area==va) & (rfdf.datakey==dk) 
                     & (rfdf['cell'].isin(g['cell'].values))] \
                 for (va, dk), g in cells0.groupby(['visual_area', 'datakey'])])

    if 'global_ix' in cells0.columns:
        cells_RF['global_ix'] = [int(cells0[(cells0.visual_area==va)
                             & (cells0.datakey==dk) 
                             & (cells0['cell']==rid)]['global_ix']\
                             .unique()) for va, dk, rid \
                             in cells_RF[['visual_area', 'datakey', 'cell']]\
                             .values]


    return cells_RF



# ======================================================================
# Calculation functions 
# ======================================================================

def computeMI(x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi


def fit_svm(zdata, targets, test_split=0.2, cv_nfolds=5,  n_processes=1,
                C_value=None, verbose=False, return_clf=False, 
                return_predictions=False,
                randi=10, inum=0):
    # For each transformation, split trials into 80% and 20%
    train_data, test_data, train_labels, test_labels = train_test_split(
                                                        zdata, 
                                                        targets['label'].values, 
                                                        test_size=test_split, 
                                                        stratify=targets['label'], 
                                                        shuffle=True, 
                                                        random_state=randi)
    if verbose:
        print("Unique train: %s (%i)" \
                % (str(np.unique(train_labels)), len(train_labels)))
    # Cross validate (tune C w/ train data)
    cv = C_value is None
    if cv:
        cv_grid = tune_C(train_data, train_labels, scoring_metric='accuracy', 
                        cv_nfolds=cv_nfolds,  
                        test_split=test_split, verbose=verbose) 
        C_value = cv_grid.best_params_['C'] 
    else:
        assert C_value is not None, "Provide value for hyperparam C..."
    # Fit SVM
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=randi)
    scores = cross_validate(trained_svc, train_data, train_labels, cv=cv_nfolds,
                            scoring=('accuracy'),
                            #scoring=('precision_macro', 'recall_macro', 'accuracy'),
                            return_train_score=True)
    iterdict = dict((s, values.mean()) for s, values in scores.items())
    if verbose:
        print('... train (C=%.2f): %.2f, test: %.2f' \
                    % (C_value, iterdict['train_score'], iterdict['test_score']))
    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=randi)\
                     .fit(train_data, train_labels) 
    # DATA - Test with held-out data
    test_data = scaler.transform(test_data)
    test_score = trained_svc.score(test_data, test_labels)
    # DATA - Calculate MI
    predicted_labels = trained_svc.predict(test_data)
    predictions = pd.DataFrame({'true': test_labels, 'predicted': predicted_labels})

    if verbose:    
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print(skmetrics.classification_report(test_labels, predicted_labels))

    #mi_dict = get_mutual_info_metrics(test_labels, predicted_labels)
    #iterdict.update(mi_dict) 
    iterdict.update({'heldout_test_score': test_score, 'C': C_value, 'randi': randi})
    
    iterdf = pd.DataFrame(iterdict, index=[inum])

    if return_clf:
        if return_predictions:
            return iterdf, trained_svc, scaler, predictions
        else:
            return iterdf, trained_svc, scaler
    else:
        return iterdf


def tune_C(sample_data, target_labels, scoring_metric='accuracy', 
                cv_nfolds=3, test_split=0.2, verbose=False, n_processes=1):
    
    train_data = sample_data.copy()
    train_labels = target_labels
 
    # DATA - Fit classifier
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    # Set the parameters by cross-validation
    tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

    results ={} 
    if verbose:
        print("# Tuning hyper-parameters for %s" % scoring_metric)
    clf = GridSearchCV(svm.SVC(kernel='linear'), tuned_parameters, 
                            scoring=scoring_metric, cv=cv_nfolds, n_jobs=1) #n_processes)  
    clf.fit(train_data, train_labels)
    if verbose:
        print("Best parameters set found on development set:")
        print(clf.best_params_)
    if verbose:
        print("Grid scores on development set (scoring=%s):" % scoring_metric)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))    
    return clf 

def get_mutual_info_metrics(curr_test_labels, predicted_labels):
    mi = skmetrics.mutual_info_score(curr_test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(curr_test_labels, predicted_labels)
    log2_mi = computeMI(curr_test_labels, predicted_labels)
    mi_dict = {'heldout_MI': mi, 
               'heldout_aMI': ami, 
               'heldout_log2MI': log2_mi}
    return mi_dict


def fit_shuffled(zdata, targets, do_pchoose=False, return_clf=False, 
                class_name='morphlevel', class_values=[0, 106], **clf_params):
#                C_value=None, test_split=0.2, cv_nfolds=5, randi=0, 
#                verbose=False, return_clf=False,
#                class_types=[0, 106], class_name='morph', 
#                do_pchoose=False, return_svc=False, inum=0):
    '''
    Shuffle target labels, do fit_svm()
    '''
    iterdf=None

    labels_shuffled = targets['label'].copy().values 
    np.random.shuffle(labels_shuffled)
    targets['label'] = labels_shuffled
    if do_pchoose:
        clf_=None; scalar_=None; predicted_=None;
        p_list=[]
        class_a, class_b = class_values
        idf_, clf_, scaler_, predicted_ = fit_svm(zdata, targets, 
                                                return_clf=True, 
                                                return_predictions=True,
                                                **clf_params) 
        # Calculate P(choose B)
        for anchor in class_values: #[class_a, class_b]:
            tmpd = idf_.copy()
            curr_guesses = predicted_[predicted_['true']==anchor]['predicted']
            n_chooseB = len(curr_guesses[curr_guesses==class_b])
            p_chooseB = float(n_chooseB) / len(curr_guesses)
            tmpd['p_chooseB'] = p_chooseB
            tmpd['%s' % class_name] = anchor
            tmpd['n_samples']=len(curr_guesses)
            p_list.append(tmpd)  
        iterdf = pd.concat(p_list, axis=0) #.reset_index(drop=True)
    else:
        iterdf = fit_svm(zdata, targets, **clf_params)
    iterdf['condition'] = 'shuffled'

    if do_pchoose and return_clf:
        return iterdf, clf_, scaler_ 
    else:
        return iterdf
 

# --------------------------------------------------------------------------------
# Wrappers for fitting functions - specifies what type of analysis to do
# --------------------------------------------------------------------------------
def do_fit_within_fov(iter_num, curr_data, sdf, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='morphlevel', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True, do_shuffle=True, return_clf=False):

    #[gdf, MEANS, sdf, sample_size, cv] * n_times)
    '''
    Does 1 iteration of SVC fit for FOV (no global rois). 
    Assumes 'config' column in curr_data.
    Args

    class_a, class_b: int/str
        Values of the targets (e.g., value of morph level).
   
    do_shuffle: (bool)
        Runs fit_svm() twice, once reg and once with labels shuffled. 

    balance_configs: (bool)
        Add step to select same N trials per config (if not done before).

    '''   
    i_list=[]
    # Select train/test configs for clf A vs B
    if class_values is None:
        class_values = sdf[class_name].unique()
    train_configs = sdf[sdf[class_name].isin(class_values)].index.tolist() 

    # Get trial data for selected cells and config types
    sample_data = curr_data[curr_data['config'].isin(train_configs)]
    if balance_configs:
        # Make sure training data has equal nums of each config
        sample_data = aggr.equal_counts_df(sample_data)
    zdata = sample_data.drop('config', 1) 

    # Get labels
    targets = pd.DataFrame(sample_data['config'].copy(), columns=['config'])
    targets['label'] = [sdf[class_name][cfg] for cfg in targets['config'].values]

    # Fit params
    cv = C_value is None
    randi = random.randint(1, 10000) 
    clf_params = {'C_value': C_value, 
                  'cv_nfolds': cv_nfolds,
                  'test_split': test_split,
                  'randi': randi,
                  'verbose': verbose,
                  'inum': iter_num}
    # Fit
    idf_ = fit_svm(zdata, targets, **clf_params)
    idf_['condition'] = 'data'
    i_list.append(idf_)

    # Shuffle labels
    clf_params['C_value'] = idf_['C'].values[0] # Update c, in case we did tuning
    if do_shuffle:
        idf_shuffled = fit_shuffled(zdata, targets, **clf_params)
        idf_shuffled['condition'] = 'shuffled'
        idf_shuffled.index = [iter_num] 
        i_list.append(idf_shuffled)
 
    # Combine TRUE/SHUFF, add Meta info
    iter_df = pd.concat(i_list, axis=0) 
    iter_df['n_cells'] = zdata.shape[1]
    iter_df['n_trials'] = zdata.shape[0]
    #for label, g in targets.groupby(['label']):
    #    iter_df['n_samples_%i' % label] = len(g['label'])
    iter_df['iteration'] = iter_num

    if return_clf:
        return iter_df, curr_clf
    else:
        return iter_df 



def train_test_size_subset(iter_num, curr_data, sdf, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='morphlevel', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True,do_shuffle=True, return_clf=True):

    ''' Train with subset of configs, test on remainder. 
    Does 1 iteration of SVC fit for FOV (no global rois). 
    Assumes 'config' column in curr_data.
    Args

    class_a, class_b: int/str
        Values of the targets (e.g., value of morph level).
   
    do_shuffle: (bool)
        Runs fit_svm() twice, once reg and once with labels shuffled. 

    balance_configs: (bool)
        Add step to select same N trials per config (if not done before).

    '''    
    return_clf=True

    #### Select train/test configs for clf A vs B
    if variation_values is None:
        variation_values = sorted(sdf[variation_name].unique())
    #### Get all combinations of n_train_configs    
    combo_train_parvals = list(itertools.combinations(variation_values, 
                                                     n_train_configs))

    # Go thru all training sizes, then test on non-trained sizes
    df_list=[]
    #i=0
    for train_parvals in combo_train_parvals: #training_sets:
        train_list=[]
        # Get train configs
        train_configs = sdf[(sdf[class_name].isin(class_values))\
                          & (sdf[variation_name].isin(train_parvals))].index.tolist()
        # TRAIN SET: trial data for selected cells and config types
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        if balance_configs:
            trainset = aggr.equal_counts_df(trainset)
        train_data = trainset.drop('config', 1)
        # TRAIN SET: labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf[class_name][cfg] for cfg in targets['config'].values]
        targets['group'] = [sdf[variation_name][cfg] for cfg in targets['config'].values]
        train_transform = '_'.join([str(int(s)) for s in train_parvals])
        # Fit params
        randi = random.randint(1, 10000)
        clf_params = {'C_value': C_value, 
                      'cv_nfolds': cv_nfolds,
                      'test_split': test_split,
                      'randi': randi,
                      'verbose': verbose,
                      'inum': iter_num}
        # FIT
        idf_, trained_svc, trained_scaler = fit_svm(train_data, targets, 
                                                return_clf=True, **clf_params)
        idf_['condition'] = 'data'
        clf_params['C_value'] = idf_['C'].values[0]
        train_list.append(idf_)
        # Shuffle labels
        if do_shuffle:
            idf_shuffled = fit_shuffled(train_data, targets, return_clf=False,
                                        **clf_params)
            idf_shuffled = idf_shuffled[0]
            idf_shuffled['condition'] = 'shuffled'
            train_list.append(idf_shuffled)
        # Aggr training results
        df_train = pd.concat(train_list, axis=0)
        df_train['train_transform'] = train_transform
        df_train['test_transform'] = train_transform 
        df_train['n_trials'] = len(targets)
        df_train['novel'] = False
        df_list.append(df_train)

        # TEST - Select generalization-test set
        train_columns = df_train.columns.tolist() 
        test_parvals = [t for t in variation_values if t not in train_parvals]
        test_configs = sdf[((sdf[class_name].isin(class_values))\
                          & (sdf[variation_name].isin(test_parvals)))].index.tolist()
        test_subset = curr_data[curr_data['config'].isin(test_configs)]
        test_data = test_subset.drop('config', 1) 
        # TEST - targets
        test_targets = pd.DataFrame(test_subset['config'].copy(), columns=['config'])
        test_targets['label'] = sdf.loc[test_targets['config'].values][class_name].astype(float).values
        test_targets['group'] = sdf.loc[test_targets['config'].values][variation_name].astype(float).values
        # TEST - fit SVM (no need shuffle)
        test_list=[]
        for test_transform, curr_test_group in test_targets.groupby(['group']):
            testdict = dict((k, None) for k in train_columns) 
            curr_test_labels = curr_test_group['label'].values
            curr_test_data = test_data.loc[curr_test_group.index].copy()
            curr_test_data = trained_scaler.transform(curr_test_data)
            curr_test_score = trained_svc.score(curr_test_data, curr_test_labels)
            # Calculate additional metrics (MI)
            predicted_labels = trained_svc.predict(curr_test_data)
            #mi_dict = get_mutual_info_metrics(curr_test_labels, predicted_labels)
            #testdict.update(mi_dict) 
            # Add current test group info
            is_novel = train_transform!=test_transform
            testdict['heldout_test_score'] = curr_test_score
            testdict['test_transform'] = test_transform 
            testdict['novel']= is_novel
            testdict['n_trials'] = len(predicted_labels) 
            idf_test = pd.DataFrame(testdict, index=[iter_num])
            test_list.append(idf_test) 
        # Aggregate test results
        df_test = pd.concat(test_list, axis=0) 
        shadow_cols = ['C', 'train_score', 'test_score', 'fit_time', 'score_time',
                      'randi', 'train_transform']
        df_test[shadow_cols] = df_train[df_train.condition=='data'][shadow_cols].values 
        df_test['condition'] = 'data'   
        df_list.append(df_test)
 
    iterdf = pd.concat(df_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = iter_num 
    iterdf['n_cells'] = curr_data.shape[1]-1

    return iterdf

def train_test_gratings_single(iter_num, curr_data, sdf, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='ori', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True,do_shuffle=True, return_clf=True):
    ''' Train with single set of nonori params, test on same. 
    *only works with ORI
    Does 1 iteration of SVC fit for FOV (no global rois). 
    Assumes 'config' column in curr_data.

    Args
    class_name: (str)
        Column to classify
    class_values: (list)
        Values of the targets (e.g., value of ORI).
   
    do_shuffle: (bool)
        Runs fit_svm() twice, once reg and once with labels shuffled. 

    balance_configs: (bool)
        Add step to select same N trials per config (if not done before).

    '''    
    return_clf=True
    # All relevant GRATINGS params
    all_params=['ori', 'sf', 'size', 'speed']
    nonX_params = [x for x in all_params if x!=class_name]
    # Select train/test configs for clf A vs B
    if class_values is None:
        class_values = sorted(sdf[class_name].unique())
    # Select non-X params 
    variation_name = 'non_%s' % class_name
    nonX_df = sdf[nonX_params].drop_duplicates()
    sdf.loc[sdf.index, variation_name] = ['%.1f_%.1f_%.1f' % (v1, v2, v3) \
                            for v1, v2, v3 in sdf[nonX_params].values]
    # Go thru all training sizes, then test on non-trained sizes
    df_list=[]
    #i=0
    for ix, train_parvals in nonX_df.iterrows():
        train_list=[]
        # Get subset of configs that match current set of params
        curr_sdf = sdf[sdf[nonX_params].eq(train_parvals).all(axis=1)].copy()
        # Get train configs
        train_configs = curr_sdf[curr_sdf[class_name].isin(class_values)]\
                                .index.tolist()
        # TRAIN SET: trial data for selected cells and config types
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        if balance_configs:
            trainset = aggr.equal_counts_df(trainset)
        train_data = trainset.drop('config', 1)
        # TRAIN SET: labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf[class_name][cfg] for cfg \
                            in targets['config'].values]
        targets['group'] = [sdf[variation_name][cfg] for cfg \
                            in targets['config'].values]
        train_transform ='%.1f_%.1f_%.1f' % tuple(train_parvals.values)
        # Fit params
        randi = random.randint(1, 10000)
        clf_params = {'C_value': C_value, 
                      'cv_nfolds': cv_nfolds,
                      'test_split': test_split,
                      'randi': randi,
                      'verbose': verbose,
                      'inum': iter_num}
        # FIT
        idf_, trained_svc, trained_scaler = fit_svm(train_data, targets, 
                                                return_clf=True, **clf_params)
        idf_['condition'] = 'data'
        clf_params['C_value'] = idf_['C'].values[0]
        train_list.append(idf_)
        # Shuffle labels
        if do_shuffle:
            idf_shuffled = fit_shuffled(train_data, targets, **clf_params)
            idf_shuffled['condition'] = 'shuffled'
            train_list.append(idf_shuffled)
        # Aggr training results
        df_train = pd.concat(train_list, axis=0)
        df_train['train_transform'] = train_transform
        df_train['test_transform'] = train_transform 
        df_train['n_trials'] = len(targets)
        df_train['novel'] = False
        df_list.append(df_train)

    iterdf = pd.concat(df_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = iter_num 
    iterdf['n_cells'] = curr_data.shape[1]-1

    return iterdf



def train_test_size_single(iter_num, curr_data, sdf, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='morphlevel', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True,do_shuffle=True, return_clf=True):

    ''' Train with subset of configs, test on remainder. 
    Does 1 iteration of SVC fit for FOV (no global rois). 
    Assumes 'config' column in curr_data.
    Args

    class_a, class_b: int/str
        Values of the targets (e.g., value of morph level).
   
    do_shuffle: (bool)
        Runs fit_svm() twice, once reg and once with labels shuffled. 

    balance_configs: (bool)
        Add step to select same N trials per config (if not done before).

    '''    
    return_clf=True

    #### Select train/test configs for clf A vs B
    if variation_values is None:
        variation_values = sorted(sdf[variation_name].unique())

    # Go thru all training sizes, then test on non-trained sizes
    df_list=[]
    #i=0
    for train_parval in variation_values: #training_sets:
        train_list=[]
        # Get train configs
        train_configs = sdf[(sdf[class_name].isin(class_values))\
                          & (sdf[variation_name]==train_parval)].index.tolist()
        # TRAIN SET: trial data for selected cells and config types
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        if balance_configs:
            trainset = aggr.equal_counts_df(trainset)
        train_data = trainset.drop('config', 1)
        # TRAIN SET: labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf[class_name][cfg] for cfg in targets['config'].values]
        targets['group'] = [sdf[variation_name][cfg] for cfg in targets['config'].values]
        train_transform = train_parval #'_'.join([str(int(s)) for s in train_parvals])
        # Fit params
        randi = random.randint(1, 10000)
        clf_params = {'C_value': C_value, 
                      'cv_nfolds': cv_nfolds,
                      'test_split': test_split,
                      'randi': randi,
                      'verbose': verbose,
                      'inum': iter_num}
        # FIT
        idf_, trained_svc, trained_scaler = fit_svm(train_data, targets, 
                                                return_clf=True, **clf_params)
        idf_['condition'] = 'data'
        clf_params['C_value'] = idf_['C'].values[0]
        train_list.append(idf_)
        # Shuffle labels
        if do_shuffle:
            idf_shuffled = fit_shuffled(train_data, targets, **clf_params)
            idf_shuffled['condition'] = 'shuffled'
            train_list.append(idf_shuffled)
        # Aggr training results
        df_train = pd.concat(train_list, axis=0)
        df_train['train_transform'] = train_transform
        df_train['test_transform'] = train_transform 
        df_train['n_trials'] = len(targets)
        df_train['novel'] = False
        df_list.append(df_train)

        # TEST - Select generalization-test set
        train_columns = df_train.columns.tolist() 
        #### Select generalization-test set
        test_configs = sdf[((sdf[class_name].isin(class_values))\
                         & (sdf[variation_name]!=train_parval))].index.tolist()
        test_subset = curr_data[curr_data['config'].isin(test_configs)]
        test_data = test_subset.drop('config', 1) 
        # TEST - targets
        test_targets = pd.DataFrame(test_subset['config'].copy(), columns=['config'])
        test_targets['label'] = sdf.loc[test_targets['config'].values][class_name].astype(float).values
        test_targets['group'] = sdf.loc[test_targets['config'].values][variation_name].astype(float).values
        # TEST - fit SVM (no need shuffle)
        test_list=[]
        for test_transform, curr_test_group in test_targets.groupby(['group']):
            testdict = dict((k, None) for k in train_columns) 
            curr_test_labels = curr_test_group['label'].values
            curr_test_data = test_data.loc[curr_test_group.index].copy()
            curr_test_data = trained_scaler.transform(curr_test_data)
            curr_test_score = trained_svc.score(curr_test_data, curr_test_labels)
            # Calculate additional metrics (MI)
            predicted_labels = trained_svc.predict(curr_test_data)
            #mi_dict = get_mutual_info_metrics(curr_test_labels, predicted_labels)
            #testdict.update(mi_dict) 
            # Add current test group info
            is_novel = train_transform!=test_transform
            testdict['heldout_test_score'] = curr_test_score
            testdict['test_transform'] = test_transform 
            testdict['novel']= is_novel
            testdict['n_trials'] = len(predicted_labels) 
            idf_test = pd.DataFrame(testdict, index=[iter_num])
            test_list.append(idf_test) 
        # Aggregate test results
        df_test = pd.concat(test_list, axis=0) 
        shadow_cols = ['C', 'train_score', 'test_score', 'fit_time', 'score_time',
                      'randi', 'train_transform']
        for c in shadow_cols:
            assert len(df_train[df_train.condition=='data'])==1
            #print(df_train[df_train.condition=='data'][c].values[0])
            #print(df_test[c].shape)
            df_test[c] = df_train[df_train.condition=='data'][c].values[0]
        df_test['condition'] = 'data'   
        df_list.append(df_test)
 
    iterdf = pd.concat(df_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = iter_num 
    iterdf['n_cells'] = curr_data.shape[1]-1

    return iterdf



def train_test_morph(iter_num, curr_data, sdf, midp=53, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='morphlevel', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True,do_shuffle=True, return_clf=True):
    #fit_psycho=True, P_model='weibull', 
    # par0=np.array([0.5, 0.5, 0.1]), nfits=20):
    '''
    Test generalization to morph stimuli (combine diff sizes)
    train_transform: 0_106 (class types)
    test_transform:  intermediate morphs

    Note: 
    If do_pchoose=True for fit_shuffled(), 
    returns df updated with ['p_chooseB', 'morphlevel', n_samples']. 
    '''   
    df_list=[]
    # Select train/test configs for clf A vs B
    if variation_values is None and variation_name is not None:
        variation_values = sorted(sdf[variation_name].unique())

    # Get train configs -- ANCHORS (A/B)
    if class_values is None:
        class_values = sdf[class_name].unique()
        assert len(class_values)==2, "Wrong # of anchors (%s): %s" \
                % (class_name, str(class_values))
    train_configs = sdf[sdf[class_name].isin(class_values)].index.tolist() 

    # TRAIN SET: trial data for selected cells and config types
    trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
    if balance_configs:
        trainset = aggr.equal_counts_df(trainset)
    train_data = trainset.drop('config', 1)
    # TRAIN SET: labels
    targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
    targets['label'] = [sdf[class_name][cfg] for cfg in targets['config'].values]
    if variation_name is not None:
        targets['group'] = [sdf[variation_name][cfg] for cfg \
                            in targets['config'].values]
    train_transform = '_'.join([str(c) for c in class_values]) #'anchor'

    # Fit params
    randi = random.randint(1, 10000)
    clf_params = {'C_value': C_value, 
                  'cv_nfolds': cv_nfolds,
                  'test_split': test_split,
                  'randi': randi,
                  'verbose': verbose,
                  'inum': iter_num}
    # Train SVM ---------------------------------------------------
    randi = random.randint(1, 10000)
    idf_, clf_, scaler_, predicted_ = fit_svm(train_data, targets, 
                                                return_clf=True,
                                                return_predictions=True, 
                                                **clf_params)
    clf_params['C_value'] = idf_['C'].values[0]

    # Calculate P(choose B)
    p_list=[]
    class_a, class_b = class_values
    for anchor in class_values: #[class_a, class_b]:
        tmpdf = idf_.copy()
        curr_guesses = predicted_[predicted_['true']==anchor]['predicted']
        n_chooseB = len(curr_guesses[curr_guesses==class_b])
        p_chooseB = float(n_chooseB) / len(curr_guesses)
        tmpdf['p_chooseB'] = p_chooseB
        tmpdf['%s' % class_name] = anchor
        tmpdf['n_samples']=len(curr_guesses) 
        tmpdf['condition'] = 'data'
        p_list.append(tmpdf)
    # Shuffle labels
    if do_shuffle:
        idf_shuff, clf_shuff, scaler_shuff = fit_shuffled(train_data, 
                                    targets, 
                                    do_pchoose=True, return_clf=True, 
                                    class_name=class_name, 
                                    class_values=class_values, 
                                    **clf_params)
        idf_shuff['condition'] = 'shuffled'
        p_list.append(idf_shuff) 
    # Aggr current train results
    df_train = pd.concat(p_list, axis=0)    
    df_train['train_transform'] = train_transform
    df_train['test_transform'] = train_transform
    df_train['novel'] = False
    df_train['n_trials'] = len(targets)
 
    df_list.append(df_train)

    # TEST SET --------------------------------------------------------
    train_columns = df_train.columns.tolist()
    novel_class_values = [c for c in sdf[class_name].unique() \
                            if c not in class_values]
    test_configs = sdf[sdf[class_name].isin(novel_class_values)]\
                            .index.tolist()
    testset = curr_data[curr_data['config'].isin(test_configs)]
    test_data = testset.drop('config', 1) 
    # Get labels.
    test_targets = pd.DataFrame(testset['config'].copy(), columns=['config'])
    test_targets['label'] = [sdf[class_name][cfg] for cfg \
                                in test_targets['config'].values]
    #test_targets['group'] = [sdf[variation_name][cfg] for cfg \
    #                            in test_targets['config'].values]
       
    # Test SVM
    test_list=[]
    for test_transform, curr_test_group in test_targets.groupby(['label']):
        p_list=[]
        testdict = dict((k, None) for k in train_columns) 
        if do_shuffle:
            shuffdict = dict((k, None) for k in train_columns)
        curr_test_labels = curr_test_group['label'].values 
        curr_test_data = test_data.loc[curr_test_group.index].copy()
        curr_test_data = scaler_.transform(curr_test_data)
        # predict
        predicted_labels = clf_.predict(curr_test_data)
        predictions = pd.DataFrame({'true': curr_test_labels, 
                                    'predicted': predicted_labels})
        # Calculate "score" rename labels, into "0" or "106"
        class_a, class_b = sorted(class_values)
        if test_transform in [-1, midp]: # Ignore midp trials
            # rando assign values
            split_labels = [class_a if i<0.5 else class_b \
                            for i in np.random.rand(len(curr_test_labels),)]
        elif test_transform < midp and test_transform!=-1:
            split_labels = [class_a]*len(curr_test_labels) 
        elif test_transform > midp:
            split_labels = [class_b]*len(curr_test_labels)
        curr_test_score = clf_.score(curr_test_data, split_labels) 

        # predict p_chooseB
        n_chooseB = len(predictions[predictions['predicted']==class_b])
        p_chooseB = float(n_chooseB) / len(predictions)
        testdict.update({'p_chooseB': p_chooseB, 
                         'heldout_test_score': curr_test_score})
        # Calculate additional metrics (MI)
        #mi_dict = get_mutual_info_metrics(split_labels, predicted_labels)
        #testdict.update(mi_dict) 
        # Add current test group info
        testdict['condition'] = 'data'
        test_ = pd.DataFrame(testdict, index=[iter_num])
        p_list.append(test_) 

        # shuffled:  predict p_chooseB
        testdict_shuff = testdict.copy()
        curr_score_shuff = clf_shuff.score(curr_test_data, split_labels) 
        predicted_labels = np.array(clf_shuff.predict(curr_test_data))
        n_chooseB = len(np.where(predicted_labels==class_b)[0])
        p_chooseB = n_chooseB/float(len(predicted_labels))
        testdict_shuff['heldout_test_score'] = curr_score_shuff
        testdict_shuff['p_chooseB'] = p_chooseB
        # shuffled:  mutual info metrics
        #mi_dict = get_mutual_info_metrics(split_labels, predicted_labels)
        #testdict.update(mi_dict)
        testdict_shuff['condition'] = 'shuffled' 
        test_shuff_ = pd.DataFrame(testdict_shuff, index=[iter_num])
        p_list.append(test_shuff_) 

        # aggr current test results
        idf_test = pd.concat(p_list, axis=0)
        idf_test['n_samples'] = len(curr_test_labels)
        idf_test['test_transform'] = test_transform 
        idf_test['novel'] = train_transform!=test_transform
        idf_test['n_trials'] = len(predicted_labels)
        idf_test[class_name] = test_transform
        test_list.append(idf_test) 

    df_test = pd.concat(test_list, axis=0)

    # Add shadow cols to include whatever train df has
    shadow_cols = ['C', 'train_score', 'test_score', 'fit_time', 'score_time',
                  'randi', 'train_transform']
    for c in shadow_cols:
        assert len(df_train[df_train.condition=='data'][c].unique())==1
        df_test[c] = df_train[df_train.condition=='data'][c].unique()[0]
    df_list.append(df_test)

    # Aggregate 
    iterdf = pd.concat(df_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = iter_num 
    
    # fit curve?
    param_names = ['threshold', 'width', 'lambda', 'gamma', 'eta']
    iterdf[param_names] = None
    max_morph = float(iterdf['morphlevel'].max())
    for cond, idf in iterdf.groupby('condition'):
        df_ = idf[idf.morphlevel!=-1][['morphlevel', 'n_samples', 'p_chooseB']].copy()
        df_['n_chooseB'] = df_['n_samples']*df_['p_chooseB']
        df_['morph_percent'] = df_['morphlevel']/max_morph
        fits = psignifit_neurometric(df_.sort_values(by='morphlevel'))
        #print(fits) 
        #fits = df_.groupby(df_.index).apply(psignifit_neurometric)
        iterdf.loc[idf.index, param_names] = fits[param_names].values

    iterdf[param_names] = iterdf[param_names].astype(float) 
#    if fit_psycho:
#        iterdf['threshold']=None
#        iterdf['slope'] = None
#        iterdf['lapse'] = None
#        iterdf['likelihood'] = None
#        for dcond, mdf in iterdf.groupby(['condition']):
#            data = mdf[mdf.morphlevel!=-1].sort_values(by=['morphlevel'])\
#                            [['morphlevel', 'n_samples', 'p_chooseB']].values.T
#            max_v = max([class_a, class_b])
#            data[0,:] /= float(max_v)
#            try:
#                par, L = mle_weibull(data, P_model=P_model, parstart=par0, nfits=nfits) 
#                iterdf['threshold'].loc[mdf.index] = float(par[0])
#                iterdf['slope'].loc[mdf.index] = float(par[1])
#                iterdf['lapse'].loc[mdf.index] = float(par[2])
#                iterdf['likelihood'].loc[mdf.index] = float(L)
#            except Exception as e:
#                traceback.print_exc()
#                continue

    return iterdf

def psignifit_neurometric(df): #, ni):
    '''Fit data array using psignifit (same params as behavior)'''
    fit_=None
    param_names = ['threshold', 'width', 'lambda', 'gamma', 'eta']
    n_params = len(param_names)

    opts = dict()
    opts['sigmoidName'] = 'gauss'
    opts['expType'] = 'YesNo'
    opts['confP'] = np.tile(0.95, n_params)

    data = df[['morph_percent', 'n_chooseB', 'n_samples']].values
    try:
        res = ps.psignifit(data, opts)
        fit_ = pd.Series(res['Fit'], index=param_names) #, name=ni)
    except AssertionError:
        # no fit
        fit_ = pd.Series([None]*n_params, index=param_names)
    
    return fit_


def train_test_morph_single(iter_num, curr_data, sdf, midp=53, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='morphlevel', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True,do_shuffle=True, return_clf=True):
    #fit_psycho=True, P_model='weibull', 
    # par0=np.array([0.5, 0.5, 0.1]), nfits=20):
    '''
    Test generalization to morph stimuli (combine diff sizes)
    train_transform: 0_106 (class types)
    test_transform:  intermediate morphs

    Note: 
    If do_pchoose=True for fit_shuffled(), 
    returns df updated with ['p_chooseB', 'morphlevel', n_samples']. 
    '''   
    assert len(class_values)==2, "Wrong # of anchors (%s): %s" \
                    % (class_name, str(class_values))
    # Select train/test configs for clf A vs B
    if variation_values is None:
        variation_values = sorted(sdf[variation_name].unique())
    # Go thru all training sizes, then test on non-trained sizes
    df_list=[]
    for train_parval in variation_values: #training_sets: 
        train_configs = sdf[(sdf[class_name].isin(class_values))\
                          & (sdf[variation_name]==train_parval)]\
                        .index.tolist() 
        # TRAIN SET: trial data for selected cells and config types
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        if balance_configs:
            trainset = aggr.equal_counts_df(trainset)
        train_data = trainset.drop('config', 1)
        # TRAIN SET: labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf[class_name][cfg] for cfg \
                            in targets['config'].values]
        targets['group'] = [sdf[variation_name][cfg] for cfg \
                            in targets['config'].values]
        train_transform = train_parval  
        # Fit params
        randi = random.randint(1, 10000)
        clf_params = {'C_value': C_value, 
                      'cv_nfolds': cv_nfolds,
                      'test_split': test_split,
                      'randi': randi,
                      'verbose': verbose,
                      'inum': iter_num}
        # Train SVM ---------------------------------------------------
        randi = random.randint(1, 10000)
        idf_, clf_, scaler_, predicted_ = fit_svm(train_data, targets, 
                                                    return_clf=True,
                                                    return_predictions=True, 
                                                    **clf_params)
        clf_params['C_value'] = idf_['C'].values[0]
        # Calculate P(choose B)
        p_list=[]
        class_a, class_b = class_values
        for anchor in class_values: #[class_a, class_b]:
            tmpdf = idf_.copy()
            guesses = predicted_[predicted_['true']==anchor]['predicted']
            n_chooseB = len(guesses[guesses==class_b])
            p_chooseB = float(n_chooseB) / len(guesses)
            tmpdf['p_chooseB'] = p_chooseB
            tmpdf['%s' % class_name] = anchor
            tmpdf['n_samples']=len(guesses) 
            tmpdf['condition'] = 'data'
            p_list.append(tmpdf)
        # Shuffle labels
        if do_shuffle:
            idf_shuff, clf_shuff, scaler_shuff = fit_shuffled(train_data, 
                                        targets, 
                                        do_pchoose=True, return_clf=True, 
                                        class_name=class_name, 
                                        class_values=class_values, 
                                        **clf_params)
            idf_shuff['condition'] = 'shuffled'
            p_list.append(idf_shuff) 
        # Aggr current train results
        df_train = pd.concat(p_list, axis=0)    
        df_train['train_transform'] = train_transform
        df_train['test_transform'] = train_transform
        df_train['novel'] = False
        df_train['n_trials'] = len(targets) 
        df_list.append(df_train)

        # TEST SET --------------------------------------------------------
        train_columns = df_train.columns.tolist()
        novel_class_values = [c for c in sdf[class_name].unique() \
                                if c not in class_values]
        # Get MORPHS at current train size
        #test_configs = sdf[(sdf[class_name].isin(novel_class_values))
        #                 & (sdf[variation_name]==train_parval)]\
        #                .index.tolist()
        #test_configs = sdf[(sdf[class_name].isin(novel_class_values))]\
        #                .index.tolist()
        #test_configs = sdf[sdf[variation_name]!=train_parval].index.tolist()
        test_configs = sdf.index.tolist()
        testset = curr_data[curr_data['config'].isin(test_configs)]
        test_data = testset.drop('config', 1) 
        # Get labels.
        test_targets = pd.DataFrame(testset['config'].copy(), \
                                    columns=['config'])
        test_targets['label'] = [sdf[class_name][cfg] for cfg \
                                    in test_targets['config'].values]
        test_targets['group'] = [sdf[variation_name][cfg] for cfg \
                                    in test_targets['config'].values]   
        # Test SVM
        test_list=[]
        for (test_morph, test_parval), curr_test_group in test_targets.groupby(['label', 'group']):
            if test_morph in class_values and test_parval==train_parval:
                continue
            p_list=[]
            testdict = dict((k, None) for k in train_columns) 
            if do_shuffle:
                shuffdict = dict((k, None) for k in train_columns)
            curr_test_labels = curr_test_group['label'].values 
            curr_test_data = test_data.loc[curr_test_group.index].copy()
            curr_test_data = scaler_.transform(curr_test_data)
            # predict
            predicted_labels = clf_.predict(curr_test_data)
            predictions = pd.DataFrame({'true': curr_test_labels, 
                                        'predicted': predicted_labels})
            # Calculate "score" rename labels, into "0" or "106"
            class_a, class_b = class_values
            if test_morph in [-1, midp]: # Ignore midp trials
                # rando assign values
                split_labels = [class_a if i<0.5 else class_b for i \
                                in np.random.rand(len(curr_test_labels),)]
            elif test_morph < midp and test_morph!=-1:
                split_labels = [class_a]*len(curr_test_labels) 
            elif test_morph > midp:
                split_labels = [class_b]*len(curr_test_labels)
            curr_test_score = clf_.score(curr_test_data, split_labels) 

            # predict p_chooseB
            n_chooseB = len(predictions[predictions['predicted']==class_b])
            p_chooseB = float(n_chooseB) / len(predictions)
            testdict.update({'p_chooseB': p_chooseB, 
                             'heldout_test_score': curr_test_score})
            # Calculate additional metrics (MI)
            #mi_dict = get_mutual_info_metrics(split_labels,predicted_labels)
            #testdict.update(mi_dict) 
            # Add current test group info
            testdict['condition'] = 'data'
            test_ = pd.DataFrame(testdict, index=[iter_num])
            p_list.append(test_) 

            # shuffled:  predict p_chooseB
            testdict_shuff = testdict.copy()
            curr_score_shuff = clf_shuff.score(curr_test_data, split_labels) 
            predicted_labels = np.array(clf_shuff.predict(curr_test_data))
            n_chooseB = len(np.where(predicted_labels==class_b)[0])
            p_chooseB = n_chooseB/float(len(predicted_labels))
            testdict_shuff['heldout_test_score'] = curr_score_shuff
            testdict_shuff['p_chooseB'] = p_chooseB
            # shuffled:  mutual info metrics
            #mi_dict = get_mutual_info_metrics(split_labels,predicted_labels)
            #testdict.update(mi_dict)
            testdict_shuff['condition'] = 'shuffled' 
            test_shuff_ = pd.DataFrame(testdict_shuff, index=[iter_num])
            p_list.append(test_shuff_) 

            # aggr current test results
            idf_test = pd.concat(p_list, axis=0)
            idf_test['n_samples'] = len(curr_test_labels)
            idf_test['test_transform'] = test_parval #test_transform 
            idf_test['novel'] = train_parval!=test_parval #test_transform
            idf_test['n_trials'] = len(predicted_labels)
            idf_test[class_name] = test_morph #test_transform
            test_list.append(idf_test) 
        df_test = pd.concat(test_list, axis=0)

        # Add shadow cols to include whatever train df has
        shadow_cols = ['C', 'train_score', 'test_score', 
                       'fit_time', 'score_time',
                      'randi', 'train_transform']
        for c in shadow_cols:
            assert len(df_train[df_train.condition=='data'][c].unique())==1
            df_test[c] = df_train[df_train.condition=='data'][c].unique()[0]
        df_list.append(df_test)

    # Aggregate 
    iterdf = pd.concat(df_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = iter_num 

    # fit curve?
    param_names = ['threshold', 'width', 'lambda', 'gamma', 'eta']
    iterdf[param_names] = None
    max_morph = float(iterdf['morphlevel'].max())
    for cond, idf in iterdf.groupby(['condition', 'train_transform', 'test_transform']):
        df_ = idf[idf.morphlevel!=-1][['morphlevel', 'n_samples', 'p_chooseB']].copy()
        df_['n_chooseB'] = df_['n_samples']*df_['p_chooseB'].astype(float)
        df_['morph_percent'] = df_['morphlevel']/max_morph
        fits = psignifit_neurometric(df_.sort_values(by='morphlevel'))
        #print(fits) 
        #fits = df_.groupby(df_.index).apply(psignifit_neurometric)
        iterdf.loc[idf.index, param_names] = fits[param_names].values

    iterdf[param_names] = iterdf[param_names].astype(float) 
#
    # fit curve?
#    if fit_psycho:
#        iterdf['threshold']=None
#        iterdf['slope'] = None
#        iterdf['lapse'] = None
#        iterdf['likelihood'] = None
#        for dcond, mdf in iterdf.groupby(['condition']):
#            data = mdf[mdf.morphlevel!=-1].sort_values(by=['morphlevel'])\
#                            [['morphlevel', 'n_samples', 'p_chooseB']].values.T
#            max_v = max([class_a, class_b])
#            data[0,:] /= float(max_v)
#            try:
#                par, L = mle_weibull(data, P_model=P_model, parstart=par0, nfits=nfits) 
#                iterdf['threshold'].loc[mdf.index] = float(par[0])
#                iterdf['slope'].loc[mdf.index] = float(par[1])
#                iterdf['lapse'].loc[mdf.index] = float(par[2])
#                iterdf['likelihood'].loc[mdf.index] = float(L)
#            except Exception as e:
#                traceback.print_exc()
#                continue

    return iterdf





def fit_svm_mp(neuraldf, sdf, test_type, n_iterations=50, n_processes=1, 
                    break_correlations=False, n_cells_sample=None,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='morphlevel', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True,do_shuffle=True, return_clf=True,
                    verbose=False):
    iter_df = None
    inargs={'C_value': C_value,
            'cv_nfolds': cv_nfolds,  
            'test_split': test_split,
            'class_name': class_name,
            'class_values': class_values,
            'variation_name': variation_name,
            'variation_values': variation_values,
            'verbose': verbose,
            'do_shuffle': do_shuffle,
            'balance_configs': balance_configs,
            'n_train_configs': n_train_configs
    }
 
    #### Define MP worker
    results = []
    terminating = mp.Event() 
    def worker(out_q, n_iters, **kwargs):
        i_list = []        
        for ni in n_iters:
            # Decoding -----------------------------------------------------
            start_t = time.time()
            i_df = select_test(ni, test_type, neuraldf, sdf, # can access from outer 
                               break_correlations, **inargs) 
            if i_df is None:
                out_q.put(None)
                raise ValueError("No results for current iter")
            end_t = time.time() - start_t
            #print("--> Elapsed time: {0:.2f}sec".format(end_t))
            i_list.append(i_df)
        iterdf_chunk = pd.concat(i_list, axis=0)
        out_q.put(iterdf_chunk)  
    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=worker, 
                           args=(
                                out_q, 
                                iter_list[chunksize * i:chunksize * (i + 1)]),
                           kwargs=inargs)
            procs.append(p)
            p.start() # start asynchronously
        # Collect all results into 1 results dict. 
        results = []
        for i in range(n_processes):
            results.append(out_q.get(99999))
        # Wait for all worker processes to finish
        for p in procs:
            p.join() # will block until finished
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
        terminating.set()
    finally:
        for p in procs:
            p.join()

    if len(results)>0:
        iterdf = pd.concat(results, axis=0)

    return iterdf



def select_test(ni, test_type, ndata, sdf, break_correlations, **kwargs):
  
    # Shuffle trials, if spec.
    if break_correlations:
        ndata = shuffle_trials(ndata.copy())
 
    #if test_type in ['train_test_size_subset', 'train_test_size_single']:
    #    kwargs['return_clf'] = True


    curri = None
    try:
        if test_type=='size_subset':
            curri = train_test_size_subset(ni, ndata, sdf, **kwargs)
        elif test_type=='size_single':
            curri = train_test_size_single(ni, ndata, sdf, **kwargs) 
        elif test_type=='morph_single':
            curri = train_test_morph_single(ni, ndata, sdf, **kwargs) 
        elif test_type=='morph':
            curri = train_test_morph(ni, ndata, sdf, **kwargs)
        elif test_type=='ori_single':
            curri = train_test_gratings_single(ni, ndata, sdf, **kwargs)
        else:
             curri = do_fit_within_fov(ni, ndata, sdf, **kwargs) 
        #print('done')
        curri['iteration'] = ni 
    except SettingWithCopyError:
        print("HANDLING")
        finfo = getframeinfo(currentframe())
        print(frameinfo.lineno) 
    except Exception as e:
        traceback.print_exc()
        raise(e)
        return None

    return curri


# --------------------------------------------------------------------
# Save info
# --------------------------------------------------------------------

def create_results_id(C_value=None,
                    visual_area='varea', trial_epoch='stimulus', 
                    response_type='dff', responsive_test='resp', 
                    break_correlations=False, 
                    match_rfs=False, overlap_thr=None): 
    '''
    test_type: generatlization test name (size_single, size_subset, morph, morph_single)
    trial_epoch: mean val over time period (stimulus, plushalf, baseline) 
    '''
    C_str = 'tuneC' if C_value is None else 'C%.2f' % C_value
    corr_str = 'nocorrs' if break_correlations else 'intact'
    #if match_rfs:
    if match_rfs is False and overlap_thr is None:
        rf_str = 'noRF'
    elif match_rfs is True and (overlap_thr is None or overlap_thr==0):
        rf_str = 'matchRF'
    elif match_rfs is True and overlap_thr>0.:
        rf_str = 'matchRFoverlap%.2f' % overlap_thr
    else:
        # match_rfs is False and overlap_thr is not None:
        rf_str = 'overlap%.2f' % overlap_thr

    #overlap_str = 'noRF' if overlap_thr is None else 'overlap%.2f' % overlap_thr
    #test_str='all' if test_type is None else test_type
    response_str = '%s-%s' % (response_type, responsive_test)
    results_id='%s__%s__%s__%s__%s__%s' \
                % (visual_area, response_str, trial_epoch, rf_str, C_str, corr_str)
   
    return results_id

def create_aggregate_id(C_value=None,
                    trial_epoch='stimulus', 
                    response_type='dff', responsive_test='resp', 
                    match_rfs=False, overlap_thr=None): 
    '''
    test_type: generatlization test name (size_single, size_subset, morph, morph_single)
    trial_epoch: mean val over time period (stimulus, plushalf, baseline) 
    '''
    #test_str='all' if test_type is None else test_type
    #response_str = '%s-%s' % (response_type, responsive_test)
    if overlap_thr is not None and isinstance(overlap_thr, (list, np.ndarray)):
        sub_results_id = create_results_id(C_value=C_value, visual_area='None', 
                                    trial_epoch=trial_epoch, 
                                    response_type=response_type,
                                    responsive_test=responsive_test,
                                    break_correlations=False, 
                                    match_rfs=match_rfs, overlap_thr=overlap_thr[0])
        tmp_id = '__'.join(sub_results_id.split('__')[1:-1])
        parts_ = tmp_id.split('__')
        print(parts_)
        aggr_id = '__'.join([parts_[0], parts_[1], parts_[3]])

    else:
        sub_results_id = create_results_id(C_value=C_value, visual_area='None', 
                                    trial_epoch=trial_epoch, 
                                    response_type=response_type,
                                    responsive_test=responsive_test,
                                    break_correlations=False, 
                                    match_rfs=match_rfs, overlap_thr=overlap_thr)
        aggr_id = '__'.join(sub_results_id.split('__')[1:-1])
    #results_id='%s__%s' % (class_name, sub_id)
   
    return aggr_id #results_id

# --------------------------------------------------------------------
# BY_FOV 
# --------------------------------------------------------------------
def decode_from_fov(datakey, experiment, neuraldf,
                    sdf=None, test_type=None, results_id='results',
                    n_iterations=50, n_processes=1, break_correlations=False,
                    traceid='traces001', verbose=False,
                    rootdir='/n/coxfs01/2p-data', **in_args): 
    '''
    Fit FOV n_iterations times (multiproc). Save all iterations in dataframe.
    '''
    curr_areas = neuraldf['visual_area'].unique()
    assert len(curr_areas)==1, "Too many visual areas in ndf: %s" % str(curr_areas)
    visual_area = curr_areas[0]

    # Set output dir and file
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                        'FOV%i_*' % fovnum, 'combined_%s_static' % experiment, 
                        'traces', '%s*' % traceid))[0]
    test_str = 'default' if test_type is None else test_type
    class_name = in_args['class_name']
    curr_dst_dir = os.path.join(traceid_dir, 'decoding', class_name, test_str)
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
        print("... saving tmp results to:\n  %s" % curr_dst_dir)
    results_outfile = os.path.join(curr_dst_dir, '%s.pkl' % results_id)
    # Remove old results
    if os.path.exists(results_outfile):
        os.remove(results_outfile)
    # Zscore data 
    ndf_z = aggr.get_zscored_from_ndf(neuraldf) 
    n_cells = int(ndf_z.shape[1]-1) 
    class_name = in_args['class_name']
    class_values = in_args['class_values']
    if verbose:
        print("... BY_FOV [%s] %s, n=%i cells" % (visual_area, datakey, n_cells))
        print("... test: %s (class %s, values: %s)" % (test_str, class_name,
                                                        str(class_values)))
    # Stimulus info
    if sdf is None:
        match_stimulus_names = experiment=='blobs'
        sdf = aggr.get_stimuli(datakey, experiment, 
                                match_names=match_stimulus_names)
    # Decode
    start_t = time.time()
    iter_results = fit_svm_mp(ndf_z, sdf, test_type, 
                            n_iterations=n_iterations,
                            n_processes=n_processes,
                            break_correlations=break_correlations,
                            **in_args)
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))
    assert iter_results is not None, "NONE returned -- %s, %s" % (visual_area, datakey)

    # Add meta info
    iter_results['n_cells'] = n_cells 
    iter_results['visual_area'] = visual_area
    iter_results['datakey'] = datakey
    # Save all iterations
    with open(results_outfile, 'wb') as f:
        pkl.dump(iter_results, f, protocol=2)

    print('----------------------------------------------------')
    if test_type is not None:
        print(iter_results.groupby(['condition', 'train_transform']).mean())   
        print(iter_results.groupby(['condition', 'train_transform']).count())   
    else:
        class_name = in_args['class_name']
        class_values = in_args['class_values']
        print("Class: %s, values: %s" % (class_name, class_values))
        print(iter_results.groupby(['condition']).mean())   
    print("@@@@@@@@@ Done. %s|%s  @@@@@@@@@@" % (visual_area, datakey))
    print(results_outfile) 
    
    return

def shuffle_trials(ndf_z):
    rois_ = [r for r in ndf_z.columns if hutils.isnumber(r)]
    c_=[]
    for cfg, cg in ndf_z.groupby(['config']):
        n_trials = cg.shape[0]
        cg_r = pd.concat([cg[ri].sample(n=n_trials, replace=False)\
                          .reset_index(drop=True) for ri in rois_], axis=1)
        cg_r['config'] = cfg
        c_.append(cg_r)
    ndf_r = pd.concat(c_, axis=0).reset_index(drop=True)

    return ndf_r

# --------------------------------------------------------------------
# BY_NCELLS
# --------------------------------------------------------------------
def decode_by_ncells(n_cells_sample, experiment, GCELLS, NDATA, 
                    sdf=None, test_type=None, results_id='results',
                    n_iterations=50, n_processes=2, break_correlations=False,
                    dst_dir='/tmp', **in_args):
    '''
    Create psuedo-population by sampling n_cells from global_rois.
    Do decoding analysis
    '''
    curr_areas = GCELLS['visual_area'].unique()
    assert len(curr_areas)==1, "Too many visual areas in global cell df"
    visual_area = curr_areas[0]

    #### Set output dir and file
    results_outfile = os.path.join(dst_dir, \
                            '%s_%03d.pkl' % (results_id, n_cells_sample))
    # remove old file
    if os.path.exists(results_outfile):
        os.remove(results_outfile)

    #### Get neural means
    print("... Starting decoding analysis")

    # ------ STIMULUS INFO -----------------------------------------
    #if sdf is None:
    #    sdf = aggr.get_master_sdf(images_only=True)
    #sdf['config'] = sdf.index.tolist()

    
    # Decode
    try:
        start_t = time.time()
        iter_results = iterate_by_ncells(NDATA, GCELLS, sdf, test_type, 
                                n_cells_sample=n_cells_sample,
                                n_iterations=n_iterations,
                                n_processes=n_processes,
                                break_correlations=break_correlations,
                                **in_args)
        end_t = time.time() - start_t
        print("--> Elapsed time: {0:.2f}sec".format(end_t))
        assert iter_results is not None, "NONE -- %s (%i cells)" \
                % (visual_area, n_cells_sample)
    except Exception as e:
        traceback.print_exc()
        return None

    # DATA - concat 3 conds
    iter_results['visual_area'] = visual_area
    iter_results['datakey'] = 'aggregate'
    iter_results['n_cells'] = n_cells_sample

    with open(results_outfile, 'wb') as f:
        pkl.dump(iter_results, f, protocol=2)

    print_cols = ['n_cells', 'condition', 'train_score', 'test_score', 'heldout_test_score', 'iteration']
    if test_type is None:
        group_cols = ['condition', 'n_cells']
    else:
        group_cols = ['condition', 'n_cells', 'train_transform']

    if 'morph' in test_type:
        print_cols.extend(['morphlevel', 'threshold'])
        group_cols.extend(['morphlevel'])

    mean_iters = iter_results.groupby(group_cols).mean().reset_index()
    #print(mean_iters.columns)
    #print(iter_results['threshold'].unique())
    print(mean_iters[print_cols])   
    print("@@@@@@@ done. %s (n=%i cells) @@@@@@@@" % (visual_area,n_cells_sample))
    print(results_outfile) 
 
    return 




def iterate_by_ncells(NDATA, GCELLS, sdf, test_type, n_cells_sample=1,
                    n_iterations=50, n_processes=1, 
                    break_correlations=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='morphlevel', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True,do_shuffle=True, return_clf=True,
                    verbose=False):
    iterdf = None
    inargs={'C_value': C_value,
            'cv_nfolds': cv_nfolds,  
            'test_split': test_split,
            'class_name': class_name,
            'class_values': class_values,
            'variation_name': variation_name,
            'variation_values': variation_values,
            'verbose': verbose,
            'do_shuffle': do_shuffle,
            'balance_configs': balance_configs,
            'n_train_configs': n_train_configs
    }
 
    # Select how to filter trial-matching
    #equalize_by='config', match_all_configs=True,
    #with_replacement=False):   

    #train_labels = sdf[sdf[class_name].isin(class_values)][equalize_by].unique()
    common_labels = None #if match_all_configs else train_labels
    with_replacement=False

    #### Define MP worker
    results = []
    terminating = mp.Event() 
    def worker_by_ncells(out_q, n_iters, **kwargs):
        i_=[]
        for ni in n_iters:
            # Get new sample set
            #print("... sampling data, n=%i cells" % n_cells_sample)
            randi_cells = random.randint(1, 10000)
            try:
                neuraldf = sample_neuraldata_for_N_cells(n_cells_sample, 
                                         NDATA, GCELLS, 
                                         with_replacement=with_replacement,
                                         train_configs=common_labels, 
                                         randi=randi_cells)
                neuraldf = aggr.get_zscored_from_ndf(neuraldf.copy())
            except Exception as e:
                raise e 
            # Decoding -----------------------------------------------------
            start_t = time.time()
            i_df = select_test(ni, test_type, neuraldf, sdf, 
                               break_correlations, **inargs)  
            if i_df is None:
                out_q.put(None)
                raise ValueError("No results for current iter")
            end_t = time.time() - start_t
            print("--> --> Elapsed time: {0:.2f}sec".format(end_t))
            i_df['randi_cells'] = randi_cells
            i_.append(i_df)
        curr_iterdf = pd.concat(i_, axis=0)
        out_q.put(curr_iterdf) 
    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=worker_by_ncells, 
                           args=(
                                out_q, 
                                iter_list[chunksize * i:chunksize * (i + 1)]),
                           kwargs=inargs)
            procs.append(p)
            p.start() # start asynchronously
        # Collect all results into 1 results dict. 
        results = []
        for i in range(n_processes):
            results.append(out_q.get(99999))
        # Wait for all worker processes to finish
        for p in procs:
            p.join() # will block until finished
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
        terminating.set()
    finally:
        for p in procs:
            p.join()

    if len(results)>0:
        iterdf = pd.concat(results, axis=0)
        iterdf['n_cells'] = n_cells_sample

    return iterdf

def sample_neuraldata_for_N_cells(n_cells_sample, NDATA, GCELLS,  
                    with_replacement=False, train_configs=None, randi=None): 

    assert len(GCELLS['visual_area'].unique())==1, "Too many areas in GCELLS"
    va = GCELLS['visual_area'].unique()[0]

    # Get current global RIDs
    ncells_t = GCELLS.shape[0]                      
    # Random sample N cells out of all cells in area (w/o replacement)
    celldf = GCELLS.sample(n=n_cells_sample, replace=with_replacement, 
                                random_state=randi)
    curr_cells = celldf['global_ix'].values
    assert len(curr_cells)==len(np.unique(curr_cells))
    # Get corresponding neural data of selected datakeys and cells
    curr_dkeys = celldf['datakey'].unique()
    ndata0 = NDATA[(NDATA.visual_area==va) \
                 & (NDATA.datakey.isin(curr_dkeys))].copy()
    ndata0['cell'] = ndata0['cell'].astype(float)

    # Make sure equal num trials per condition for all dsets
    if train_configs is not None:
        count_these = ndata0[ndata0['config'].isin(train_configs)].copy() 
    else:
        count_these = ndata0.copy()
    min_ntrials_by_config = count_these[['datakey', 'config', 'trial']]\
                                    .drop_duplicates()\
                                    .groupby(['datakey'])['config']\
                                    .value_counts().min()
    #print("Min samples per config: %i" % min_ntrials_by_config)
    # Sample the data
    d_list=[]
    for dk, dk_rois in celldf.groupby(['datakey']):
        assert dk in ndata0['datakey'].unique(), "ERROR: %s not found" % dk
        # Get current trials, make equal to min_ntrials_by_config
        subdata = ndata0[(ndata0.datakey==dk) 
                       & (ndata0['cell'].isin(dk_rois['cell'].values))].copy()
        tmpd = pd.concat([tmat.sample(n=min_ntrials_by_config,\
                                    replace=False, random_state=None) \
                             for (rid, cfg), tmat in subdata\
                              .groupby(['cell', 'config'])], axis=0)
        tmpd['cell'] = tmpd['cell'].astype(float)

        # For each RID sample belonging to current dataset, get RID order
        # Use global index to get matching dset-relative cell ID
        sampled_cells = pd.concat([\
                            dk_rois[dk_rois['global_ix']==gid][['cell', 'global_ix']] 
                            for gid in curr_cells])
        sampled_dset_rois = sampled_cells['cell'].values
        sampled_global_rois = sampled_cells['global_ix'].values
        cell_lut = dict((k, v) for k, v in zip(sampled_dset_rois, sampled_global_rois))

        # Get response + config, replace dset roi  name with global roi name
        curr_ndata = pd.concat([tmpd[tmpd['cell']==rid][['config', 'response']]\
                            .rename(columns={'response': cell_lut[rid]})\
                            .sort_values(by='config').reset_index(drop=True) \
                            for rid in sampled_dset_rois], axis=1)
        # drop duplicate config columns
        curr_ndata = curr_ndata.loc[:,~curr_ndata.T.duplicated(keep='first')]
        d_list.append(curr_ndata)
    new_neuraldf = pd.concat(d_list, axis=1)[curr_cells]

    # And, get configs
    new_cfgs = pd.concat(d_list, axis=1)['config']
    assert new_cfgs.shape[0]==new_neuraldf.shape[0], "Bad trials"
    if len(new_cfgs.shape) > 1:
        #print("Requested configs: %s" % 'all' if train_configs is None \
        #        else str(train_configs)) 
        new_cfgs = new_cfgs.loc[:,~new_cfgs.T.duplicated(keep='first')]
        assert new_cfgs.shape[1]==1, "Bad configs: %s" \
                % str(celldf['datakey'].unique())

    new_ndf = pd.concat([new_neuraldf, new_cfgs], axis=1)
    
    # Restack
    ndf = aggr.unstacked_neuraldf_to_stacked(new_ndf, response_type='response', \
                        id_vars=['config', 'trial'])

    return ndf

# --------------------------------------------------------------------
# Main functions
# --------------------------------------------------------------------

def decoding_analysis(dk, va, experiment,  
                analysis_type='by_fov',trial_epoch='stimulus',
                traceid='traces001', 
                responsive_test='nstds', responsive_thr=10.,
                match_rfs=False, 
                rf_lim='percentile', rf_metric='fwhm_avg',
                overlap_thr=None,response_type='dff', 
                do_spherical_correction=False,
                test_type=None, 
                break_correlations=False,
                n_cells_sample=None, drop_repeats=True, 
                C_value=None, test_split=0.2, cv_nfolds=5, 
                images_only=False,
                class_name='morphlevel', class_values=None,
                variation_name='size', variation_values=None,
                n_train_configs=4, 
                balance_configs=True, do_shuffle=True,
                n_iterations=50, n_processes=1,
                rootdir='/n/coxfs01/2p-data', verbose=False,
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                visual_areas=['V1', 'Lm', 'Li']): 
    '''
    match_rfs: (bool)
        Limit included cells with RFs matched according to rf_lim. 

    rf_lim: (str)
        Limit included cells with RFs within specified range (default: percentile)
        TODO: match RF size 1-to-1
  
    overlap_thr: (None, float)
        Only include cells with RFs overlapping MINIMALLY by x %

    do_spherical_correction: (bool)
        Param. for which RF type to select

    break_correlations: (bool)
        Shuffle trials for each cell to break noise correlations

    images_only: (bool)
        Only include NON-fullfield stimuli (excludes lum controls or ff gratings).
        This will exclude older dsets that don't have lum controls, 
        or additional gratings datasets where only FF acquired. 
    '''
    # Metadata    
    sdata0, cells0 = aggr.get_aggregate_info(visual_areas=visual_areas, 
                                            return_cells=True)
    excluded = {'blobs': ['20190314_JC070_fov1', 
                          '20190426_JC078_fov1'], # '20190602_JC091_fov1'],
                'gratings': ['20190517_JC083_fov1']
    }
    sdata = sdata0[~sdata0['datakey'].isin(excluded[experiment])].copy()
    meta = sdata[sdata.experiment==experiment].copy()

    # Make sure stim configs match (if gratings)
    if analysis_type=='by_ncells':
        SDF = aggr.check_sdfs(meta['datakey'].unique(), experiment=experiment,
                            rename=True, return_incorrect=False, 
                            images_only=images_only,
                            return_all=False)
        incl_dkeys = SDF['datakey'].unique()
        print("**Keeping %i of %i total datakeys" \
                % (len(incl_dkeys), len(meta['datakey'].unique())))
        meta = meta[meta.datakey.isin(incl_dkeys)]


    # Load all the data
    NDATA0 = aggr.load_responsive_neuraldata(experiment, meta=meta,
                      traceid=traceid,
                      response_type=response_type, trial_epoch=trial_epoch,
                      responsive_test=responsive_test, 
                      responsive_thr=responsive_thr)
    # Outfile name
    results_id = create_results_id(
                                C_value=C_value, 
                                visual_area=va,
                                trial_epoch=trial_epoch,
                                response_type=response_type, 
                                responsive_test=responsive_test,
                                break_correlations=break_correlations,
                                match_rfs=match_rfs,
                                overlap_thr=overlap_thr) 
    print("~~~~~~~~~~~~~~~~ RESULTS ID ~~~~~~~~~~~~~~~~~~~~~")
    print(results_id)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # Classif params
    in_args={'class_name': class_name,
                'class_values': class_values,
                'variation_name': variation_name,
                'variation_values': variation_values,
                'n_train_configs': n_train_configs,
                'C_value': C_value,
                'cv_nfolds': cv_nfolds,
                'test_split': test_split,
                'balance_configs': balance_configs,
                'do_shuffle': do_shuffle,
                'verbose': verbose}

    # Assign global ix to all cells
    cells0 = aggr.get_all_responsive_cells(cells0, NDATA0)
    if match_rfs:
        # Match cells for RF size
        print("[RF]: matching RF size (%s)." % rf_lim)
        print(cells0[['visual_area','datakey','cell']]\
                .drop_duplicates()['visual_area'].value_counts())
        cells0 = get_cells_with_matched_rfs(cells0, sdata, 
                    rf_lim=rf_lim, rf_metric=rf_metric,
                    response_type=response_type, 
                    do_spherical_correction=do_spherical_correction)
        print("----> post:")
        print(cells0[['visual_area','datakey', 'cell']]\
                .drop_duplicates()['visual_area'].value_counts())
    if overlap_thr is not None:
        # only include cells w overlapping stimulus
        print("[RF]: Calculating overlap with stimuli.")
        print(cells0[['visual_area','datakey','cell']]\
                .drop_duplicates()['visual_area'].value_counts())
        cells0 = get_cells_with_overlap(cells0, sdata, greater_than=False,
                                        overlap_thr=overlap_thr)
        print("----> post:")
        print(cells0[['visual_area','datakey', 'cell']]\
                .drop_duplicates()['visual_area'].value_counts())

    if analysis_type=='by_ncells' and drop_repeats:
        # drop repeats
        print("~~~ dropping repeats ~~~")
        print(cells0[['visual_area','datakey', 'cell']]\
                .drop_duplicates()['visual_area'].value_counts())
        cells0 = aggr.unique_cell_df(cells0, criterion='max', 
                                        colname='cell')
        print("~~~ post ~~~")
        print(cells0[['visual_area','datakey', 'cell']]\
                .drop_duplicates()['visual_area'].value_counts()) 


    # Get final neuraldata
    
    NDATA = aggr.get_neuraldata_for_included_cells(cells0, NDATA0)

    match_stimulus_names = experiment=='blobs'
    # ANALYSIS.
    if analysis_type=='by_fov':
        # -------------------------------------------------------------
        # BY_FOV - for each fov, do_decode
        # -------------------------------------------------------------
        if dk is not None and va is None:
            meta = meta[(meta.datakey==dk)]
        elif dk is not None and va is not None:
            meta = meta[(meta.datakey==dk) & (meta.visual_area==va)]
        # otherwwise, just cycles thru all datakeys in visual area
        for (va, dk), g in meta.groupby(['visual_area', 'datakey']):   
            neuraldf = NDATA[(NDATA.visual_area==va) 
                           & (NDATA.datakey==dk)].copy()
            # Get stimuli
            sdf = aggr.get_stimuli(dk, experiment, 
                        match_names=match_stimulus_names)
            if experiment=='gratings' and class_name=='sf':
                # Need to convert to int
                sdf['sf'] = (sdf['sf']*10.).astype(int)
            print(va, dk, neuraldf.shape)
            if int(neuraldf.shape[0])==0:
                return None

            decode_from_fov(dk, experiment, neuraldf, sdf=sdf, 
                            test_type=test_type, 
                            results_id=results_id,
                            n_iterations=n_iterations, 
                            traceid=traceid,
                            break_correlations=break_correlations,
                            n_processes=n_processes, **in_args)
        print("--- done by_fov ---")

    elif analysis_type=='by_ncells':
        # -------------------------------------------------------------
        # BY_NCELLS - aggregate cells
        # -------------------------------------------------------------
            
        counts = cells0[['visual_area', 'datakey', 'cell']]\
                    .drop_duplicates().groupby(['visual_area'])\
                    .count().reset_index()
        cell_counts = dict((k, v) for (k, v) \
                        in zip(counts['visual_area'], counts['cell']))
        print("FINAL COUNTS:")
        print(cell_counts)
        min_ncells_total = min(cell_counts.values())

        if n_cells_sample is not None and int(n_cells_sample)>min_ncells_total:
            print("ERR: Sample size (%i) must be <= min ncells total (%i)" \
                    % (n_cells_sample, min_ncells_total))
            return None
           
        # Set global output dir (since not per-FOV):
        test_str = 'default' if test_type is None else test_type
        dst_dir = os.path.join(aggregate_dir, 'decoding', 
                            'py3_by_ncells', class_name, '%s' % test_str)
        curr_results_dir = os.path.join(dst_dir, 'files')
        if not os.path.exists(curr_results_dir):
            os.makedirs(curr_results_dir)
        print("... saving tmp results to:\n  %s" % curr_results_dir)

        # Save inputs
        filter_str = '__'.join(results_id.split('__')[1:4])
        whichdata = '%s_' % va if va is not None else ''
        inputs_file = os.path.join(curr_results_dir, 
                                'inputcells-%s%s.pkl' % (whichdata, filter_str))
        with open(inputs_file, 'wb') as f:
            pkl.dump(cells0, f, protocol=2) 

        # Stimuli
        sdf = aggr.get_master_sdf(experiment, images_only=False)
        if experiment=='gratings' and class_name=='sf':
            # Need to convert to int
            sdf['sf'] = (sdf['sf']*10.).astype(int)

        # NDATA0 = NDATA.copy()
        NDATA = NDATA[NDATA.config.isin(sdf.index.tolist())].copy() 

        # Get cells for current visual area
        GCELLS = cells0[cells0['visual_area']==va].copy()
        if n_cells_sample is not None: 
            decode_by_ncells(n_cells_sample, experiment, GCELLS, NDATA, 
                        sdf=sdf,
                        test_type=test_type, 
                        results_id=results_id,
                        n_iterations=n_iterations, 
                        break_correlations=break_correlations,
                        n_processes=n_processes,
                        dst_dir=curr_results_dir, **in_args)
            print("--- done %i." % n_cells_sample)
        else:
            # Loop thru NCELLS
            min_cells_total = min(cell_counts.values())
            reasonable_range = [2**i for i in np.arange(0, 10)]
            incl_range = [i for i in reasonable_range if i<min_cells_total]
            incl_range.append(min_cells_total)                
            print("Looping thru range: %s" % str(incl_range))
            for n_cells_sample in incl_range:
                decode_by_ncells(n_cells_sample, experiment, 
                        GCELLS, NDATA, sdf=sdf,
                        test_type=test_type, 
                        results_id=results_id,
                        n_iterations=n_iterations, 
                        break_correlations=break_correlations,
                        n_processes=n_processes,
                        dst_dir=curr_results_dir, **in_args)
                print("--- done %i." % n_cells_sample)

    return


# --------------------------------------------------------------------
# Aggregate functions
# --------------------------------------------------------------------
def aggregate_iterated_results(meta, class_name, experiment=None,
                      analysis_type='by_fov', test_type=None,
                      traceid='traces001',
                      trial_epoch='plushalf', responsive_test='nstds', 
                      C_value=1., break_correlations=False, 
                      overlap_thr=None, match_rfs=False,
                      rootdir='/n/coxfs01/2p-data',
                      aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    test_str = 'default' if test_type is None else test_type
    if experiment is None:
        experiment = 'gratings' if class_name=='ori' else 'blobs'
    iterdf=None
    missing_=[]
    d_list=[]
    if analysis_type=='by_fov':
        n_found = dict((k, 0) for k in ['V1', 'Lm', 'Li'])
        for (va, dk), g in meta.groupby(['visual_area', 'datakey']):
            results_id = create_results_id(C_value=C_value, 
                                           visual_area=va,
                                           trial_epoch=trial_epoch,
                                           responsive_test=responsive_test,
                                           break_correlations=break_correlations,
                                            match_rfs=match_rfs,
                                           overlap_thr=overlap_thr)
            try:
                session, animalid, fovn = hutils.split_datakey_str(dk)
                results_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                                  'FOV%i_*' % fovn,
                                  'combined_%s_*' % experiment, 
                                  'traces/%s*' % traceid, 
                                  'decoding', class_name, test_str))[0]
                results_fpath = os.path.join(results_dir, '%s.pkl' % results_id)
                assert os.path.exists(results_fpath), \
                                    'Not found:\n    %s' % results_fpath
                with open(results_fpath, 'rb') as f:
                    res = pkl.load(f)
                d_list.append(res)
            except Exception as e:
                missing_.append((va, dk))
                #traceback.print_exc()
                continue
            n_found[va] += 1

        for va, nf in n_found.items():
            print("(%s) Found %i paths" % (va, nf))

    elif analysis_type=='by_ncells':
        results_dir = glob.glob(os.path.join(aggregate_dir, 'decoding',\
                                'py3_by_ncells', class_name, test_str, 'files'))
        if len(results_dir)==0:
            print("No results by_ncells")
            return None, None
        results_dir=results_dir[0]
        for va, g in meta.groupby(['visual_area']):
            results_id = create_results_id(C_value=C_value, 
                                           visual_area=va,
                                           trial_epoch=trial_epoch,
                                           responsive_test=responsive_test,
                                           break_correlations=break_correlations,
                                            match_rfs=match_rfs,
                                           overlap_thr=overlap_thr)
            found_fpaths = glob.glob(os.path.join(\
                                    results_dir, '%s_*.pkl' % results_id))    
            print("(%s) Found %i paths" % (va, len(found_fpaths)))
            for fpath in found_fpaths:
                try:
                    with open(fpath, 'rb') as f:
                        res = pkl.load(f)
                    d_list.append(res)
                except Exception as e:
                    missing_.append(fpath)
                    #traceback.print_exc()
                    continue
    if len(d_list)>0:
        iterdf = pd.concat(d_list)
    
    return iterdf, missing_


def load_iterdf(meta, class_name, experiment=None,
                      analysis_type=None, test_type=None,
                      traceid='traces001', trial_epoch='stimulus', 
                      responsive_test='nstds', 
                      C_value=1, break_correlations=False, 
                      match_rfs=False, overlap_thr=None):
    '''
    Load and aggregate results
    '''
    if overlap_thr is None or not isinstance(overlap_thr, (list, np.ndarray)):
        overlap_thr = [overlap_thr]
    i_list=[]
    missing_dict = dict((k, {}) for k in overlap_thr)
    #print(overlap_thr)

    for overlap_val in overlap_thr:
        iterdf=None; iterdf_b=None;
        missing_=None; missing_b=None;
        iterdf, missing_ = aggregate_iterated_results(meta, 
                              class_name, experiment=experiment,
                              analysis_type=analysis_type,
                              test_type=test_type,
                              traceid=traceid,
                              trial_epoch=trial_epoch, 
                              responsive_test=responsive_test, 
                              C_value=C_value, break_correlations=False, 
                              match_rfs=match_rfs, overlap_thr=overlap_val)
        missing_dict[overlap_val]['intact'] = missing_

        if iterdf is None:
            continue
        iterdf['intact'] = True
        df_ = iterdf.copy()

        #if test_type is None:
        print('    checking for break-corrs')
        iterdf_b, missing_b = aggregate_iterated_results(meta, 
                                  class_name, experiment=experiment,
                                  analysis_type=analysis_type,                   
                                  test_type=test_type,
                                  traceid=traceid,
                                  trial_epoch=trial_epoch, 
                                  responsive_test=responsive_test, 
                                  C_value=C_value, break_correlations=True, 
                                  match_rfs=match_rfs, overlap_thr=overlap_val)
        missing_dict[overlap_val]['no_cc'] = missing_b
        if iterdf_b is not None:
            iterdf_b['intact'] = False
            df_ = pd.concat([iterdf, iterdf_b], axis=0)
        else:
            df_ = iterdf.copy()

        if df_ is not None:
            df_['overlap_thr'] = overlap_val

        i_list.append(df_)

    df = pd.concat(i_list, axis=0)

    return df, missing_dict

def average_across_iterations(iterdf, iterdf_b=None, analysis_type='by_fov', 
                              test_type=None):
    '''
    Get average value (across iterations) for each columns
    grouper=['visual_area', 'datakey', 'condition']
    if test_type is not None:
        grouper.extend(['novel', 'train_transform', 'test_transform'])
    '''
    grouper=['visual_area', 'datakey', 'condition']
    if analysis_type=='by_ncells':
        grouper.append('n_cells')
    if test_type is not None:
        grouper.extend(['novel', 'train_transform', 'test_transform'])
    df_intact = iterdf.groupby(grouper).mean().reset_index()
    df_intact['intact'] = True
    
    if iterdf_b is not None:
        df_nocc = iterdf_b.groupby(grouper).mean().reset_index()
        df_nocc['intact'] = False
        DF = pd.concat([df_intact, df_nocc], axis=0)
    else:
        DF = df_intact.copy()
    return DF


def average_across_iterations_by_fov(iterdf, analysis_type='by_fov', 
                           test_type='morph',
                        grouper=['visual_area', 'condition', 'datakey']):

    if test_type is not None:
        grouper.extend(['novel', 'train_transform', 'test_transform', 'morphlevel'])

    mean_by_iters = iterdf.groupby(grouper).mean().reset_index()
    return mean_by_iters


def average_within_iterations_by_ncells(iterdf, analysis_type='by_ncells', 
                              test_type='size_single',
                        grouper=['visual_area', 'condition', 'iteration']):
    '''
    For each iteration, get average of relevant columns. Returns distn of 
    iteration results.
    '''
    if analysis_type=='by_ncells':
        grouper.append('n_cells')
    if test_type is not None:
        grouper.append('novel')
    print(grouper)
    mean_by_iters = iterdf.groupby(grouper).mean().reset_index()
    return mean_by_iters


def generalization_score_by_iter(mean_df, max_ncells=None, 
                                 metric='heldout_test_score'):
    '''Calculate generalization score for each iteration.
    mean_df: (pd.DataFrame)
        Average val (novel and trained) for each iteration.
    '''
    if max_ncells is None:
        max_ncells = int(mean_df['n_cells'].max())
    byiter_data = mean_df[(mean_df['n_cells']<=max_ncells) 
                      & (mean_df['condition']=='data')].copy()
    # remove unnec. columns
    drop_cols = ['fit_time', 'score_time', 'train_score', 'test_score']
    cols = [c for c in byiter_data.columns if c not in drop_cols]
    # Get NOVEL only
    byiter_novel = byiter_data[byiter_data['novel']==True][cols].copy()
    # Calculate generalization score
    byiter_novel['generalization'] = byiter_data[(byiter_data.novel)][metric].values\
                                        /byiter_data[~(byiter_data.novel)][metric].values
    return byiter_novel


# --------------------------------------------------------------------
# standard plotting 
# --------------------------------------------------------------------

# by_ncells plotting
def plot_score_v_ncells_color_X(finaldf, sample_sizes, ax=None,
                         metric='heldout_test_score', hue_name='visual_area', 
                         palette='viridis', dpi=150, legend=True):
    '''
    Plot test score vs n_cells for each visual area.
    '''
    if ax is None:
        fig, ax = pl.subplots(figsize=(4,4), dpi=dpi)
    sns.lineplot(x='n_cells', y=metric, data=finaldf, ax=ax,
            style='condition', dashes=['', (1,1)], 
            style_order=['data', 'shuffled'],
            err_style='bars', hue=hue_name, palette=palette, ci='sd') 

    if legend:
        ax.legend(bbox_to_anchor=(1,1), loc='upper left', frameon=False)
        ax.set_xticks(sample_sizes)
    else:
        ax.legend_.remove()    

# morph plotting
def plot_pchooseb_by_area(plotd, metric='p_chooseB', area_colors=None, ax=None):
    '''plot neurometric curves, color by visual_area'''
    if area_colors is None:
        visual_areas, area_colors = pplot.set_threecolor_palette()
    if ax is None:
        fig, ax = pl.subplots()
    sns.lineplot(x='morphlevel', y=metric, ax=ax, 
                data=plotd[plotd.morphlevel!=-1], 
                hue='visual_area', err_style='bars', ci='sd', 
                palette=area_colors, err_kws={'lw': 1}, lw=1)
    ax.legend(bbox_to_anchor=(1., 1.1), loc='upper left', frameon=False)
    ax.axhline(y=0.5, ls=':', lw=0.5, color='k')
    ax.set_ylim([0, 1])
    pplot.set_morph_xticks(ax) 
    sns.despine(trim=True, ax=ax)

    return ax

 

def plot_pchooseb_lum_by_area(plotd, metric='p_chooseB', ax=None, color=0.8,
                              visual_areas=['V1', 'Lm', 'Li']):
    '''plot pchooseB for LUMINANCE cond (morphlevel=-1)'''
    if ax is None:
        fig, ax = pl.subplots()

    sns.barplot(x='visual_area', y=metric, ax=ax, 
                data=plotd[plotd.morphlevel==-1], ci='sd', 
                order=visual_areas, color=[color]*3, errwidth=0.5)
    ax.set_title('FF luminance')
    ax.set_aspect(6, anchor='C')
    ax.set_xlabel('')
    ax.tick_params(which='both', axis='x', size=0)
    sns.despine(trim=True, bottom=True, ax=ax)    
    return ax



#--------------------------------------------------------------------
# run main
# --------------------------------------------------------------------

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', 
                      default='/n/coxfs01/2p-data',\
                      help='data root dir [default: /n/coxfs01/2pdata]')

    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', 
                        default='blobs', help="experiment type [default: blobs]")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', 
                        default='traces001', \
                      help="name of traces ID [default: traces001]")
      
    # data filtering 
    choices_c = ('all', 'ROC', 'nstds', None, 'None')
    default_c = 'nstds'
    parser.add_option('-R', '--responsive-test', action='store', 
            dest='responsive_test', 
            default=default_c, type='choice', choices=choices_c,
            help="Responsive test, choices: %s. (default: %s" % (choices_c, default_c))
    parser.add_option('-r', '--responsive-thr', action='store', dest='responsive_thr', 
            default=10, help="response type [default: 10, nstds]")
    parser.add_option('-d', '--response-type', action='store', dest='response_type', 
            default='dff', help="response type [default: dff]")

    choices_e = ('stimulus', 'firsthalf', 'plushalf', 'baseline')
    default_e = 'stimulus'
    parser.add_option('--epoch', action='store', dest='trial_epoch', 
            default=default_e, type='choice', choices=choices_e,
            help="Trial epoch, choices: %s. (default: %s" % (choices_e, default_e))

    # classifier
    parser.add_option('--class-name', action='store', dest='class_name',
            default=None, help='Class name (morphlevel, size, sf, ori, etc.)')
    parser.add_option('--class-values', action='append', dest='class_values',
            default=[], nargs=1, help='Class values (leave empty to get default')
    parser.add_option('--var-name', action='store', dest='variation_name',
            default=None, help='Variation name (transform to keep constant)')
    parser.add_option('--var-values', action='append', dest='variation_values',
            default=[], nargs=1, help='Class values (leave empty to get default), if var_name=morphlevel, set value to -1 (not testd)')

#    parser.add_option('-a', action='store', dest='class_a', 
#            default=0, help="m0 (default: 0 morph)")
#    parser.add_option('-b', action='store', dest='class_b', 
#            default=106, help="m100 (default: 106 morph)")
    parser.add_option('-n', action='store', dest='n_processes', 
            default=1, help="N processes (default: 1)")
    parser.add_option('-N', action='store', dest='n_iterations', 
            default=500, help="N iterations (default: 500)")

    parser.add_option('--verbose', action='store_true', dest='verbose', 
            default=False, help="verbose printage")
    parser.add_option('--new', action='store_true', dest='create_new', 
            default=False, help="re-do decode")
    parser.add_option('-C','--cvalue', action='store', dest='C_value', 
            default=1.0, help="Set None to tune C (default: 1)")
    parser.add_option('--folds', action='store', dest='cv_nfolds', 
            default=5, help="N folds for CV tuning C (default: 5")

    choices_a = ('by_fov', 'split_pupil', 'by_ncells', 'single_cells')
    default_a = 'by_fov'
    parser.add_option('-X','--analysis', action='store', dest='analysis_type', 
            default=default_a, type='choice', choices=choices_a,
            help="Analysis type, choices: %s. (default: %s)" % (choices_a, default_a))

    parser.add_option('-V','--visual-area', action='store', dest='visual_area', 
            default=None, help="(set for by_ncells) Must be None to run all serially")
    parser.add_option('-k','--datakey', action='store', dest='datakey', 
            default=None, help="(set for single_cells) Must be None to run serially")

    parser.add_option('--no-shuffle', action='store_false', dest='do_shuffle', 
            default=True, help="don't do shuffle")

    choices_t = (None, 'None', 'size_single', 'size_subset', 
                'morph', 'morph_single', 'ori_single')
    default_t = None  
    parser.add_option('-T', '--test', action='store', dest='test_type', 
            default=default_t, type='choice', choices=choices_t,
            help="Test type, choices: %s. (default: %s)" % (choices_t, default_t))

    parser.add_option('--ntrain', action='store', dest='n_train_configs', 
            default=4, help="N training sizes to use (default: 4, test 1)")
    parser.add_option('--break', action='store_true', dest='break_correlations', 
            default=False, help="Break noise correlations")
 
    parser.add_option('--incl-repeats', action='store_false', dest='drop_repeats', 
            default=True, help="BY_NCELLS:  Drop repeats before sampling")
    parser.add_option('-S', '--ncells', action='store', dest='n_cells_sample', 
            default=None, 
            help="BY_NCELLS: n cells to sample (None, cycle thru all)")
    parser.add_option('--match-rfs', action='store_true', dest='match_rfs', 
            default=False, help="Only include cells with matched RF sizes")

    parser.add_option('-O', '--overlap', action='store', dest='overlap_thr', 
            default=None, 
            help="BY_NCELLS: Only cells overlapping wth stimulus by >= overlap_thr")
    parser.add_option('--sphere', action='store_true', 
            dest='do_spherical_correction', 
            default=False, help="For RF metrics, use spherically-corrected fits")

    (options, args) = parser.parse_args(options)

    return options



def main(options):
    opts = extract_options(options)
    fov_type = 'zoom2p0x'
    state = 'awake'
    aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    rootdir = opts.rootdir
    create_new = opts.create_new
    verbose=opts.verbose

    # Pick data ------------------------------------ 
    traceid = opts.traceid #'traces001'
    responsive_test = opts.responsive_test #'nstds' # 'nstds' #'ROC' #None
    if responsive_test=='None':
        responsive_test=None
    responsive_thr = float(opts.responsive_thr) \
                            if responsive_test is not None else 0.05 #10
    if responsive_test == 'ROC':
        responsive_thr = 0.05
    trial_epoch = opts.trial_epoch #'plushalf' # 'stimulus'

    # Dataset
    visual_area = opts.visual_area
    datakey = opts.datakey
    experiment = opts.experiment #'blobs'
    drop_repeats = opts.drop_repeats

    # Classifier info ---------------------------------
    n_iterations=int(opts.n_iterations) #100 
    n_processes=int(opts.n_processes) #2
    if n_processes > n_iterations:
        n_processes = n_iterations
    analysis_type=opts.analysis_type
    # CV
    do_shuffle=opts.do_shuffle
    test_split=0.2
    cv_nfolds= int(opts.cv_nfolds) 
    C_value = None if opts.C_value in ['None', None] else float(opts.C_value)
    do_cv = C_value in ['None', None]
    print("Do CV -%s- (C=%s)" % (str(do_cv), str(C_value)))

    # Generalization Test ------------------------------
    test_type = None if opts.test_type in ['None', None] else opts.test_type
    n_train_configs = int(opts.n_train_configs) 
    print("~~~~~~~~~~~~~~")
    print("%s" % test_type)
    print("~~~~~~~~~~~~~~")
    # Pupil -------------------------------------------
    pupil_feature='pupil_fraction'
    pupil_alignment='trial'
    pupil_epoch='stimulus' #'pre'
    pupil_snapshot=391800
    redo_pupil=False
    pupil_framerate=20.
    pupil_quantiles=3.
    equalize_conditions=True
    match_all_configs=True #False #analysis_type=='by_ncells'
    # -------------------------------------------------
    # Alignment 
    iti_pre=1.
    iti_post=1.
    # -------------------------------------------------a

    # RF stuff 
    rf_filter_by=None
    reliable_only = True
    rf_fit_thr = 0.5
    # Retino stuf
    retino_mag_thr = 0.01
    retino_pass_criterion='all'
   
    # do it --------------------------------
    variation_values = None if None in opts.variation_values \
                        or 'None' in opts.variation_values else opts.variation_values
    variation_name = None if opts.variation_name in [None, 'None'] \
                        else opts.variation_name
    class_name = None if opts.class_name in [None, 'None'] \
                        else opts.class_name
    class_values = None if None in opts.class_values \
                        or 'None' in opts.class_values else opts.class_values

    if class_name is None:
        class_name = 'morphlevel' if experiment=='blobs' else 'ori'
    if experiment=='blobs':
        assert class_name in ['morphlevel', 'size']
        class_values = [0, 106] if class_name=='morphlevel' else None
        if test_type is not None: # generalization tests
            variation_name = 'size' if class_name=='morphlevel' else 'morphlevel'
            variation_values = None # these get filled later
            # TODO:  if decoding LUMINANCE, class_name='size':
            # variation_name='morphlevel', AND variation_values=[-1]
    elif experiment=='gratings':
        assert class_name in ['ori', 'sf', 'size', 'speed'], \
                    "Bad class_name for %s: %s" % (experiment, str(class_name))
        class_values = None
        if test_type in [None, 'ori_single']:
            # TODO: not implemented 
            variation_name = None # only works for ORI as class_name 
            variation_values=None 

    balance_configs=True
    break_correlations = opts.break_correlations
    n_cells_sample = None if opts.n_cells_sample in ['None', None] \
                        else int(opts.n_cells_sample)
    drop_repeats = opts.drop_repeats
    match_rfs = opts.match_rfs
    overlap_thr = None if opts.overlap_thr in ['None', None] \
                        else float(opts.overlap_thr)
    response_type=opts.response_type
    do_spherical_correction=opts.do_spherical_correction

    print("EXPERIMENT: %s, values: %s" % (experiment, str(class_values)))

    print("OVERLAP: %s" % str(overlap_thr))

    decoding_analysis(datakey, visual_area, experiment,  
                    analysis_type=analysis_type,
                    traceid=traceid,
                    trial_epoch=trial_epoch,
                    responsive_test=responsive_test, 
                    responsive_thr=responsive_thr,
                    test_type=test_type, 
                    break_correlations=break_correlations,
                    n_cells_sample=n_cells_sample,
                    drop_repeats=drop_repeats,
                    overlap_thr=overlap_thr, match_rfs=match_rfs,
                    response_type=response_type, 
                    do_spherical_correction=do_spherical_correction,
                    C_value=C_value, test_split=test_split, cv_nfolds=cv_nfolds, 
                    class_name=class_name, 
                    class_values=class_values,
                    variation_name=variation_name, 
                    variation_values=variation_values,
                    n_train_configs=n_train_configs, 
                    balance_configs=balance_configs, do_shuffle=do_shuffle,
                    n_iterations=n_iterations, n_processes=n_processes,
                    verbose=verbose) 
    return


if __name__ == '__main__':
    main(sys.argv[1:])


