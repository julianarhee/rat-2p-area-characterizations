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


from inspect import currentframe, getframeinfo
from pandas.core.common import SettingWithCopyError
pd.options.mode.chained_assignment='warn' #'raise' # 'warn'

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
                C_value=None, verbose=False, return_clf=False, return_predictions=False,
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
        print("Unique train: %s (%i)" % (str(np.unique(train_labels)), len(train_labels)))
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
    if verbose:    
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print(skmetrics.classification_report(test_labels, predicted_labels))

    mi_dict = get_mutual_info_metrics(test_labels, predicted_labels)
    iterdict.update(mi_dict) 
    iterdict.update({'heldout_test_score': test_score, 'C': C_value, 'randi': randi})
    
    iterdf = pd.DataFrame(iterdict, index=[inum])

    if return_clf:
        if return_predictions:
            return iterdf, trained_svc, scaler, (predicted_labels, test_labels)
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


def fit_shuffled(zdata, targets, 
                C_value=None, test_split=0.2, cv_nfolds=5, randi=0, 
                verbose=False, return_clf=False,
                class_types=[0, 106], class_name='morph', 
                do_pchoose=False, return_svc=False, inum=0):
    '''
    Shuffle target labels, do fit_svm()
    '''
    iterdf=None; trained_svc=None; trained_scaler=None;

    labels_shuffled = targets['label'].copy().values 
    np.random.shuffle(labels_shuffled)
    targets['label'] = labels_shuffled
    if do_pchoose:
        tmp_=[]
        class_a, class_b = class_types
        iter_shuffled, trained_svc, trained_scaler, (predicted_, true_) = fit_svm(
                                                            zdata, targets, 
                                                            return_clf=True, 
                                                            return_predictions=True,
                                                            test_split=test_split, 
                                                            cv_nfolds=cv_nfolds, 
                                                            C_value=C_value, 
                                                            randi=randi, verbose=verbose,
                                                            inum=inum) 
        # Calculate P(choose B)
        for anchor in class_types: #[class_a, class_b]:
            a_ixs = [i for i, v in enumerate(true_) if v==anchor] 
            p_chooseB = sum([1 if p==class_b else 0 \
                            for p in predicted_[a_ixs]])/float(len(a_ixs))
            iter_shuffled['p_chooseB'] = p_chooseB
            iter_shuffled['%s' % class_name] = anchor
            iter_shuffled['n_samples']=len(a_ixs)
#            iter_shuffled.update({'p_chooseB': p_chooseB, 
#                                  '%s' % class_name: anchor, 
#                                  'n_samples': len(a_ixs)})
#            tmpdf_shuffled = pd.DataFrame(iter_shuffled, index=[i])
            #i+=1
            tmp_.append(tmpdf_shuffled)            
        iterdf = pd.concat(tmp_, axis=0) #.reset_index(drop=True)
    else:
        iterdf = fit_svm(zdata, targets, 
                            C_value=C_value,
                            test_split=test_split, 
                            cv_nfolds=cv_nfolds, 
                            randi=randi, verbose=verbose, inum=inum)
        #iterdf = pd.DataFrame(iterdict, index=[i])
    iterdf['condition'] = 'shuffled'

    if return_clf:
        return iterdf, trained_svc, trained_scaler
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
                  'return_clf': return_clf,
                  'verbose': verbose,
                  'inum': iter_num}
    # Fit
    if return_clf:
        idf_, clf_, _ = fit_svm(zdata, targets, **clf_params)
    else:
        idf_ = fit_svm(zdata, targets, **clf_params)
    idf_['condition'] = 'data'
    i_list.append(idf_)

    # Shuffle labels
    clf_params['C_value'] = idf_['C'].values[0] # Update c, in case we did tuning
    if do_shuffle:
        if return_clf:
            idf_shuffled, _, _ = fit_shuffled(zdata, targets, **clf_params)
        else:
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
    combo_train_parvals = list(itertools.combinations(variation_values, n_train_configs))

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
                      'return_clf': return_clf,
                      'verbose': verbose,
                      'inum': iter_num}
        # FIT
        idf_, trained_svc, trained_scaler = fit_svm(train_data, targets, **clf_params)
        idf_['condition'] = 'data'
        clf_params['C_value'] = idf_['C'].values[0]
        train_list.append(idf_)
        # Shuffle labels
        if do_shuffle:
            idf_shuffled = fit_shuffled(train_data, targets, **clf_params)
            if clf_params['return_clf']:
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
            mi_dict = get_mutual_info_metrics(curr_test_labels, predicted_labels)
            testdict.update(mi_dict) 
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
                      'return_clf': return_clf,
                      'verbose': verbose,
                      'inum': iter_num}
        # FIT
        idf_, trained_svc, trained_scaler = fit_svm(train_data, targets, **clf_params)
        idf_['condition'] = 'data'
        clf_params['C_value'] = idf_['C'].values[0]
        train_list.append(idf_)
        # Shuffle labels
        if do_shuffle:
            idf_shuffled = fit_shuffled(train_data, targets, **clf_params)
            if clf_params['return_clf']:
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
            mi_dict = get_mutual_info_metrics(curr_test_labels, predicted_labels)
            testdict.update(mi_dict) 
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



def fit_svm_mp(neuraldf, sdf, test_type, n_iterations=50, n_processes=1, 
                    break_correlations=False,
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
    def worker(out_q, n_iters, test_type, **kwargs):
        i_list = []        
        for ni in n_iters:
            # Decoding -----------------------------------------------------
            start_t = time.time()
            i_df = select_test(ni, test_type, neuraldf, sdf, 
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
                                iter_list[chunksize * i:chunksize * (i + 1)],
                                test_type),
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
 
    if test_type in ['train_test_size_subset', 'train_test_size_single']:
        kwargs['return_clf'] = True

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
        return None

    return curri


# --------------------------------------------------------------------
def create_results_id(C_value=None,
                    visual_area='varea', trial_epoch='stimulus', 
                    response_type='dff', responsive_test='resp', 
                    break_correlations=False, overlap_thr=None): 
    '''
    test_type: generatlization test name (size_single, size_subset, morph, morph_single)
    trial_epoch: mean val over time period (stimulus, plushalf, baseline) 
    '''
    C_str = 'tuneC' if C_value is None else 'C%.2f' % C_value
    corr_str = 'nocorrs' if break_correlations else 'intact'
    overlap_str = 'noRF' if overlap_thr is None else 'overlap%.2f' % overlap_thr
    #test_str='all' if test_type is None else test_type
    response_str = '%s-%s' % (response_type, responsive_test)
    results_id='%s__%s__%s__%s__%s__%s' \
                % (visual_area, response_str, trial_epoch, overlap_str, C_str, corr_str)
   
    return results_id

def create_aggregate_id(experiment, C_value=None,
                    trial_epoch='stimulus', 
                    response_type='dff', responsive_test='resp', 
                    overlap_thr=None): 
    '''
    test_type: generatlization test name (size_single, size_subset, morph, morph_single)
    trial_epoch: mean val over time period (stimulus, plushalf, baseline) 
    '''
    C_str = 'tuneC' if C_value is None else 'C%.2f' % C_value
    overlap_str = 'noRF' if overlap_thr is None else 'overlap%.2f' % overlap_thr
    #test_str='all' if test_type is None else test_type
    response_str = '%s-%s' % (response_type, responsive_test)
    results_id='%s__%s__%s__%s__%s' \
                % (experiment, response_str, trial_epoch, overlap_str, C_str)
   
    return results_id



def decode_from_fov(datakey, visual_area, experiment, neuraldf,
                    sdf=None, test_type=None, results_id='results',
                    n_iterations=50, n_processes=1, break_correlations=False,
                    traceid='traces001', 
                    rootdir='/n/coxfs01/2p-data', **clf_params): 
    '''
    Fit FOV n_iterations times (multiproc). Save all iterations in dataframe.
    '''
    # Set output dir and file
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                        'FOV%i_*' % fovnum, 'combined_%s_static' % experiment, 
                        'traces', '%s*' % traceid))[0]
    test_str = 'default' if test_type is None else test_type
    curr_dst_dir = os.path.join(traceid_dir, 'decoding_test', test_str)
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
    if clf_params['verbose']:
        print("... BY_FOV [%s] %s, n=%i cells" % (visual_area, datakey, n_cells))
    # Stimulus info
    if sdf is None:
        sdf = aggr.get_stimuli(dk, experiment)
    # Decode
    start_t = time.time()
    iter_results = fit_svm_mp(ndf_z, sdf, test_type, 
                            n_iterations=n_iterations,
                            n_processes=n_processes,
                            break_correlations=break_correlations,
                            **clf_params)
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

    if test_type is not None:
        print(iter_results.groupby(['condition', 'train_transform']).mean())   
        print(iter_results.groupby(['condition', 'train_transform']).count())   
    else:
        print(iter_results.groupby(['condition']).mean())   
    print("@@@@@@@@@ done. %s|%s  @@@@@@@@@@" % (visual_area, datakey))
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


# -----------------------------------------------------------------------
def decoding_analysis(dk, va, experiment,  
                    analysis_type='by_fov',
                    response_type='dff', traceid='traces001',
                    trial_epoch='stimulus',
                    responsive_test='nstds', responsive_thr=10.,
                    overlap_thr=None,
                    test_type=None, 
                    break_correlations=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, 
                    class_name='morphlevel', class_values=None,
                    variation_name='size', variation_values=None,
                    n_train_configs=4, 
                    balance_configs=True, do_shuffle=True,
                    n_iterations=50, n_processes=1,
                    rootdir='/n/coxfs01/2p-data', verbose=False): 

    # Load all the data
    NDATA = aggr.load_responsive_neuraldata(experiment, traceid=traceid,
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
                                overlap_thr=overlap_thr)
    
    print("~~~~~~~~~~~~~~~~ RESULTS ID ~~~~~~~~~~~~~~~~~~~~~")
    print(results_id)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # Classif params
    clf_params={'class_name': class_name,
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

    if analysis_type=='by_fov':
        # -----------------------------------------------------------------------
        # BY_FOV - for each fov, do_decode
        # -----------------------------------------------------------------------
        neuraldf = NDATA[(NDATA.visual_area==va) & (NDATA.datakey==dk)].copy()
        sdf = aggr.get_stimuli(dk, experiment)
        if int(neuraldf.shape[0]-1)==0:
            return None
        decode_from_fov(dk, va, experiment, neuraldf, sdf=sdf, 
                        test_type=test_type, 
                        results_id=results_id,
                        n_iterations=n_iterations, 
                        traceid=traceid,
                        break_correlations=break_correlations,
                        n_processes=n_processes, **clf_params)
        print("--- done by_fov ---")

    return


# --------------------------------------------------------------------
# Aggregate functions
# --------------------------------------------------------------------
def aggregate_iterated_results(experiment, meta, test_type=None,
                      traceid='traces001',
                      trial_epoch='plushalf', responsive_test='nstds', 
                      C_value=1., break_correlations=False, 
                      overlap_thr=None, 
                      rootdir='/n/coxfs01/2p-data'):
    test_str = 'default' if test_type is None else test_type
    iterdf=None
    missing_=[]
    d_list=[]
    for (va, dk), g in meta.groupby(['visual_area', 'datakey']):
        results_id = create_results_id(C_value=C_value, 
                                       visual_area=va,
                                       trial_epoch=trial_epoch,
                                       responsive_test=responsive_test,
                                       break_correlations=break_correlations,
                                       overlap_thr=overlap_thr)
        try:
            session, animalid, fovn = hutils.split_datakey_str(dk)
            results_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                              'FOV%i_*' % fovn,
                              'combined_%s_*' % experiment, 'traces/%s*' % traceid, 
                              'decoding_test', test_str))[0]
            results_fpath = os.path.join(results_dir, '%s.pkl' % results_id)
            assert os.path.exists(results_fpath), 'Not found:\n    %s' % results_fpath
        except Exception as e:
            missing_.append((va, dk))
            #traceback.print_exc()
            continue
        with open(results_fpath, 'rb') as f:
            res = pkl.load(f)
        d_list.append(res)
    if len(d_list)>0:
        iterdf = pd.concat(d_list)
    
    return iterdf, missing_


def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', 
                      default='/n/coxfs01/2p-data',\
                      help='data root dir [default: /n/coxfs01/2pdata]')

    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', 
                        default='blobs', help="experiment type [default: blobs]")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
      
    # data filtering 
    choices_c = ('all', 'ROC', 'nstds', None, 'None')
    default_c = 'nstds'
    parser.add_option('-R', '--responsive_test', action='store', dest='responsive_test', 
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
            help="Trial epoch for input data, choices: %s. (default: %s" % (choices_e, default_e))


    # classifier
    parser.add_option('-a', action='store', dest='class_a', 
            default=0, help="m0 (default: 0 morph)")
    parser.add_option('-b', action='store', dest='class_b', 
            default=106, help="m100 (default: 106 morph)")
    parser.add_option('-n', action='store', dest='n_processes', 
            default=1, help="N processes (default: 1)")
    parser.add_option('-N', action='store', dest='n_iterations', 
            default=100, help="N iterations (default: 100)")

    parser.add_option('-o', action='store', dest='overlap_thr', 
            default=None, help="% overlap between RF and stimulus (default: None)")
    parser.add_option('--verbose', action='store_true', dest='verbose', 
            default=False, help="verbose printage")
    parser.add_option('--new', action='store_true', dest='create_new', 
            default=False, help="re-do decode")
    parser.add_option('-C','--cvalue', action='store', dest='C_value', 
            default=None, help="tune for C (default: None, tunes C)")
    parser.add_option('--folds', action='store', dest='cv_nfolds', 
            default=5, help="N folds for CV tuning C (default: 5")

    choices_a = ('by_fov', 'split_pupil', 'by_ncells', 'single_cells')
    default_a = 'by_fov'
    parser.add_option('-X','--analysis', action='store', dest='analysis_type', 
            default=default_a, type='choice', choices=choices_a,
            help="Analysis type, choices: %s. (default: %s)" % (choices_a, default_a))

    parser.add_option('-V','--visual-area', action='store', dest='visual_area', 
            default=None, help="(set for by_ncells) Must be None to run all serially")
    parser.add_option('-S','--ncells', action='store', dest='ncells', 
            default=None, help="Must be None to run all serially")
    parser.add_option('-k','--datakey', action='store', dest='datakey', 
            default=None, help="(set for single_cells) Must be None to run all serially")

    parser.add_option('--no-shuffle', action='store_false', dest='do_shuffle', 
            default=True, help="don't do shuffle")

    choices_t = (None, 'None', 'size_single', 'size_subset', 'morph', 'morph_single')
    default_t = None  
    parser.add_option('-T', '--test', action='store', dest='test_type', 
            default=default_t, type='choice', choices=choices_t,
            help="Test type, choices: %s. (default: %s)" % (choices_t, default_t))

    parser.add_option('--ntrain', action='store', dest='n_train_configs', 
            default=4, help="N training sizes to use (default: 4, test 1)")

    parser.add_option('--drop-repeats', action='store_true', dest='drop_repeats', 
            default=False, help="Drop repeats (Note: really only relevant for analysis_type=by_ncells)")
    parser.add_option('--break', action='store_true', dest='break_correlations', 
            default=False, help="Break noise correlations")
 
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
    response_type = opts.response_type #'dff'
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

    # Create data ID for labeling figures with data-types
    #response_str = '%s_%s-thr-%.2f' % (response_type, responsive_test, responsive_thr) 
    #data_id = '|'.join([traceid, response_str])
    #print(data_id)
    
    # do it --------------------------------
    variation_name=None
    variation_values=None
    class_name=None
    class_values=None
    if experiment=='blobs':
        class_name = 'morphlevel'
        class_values = [0, 106]
        if test_type is not None: # generalization tests
            variation_name = 'size'
            variation_values = None
    elif experiment=='gratings':
        class_name = 'ori'
        class_values = None
        if test_type is not None:
            # TODO: not implemented 
            variation_name = None
            variation_values=None 
    balance_configs=True
    break_correlations = opts.break_correlations


    visual_areas = ['V1', 'Lm', 'Li']
    if datakey is None:
        sdata, cells0 = aggr.get_aggregate_info(visual_areas=visual_areas, 
                                                return_cells=True)
        if visual_area is not None:
            meta = sdata[(sdata.visual_area==visual_area)
                        & (sdata.experiment==experiment)].copy()
        else:
            meta = sdata[sdata.experiment==experiment].copy()
    
        for (va, dk), g in meta.groupby(['visual_area', 'datakey']):   
            decoding_analysis(dk, va, experiment,  
                            analysis_type=analysis_type,
                            response_type=response_type, traceid=traceid,
                            trial_epoch=trial_epoch,
                            responsive_test=responsive_test, 
                            responsive_thr=responsive_thr,
                            test_type=test_type, 
                            break_correlations=break_correlations,
                            C_value=C_value, test_split=test_split, cv_nfolds=cv_nfolds, 
                            class_name=class_name, 
                            class_values=class_values,
                            variation_name=variation_name, 
                            variation_values=variation_values,
                            n_train_configs=n_train_configs, 
                            balance_configs=balance_configs, do_shuffle=do_shuffle,
                            n_iterations=n_iterations, n_processes=n_processes,
                            verbose=verbose) 

    else:
        decoding_analysis(datakey, visual_area, experiment,  
                        analysis_type=analysis_type,
                        response_type=response_type, traceid=traceid,
                        trial_epoch=trial_epoch,
                        responsive_test=responsive_test, 
                        responsive_thr=responsive_thr,
                        test_type=test_type, 
                        break_correlations=break_correlations,
                        C_value=C_value, test_split=test_split, cv_nfolds=cv_nfolds, 
                        class_name=class_name, class_values=class_values,
                        variation_name=variation_name, variation_values=variation_values,
                        n_train_configs=n_train_configs, 
                        balance_configs=balance_configs, do_shuffle=do_shuffle,
                        n_iterations=n_iterations, n_processes=n_processes,
                        verbose=verbose) 



if __name__ == '__main__':
    main(sys.argv[1:])


