#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 2 16:20:01 2019

@author: julianarhee
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use('agg')
import datetime
import os
import cv2
import glob
import h5py
import sys
import optparse
import copy
import json
import traceback
import time
#import shutil

import pylab as pl
from collections import Counter
import seaborn as sns
import _pickle as pkl
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns
import tifffile as tf


import multiprocessing as mp
import itertools

from skimage import exposure
from matplotlib import patches

from scipy import stats as spstats
from scipy.interpolate import interp1d
import scipy.optimize as spopt

import analyze2p.utils as hutils
import analyze2p.aggregate_datasets as aggr
import analyze2p.gratings.utils as utils
import analyze2p.plotting as pplot
import analyze2p.extraction.traces as traceutils
#%%
# #############################################################################
# Metric calculations:
# #############################################################################
def group_configs(group, response_type):
    config = group['config'].unique()[0]
    group.index = np.arange(0, group.shape[0])

    return pd.DataFrame(data={'%s' % config: group[response_type]})

def get_ASI(response_vector, thetas):
    if np.max(thetas) > np.pi:
        thetas = [np.deg2rad(th) for th in thetas]
    asi_vec = [theta_resp * np.exp((2j*2*np.pi*theta_val) / (2*np.pi))\
                    for theta_resp, theta_val in zip(response_vector, thetas)] 
    asi = np.abs(np.sum(asi_vec)) / np.sum(np.abs(response_vector))

    return asi

def get_DSI(response_vector, thetas):
    if np.max(thetas) > np.pi:
        thetas = [np.deg2rad(th) for th in thetas]
    dsi_vec = [theta_resp * np.exp((1j*2*np.pi*theta_val) / (2*np.pi))\
                    for theta_resp, theta_val in zip(response_vector, thetas)] 
    dsi = np.abs( np.sum( dsi_vec ) ) / np.sum(np.abs(response_vector))

    return dsi


def get_circular_variance(response_vector, thetas):
    if np.max(thetas) > np.pi:
        thetas = [np.deg2rad(th) for th in thetas]
    
    asi_vec = [theta_resp*np.exp( (1j*theta_val) ) \
                    for theta_resp, theta_val in zip(response_vector, thetas)]
    circvar_asi = 1-( np.abs(np.sum( asi_vec )) / np.sum(np.abs(response_vector)) )
   
    dsi_vec = [theta_resp*np.exp( (1j*theta_val*2) ) \
                    for theta_resp, theta_val in zip(response_vector, thetas)]
    circvar_dsi = 1 - ( np.abs(np.sum( dsi_vec ))/np.sum(np.abs(response_vector)) )
    
    return circvar_asi, circvar_dsi

#%%

# #############################################################################
# Data loading, saving, formatting
# #############################################################################
def get_run_name(datakey, traceid='traces001', verbose=False, 
                rootdir='/n/coxfs01/2p-data'):
    '''Gets correct run name for GRATINGS experiment. 
    Prior to 20190512, 'gratings' might actually be RF experiments.
    '''
    run_name=None
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    sessiondir = os.path.join(rootdir, animalid, session) 
    found_dirs = glob.glob(os.path.join(sessiondir, 'FOV%i_*' % fovnum, 
                    'combined_gratings*', 'traces', '%s*' % traceid, 
                    'data_arrays', 'np_subtracted.npz'))
    if 20190405 < int(session) < 20190511:
        # Ignore sessions where gratings was RFs? 
        # TODO: fix so that we still load, but only if true gratings
        return None
    
    try:
        assert len(found_dirs) == 1, \
                "ERROR: [%s, %s] >1 experiment found, no COMBINED!" % (datakey, traceid)
        extracted_dir = found_dirs[0]
    except AssertionError as e:
        if verbose:
            print(e)
        return None
    except Exception as e:
        traceback.print_exc()
        return None
    
    run_name = os.path.split(extracted_dir.split('/traces')[0])[-1]
        
    return run_name



#def get_stimulus_configs(animalid, session, fov, run_name, rootdir='/n/coxfs01/2p-data'):
#    # Get stimulus configs
#    stiminfo_fpath = os.path.join(rootdir, animalid, session, fov, run_name, 'paradigm', 'stimulus_configs.json')
#    with open(stiminfo_fpath, 'r') as f:
#        stiminfo = json.load(f)
#    sdf = pd.DataFrame(stiminfo).T
#    
#    return sdf
#    

def load_tuning_results(datakey='', run_name='gratings', traceid='traces001',
                        fit_desc='', traceid_dir=None, return_missing=False,
                        rootdir='/n/coxfs01/2p-data', verbose=False):

    if verbose:
        print("... loading existing fits")

    bootresults=None; fitparams=None;
    if traceid_dir is None:
        search_name = run_name if 'combined_' in run_name else 'combined_%s_' % run_name
        session, animalid, fovnum = hutils.split_datakey_str(datakey)
        fitdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovnum, 
                                        '%s*' % search_name,
                                        'traces/%s*' % traceid, 'tuning*', fit_desc))
    else:
        fitdir = glob.glob(os.path.join(traceid_dir, 'tuning*', fit_desc))

    if len(fitdir)>0:
        assert len(fitdir)==1, "More than 1 osi results found: %s" % str(fitdir)  
        results_fpath = os.path.join(fitdir[0], 'fitresults.pkl')
        params_fpath = os.path.join(fitdir[0], 'fitparams.json')
        
        if os.path.exists(results_fpath):
            with open(results_fpath, 'rb') as f:
                bootresults = pkl.load(f, encoding='latin1')
            with open(params_fpath, 'r') as f:
                fitparams = json.load(f)

        missing_ = os.path.exists(results_fpath) is False

        if fitparams is not None and 'nonori_configs' not in fitparams.keys():
            sdf = aggr.get_stimuli(datakey, 'gratings')
            nonori_params = get_non_ori_params(sdf)
            if 'nonori_params' in fitparams.keys():
                fitparams.pop('nonori_params')
            fitparams.update({'nonori_configs': nonori_params})
            with open(params_fpath, 'w') as f:
                json.dump(fitparams, f, indent=4, sort_keys=True)
    else:
        missing_ = True
 
    if return_missing:
        return bootresults, fitparams, missing_ 
    else:
        return bootresults, fitparams


def save_tuning_results(bootresults, fitparams):
    fitdir = fitparams['directory']
    results_fpath = os.path.join(fitdir, 'fitresults.pkl')
    params_fpath = os.path.join(fitdir, 'fitparams.json')
    #bootdata_fpath = os.path.join(fitdir, 'tuning_bootstrap_data.pkl')
    
    with open(results_fpath, 'wb') as f:
        pkl.dump(bootresults, f, protocol=2) #pkl.HIGHEST_PROTOCOL)
   
    # Save params:
    with open(params_fpath, 'w') as f:
        json.dump(fitparams, f, indent=4, sort_keys=True)

    print("Saved!")
    return fitdir

#%%
# #############################################################################
# Fitting functions:
# #############################################################################

def get_init_params(response_vector):
    theta_pref = response_vector.idxmax()
    theta_null = (theta_pref + 180) % 360.
    r_pref = response_vector.loc[theta_pref]
    r_null = response_vector.loc[theta_null]
    sigma = np.mean(np.diff([response_vector.index.tolist()]))
    non_prefs = [t for t in response_vector.index.tolist() if t not in [theta_pref, theta_null]]
    r_offset = np.mean([response_vector.loc[t] for t in non_prefs])
    return r_pref, r_null, theta_pref, sigma, r_offset


def angdir180(x):
    '''wraps anguar diff values to interval 0, 180'''
    return min(np.abs([x, x-360, x+360]))

def double_gaussian( x, c1, c2, mu, sigma, C ):
    #(c1, c2, mu, sigma) = params
    x1vals = np.array([angdir180(xi - mu) for xi in x])
    x2vals = np.array([angdir180(xi - mu - 180 ) for xi in x])
    res =   C + c1 * np.exp( -(x1vals**2.0) / (2.0 * sigma**2.0) )             + c2 * np.exp( -(x2vals**2.0) / (2.0 * sigma**2.0) )

#    res =   C + c1 * np.exp( - ((x - mu) % 360.)**2.0 / (2.0 * sigma**2.0) ) \
#            + c2 * np.exp( - ((x + 180 - mu) % 360.)**2.0 / (2.0 * sigma**2.0) )

#        res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) \
#                #+ c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

def evaluate_fit_params(x, y, popt):
    fitr = double_gaussian( x, *popt)
        
    # Get residual sum of squares 
    residuals = y - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return fitr
    
def fit_params(x, y, init_params=[0, 0, 0, 0, 0], 
                    bounds=[np.inf, np.inf, np.inf, np.inf, np.inf]):

    roi_fit=None; fitr=None;
    try:
        popt, pcov = spopt.curve_fit(double_gaussian, x, y, p0=init_params, maxfev=1000, bounds=bounds)
        fitr = double_gaussian( x, *popt)
        assert pcov.max() != np.inf
        success = True
    except Exception as e:
        success = False
        pcov =None
        popt = None
        
#    # Get residual sum of squares 
#    residuals = y - fitr
#    ss_res = np.sum(residuals**2)
#    ss_tot = np.sum((y - np.mean(y))**2)
#    r2 = 1 - (ss_res / ss_tot)
        
    roi_fit = {'pcov': pcov,
               'popt': popt,
               #'fit_y': fitr,
               #'r2': r2,
                 #'x': x,
                 #'y': y,
               #'init': init_params,
               'success': success}

        
    return roi_fit, fitr


def interp_values(response_vector, n_intervals=3, as_series=False, 
                    wrap_value=None, wrap=True):
    '''Interpolate for fine-grained sampling of data for fits'''
    resps_interp = []
    rvectors = copy.copy(response_vector)
    if wrap_value is None and wrap is True:
        wrap_value = response_vector[0]
    if wrap:
        rvectors = np.append(response_vector, wrap_value)

    for orix, rvec in enumerate(rvectors[0:-1]):
        if rvec == rvectors[-2]:
            resps_interp.extend(np.linspace(rvec, rvectors[orix+1], endpoint=True, num=n_intervals+1))
        else:
            resps_interp.extend(np.linspace(rvec, rvectors[orix+1], endpoint=False, num=n_intervals))    
    
    if as_series:
        return pd.Series(resps_interp, name=response_vector.name)
    else:
        return resps_interp


def do_fit(responsedf, n_intervals_interp=3, check_offset=False):
    '''
    responsedf = Series
        index : tested_oris
        values : mean value at tested ori
    (prev called:  fit_ori_tuning())

    ''' 
    response_pref=None; response_null=None; theta_pref=None; 
    sigma=None; response_offset=None;
    asi_t=None; dsi_t=None;
    circvar_asi_t=None; circvar_dsi_t=None;

    if check_offset and responsedf.min()<0:
        responsedf -= responsedf.min()

    # interpolate values
    tested_oris = responsedf.index.tolist()
    oris_interp = interp_values(tested_oris, 
                    n_intervals=n_intervals_interp, wrap_value=360)
    resps_interp = interp_values(responsedf, 
                    n_intervals=n_intervals_interp, wrap_value=responsedf[0])
    # initialize params
    init_params = get_init_params(responsedf)
    r_pref, r_null, theta_pref, sigma, r_offset = init_params
    init_bounds = ([0, 0, -np.inf, sigma/2., -r_pref], \
                   [3*r_pref, 3*r_pref, np.inf, np.inf, r_pref])
    # do fit 
    rfit, fitv = fit_params(oris_interp, resps_interp, 
                                init_params, bounds=init_bounds) 
    # Calculate metrics
    if fitv is not None: 
        # Normalize range, 0 to 1 to calculate ASI/DSI
        rvec = fitv.copy()
        response_vector = hutils.convert_range(rvec, 
                                    oldmin=rvec.min(), oldmax=rvec.max(), 
                                    newmin=0, newmax=1)
    if rfit['success']:
         #asi_t = get_ASI(fitv[0:], oris_interp[0:])
         #dsi_t = get_DSI(fitv[0:], oris_interp[0:])
         asi_t = get_ASI(response_vector, oris_interp)
         dsi_t = get_DSI(response_vector, oris_interp)
         circvar_asi_t, circvar_dsi_t = get_circular_variance(
                                        fitv, oris_interp[0:])
         response_pref, response_null, theta_pref, sigma, response_offset = rfit['popt']
         #r2 = rfit['r2']
        
    fitres = pd.Series({'response_pref': response_pref,
                        'response_null': response_null,
                        'theta_pref': theta_pref,
                        'sigma': sigma,
                        'response_offset': response_offset,
                        'asi': asi_t,
                        'dsi': dsi_t, 
                        'circvar_asi': circvar_asi_t,
                        'circvar_dsi': circvar_dsi_t
                        },  name=responsedf.name
    )
    
    return fitres


def fit_from_params( fitres, tested_oris, n_intervals_interp=3):
    '''Given param values, calculate gaussian'''
    params_list = ['response_pref', 'response_null', 'theta_pref', 'sigma', 'response_offset']
    popt = tuple(fitres.loc[params_list].values)
    oris_interp = interp_values(tested_oris, 
                    n_intervals=n_intervals_interp, wrap_value=360)
    try:
        fitr = double_gaussian( oris_interp, *popt)
    except Exception as e:
        fitr = [None for _ in range(len(oris_interp))]
        
    return pd.Series(fitr, name=fitres.name)

def get_r2(fitr, y):
        
    # Get residual sum of squares 
    residuals = y - fitr
    ss_res = np.sum(residuals**2, axis=0)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

#%%
import math
from functools import partial
from contextlib import contextmanager

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
import multiprocessing.pool
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_


def bootstrap_fit_by_config(roi_df, sdf=None, statdf=None, 
                            params=None, create_new=False):
#                            response_type='dff',
#                            n_bootstrap_iters=1000, n_resamples=20, 
#                            n_intervals_interp=3, min_cfgs_above=2, min_nframes_above=10):
    '''
    Inputs
        roi_df (pd.DataFrame) 
            Response metrics for all trials for 1 cell.
        sdf (pd.DataFrame) 
            Stimulus config key / info
        statdf (pd.DataFrame) 
            Dataframe from n_stds responsive test (N frames above baseline for each stimulus config (avg over trials))
        response_type (str)
            Response metric to use for calculating tuning.
        min_cfgs_above (int)
            Number of stimulus configs that should pass the "responsive" threshold for cell to count as responsive
        min_nframes_above (int)
            Min num frames (from statdf) that counts as a cell to be reposnsive for a given stimulus config
            
    Returns
        List of dicts from mp for each roi's results
        {configkey:   
            'data': {
                'responses': dataframe, rows=trials and cols=configs,
                'tested_values': tested values we are interpreting over and fitting
            }
            'stimulus_configs': list of stimulus configs corresponding to current set
            'fits': {
                'xv': interpreted x-values,
                'yv': dataframe of each boot iter of response values
                'fitv': dataframe of each boot iter's fit responses
            }
            'results': dataframe of all fit params 
        }

        Non-responsive cells are {roi: None}
        Responsive cells w/o tuning fits will return with 'fits' and 'results' as None 
    '''
    #roi = roi_df.index[0]    
    roi = int(np.unique([r for r in roi_df.columns if r not in ['config', 'trial']])) 

    if not create_new:
        try:
            oridata = load_roi_fits(roi, params)
        except Exception as e:
            create_new=True

    if create_new:
        response_type=params['response_type']
        n_bootstrap_iters=params['n_bootstrap_iters']
        #n_resamples=params['n_resamples']
        n_intervals_interp=params['n_intervals_interp']
        min_cfgs_above=params['min_cfgs_above']
        min_nframes_above=params['min_nframes_above']

        filter_configs = statdf is not None

        constant_params = ['aspect', 'luminance', 'position', 'stimtype', 'direction', 'xpos', 'ypos']
        params_list = [c for c in sdf.columns if c not in constant_params]
        stimdf = sdf[params_list]
        tested_oris = sorted(sdf['ori'].unique())

        # Each configset is a set of the 8 tested oris at 
        # a specific combination of non-ori params
        start_t = time.time()
        oridata = {}
        for ckey, cfg_ in stimdf.groupby(['sf', 'size', 'speed']):
            currcfgs = cfg_.sort_values(by='ori').index.tolist()
            # Idenfify cells that are responsive before fitting cond.
            if filter_configs:
                responsive = len(np.where(statdf[roi].loc[currcfgs]>=min_nframes_above)[0])>=min_cfgs_above
                if not responsive:
                    continue

            # Get all trials of current cfgs (cols=configs, rows=trial reps)
            rdf = roi_df[roi_df['config'].isin(currcfgs)][[roi, 'config']]
            responses_df = pd.concat([pd.Series(g[roi], name=c)\
                            .reset_index(drop=True)\
                            for c, g in rdf.groupby(['config'])], axis=1).dropna(axis=0)
            datadict = {'responses': responses_df, 'tested_values': tested_oris}
            
            # Bootstrap distN of responses (rand w replacement):
            n_resamples = responses_df.shape[0]
            bootdf_tmp = pd.concat([responses_df.sample(n_resamples, replace=True)\
                            .mean(axis=0) \
                            for ni in range(n_bootstrap_iters)], axis=1)
            bootdf_tmp.index = [sdf['ori'][c] for c in bootdf_tmp.index]
            #bootdf = np.abs((bootdf_tmp - bootdf_tmp.mean())) 
            #if bootdf_tmp.min()<0:
            #    bootdf = (bootdf_tmp-bootdf_tmp.min()) #- (bootdf_tmp-bootdf_tmp.mean()).min()
            bootdf = bootdf_tmp.copy()

            # Find init params for tuning fits and set fit constraints:
            fitp = bootdf.apply(do_fit, args=[n_intervals_interp, True], axis=0) 
            fitdict=None;
            if fitp.dropna().shape[0]>0:
                # Get fits
                fitv = fitp.apply(fit_from_params, args=[tested_oris], axis=0)
                # Interpolate boot responses 
                yvs = bootdf.apply(interp_values, args=[n_intervals_interp, True], 
                                    axis=0) #, result_type='reduce')
                xvs = interp_values(tested_oris, n_intervals=n_intervals_interp, 
                                    wrap_value=360)
                fitdict = {'xv': xvs, 'yv': yvs, 
                           'fitv': fitv, 'n_intervals_interp': n_intervals_interp} 
                # Create dataframe of all fit params
                fitp = fitp.T
                fitp['r2'] = get_r2(fitv, yvs) # Calculate coeff of deterim
                fitp['cell'] = [roi for _ in range(n_bootstrap_iters)]
            else:
                fitp = None 
            oridata[ckey] = {'results': fitp, 
                             'fits': fitdict,
                             'data': datadict,
                             'stimulus_configs': currcfgs}        
            end_t = time.time() - start_t
        save_roi_fits(roi, oridata, params)    
        print("--> (cell {0:d}, {1:d} cfgs) Elapsed: {2:.2f}sec"\
                        .format(roi, len(oridata.keys()), end_t))

    return {roi: oridata}

def save_roi_fits(roi, oridata, fitparams):
    #print(fitparams)
    roi_outfile = os.path.join(fitparams['directory'], 'roi-fits', 
                                'files', 'roi%03d.pkl' % int(roi+1))
    #print(roi_outfile)

    with open(roi_outfile, 'wb') as f:
        pkl.dump(oridata, f, protocol=2)

    return 

def load_roi_fits(roi, fitparams):
    roi_outfile = os.path.join(fitparams['directory'], 'roi-fits', \
                                'files', 'roi%03d.pkl' % int(roi+1))
    with open(roi_outfile, 'rb') as f:
        oridata = pkl.load(f, encoding='latin1')

    return oridata


def bootstrap_osi_mp(rdf_list, sdf, statdf=None, params=None, n_processes=1, create_new=False):
    #### Define multiprocessing worker
    terminating = mp.Event()    
    def worker(iter_list, sdf, statdf, params, create_new, out_q):
        bootr = {}        
        for roi_df in iter_list: 
            roi_results = bootstrap_fit_by_config(roi_df, sdf=sdf, statdf=statdf, params=params, create_new=create_new)
            bootr.update(roi_results)
        out_q.put(bootr)
        
    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(rdf_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=worker,
                        args=(rdf_list[chunksize * i:chunksize * (i + 1)],
                        sdf, statdf, params, create_new, out_q)) 
            #print(os.getpid())
            print(p.name, p._identity)

            procs.append(p)
            
            p.start()

        # Collect all results into single results dict. 
        results = {}
        for i in range(n_processes):
            results.update(out_q.get(99999))
        # Wait for all worker processes to finish
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
    finally:
        for p in procs:
            p.join()

    print("BOOTSTRAP ANALYSIS COMPLETE.")
    return results


# ###################################################################

def pool_bootstrap(rdf_list, sdf, statdf=None, params=None, 
                    n_processes=1, create_new=False):
#
    bootresults = {}
    results = None
    terminating = mp.Event()        
    #pool = mp.get_context("spawn")
    pool = mp.Pool(initializer=initializer, 
                                        initargs=(terminating, ), processes=n_processes)
    try:
        results = pool.map_async(partial(bootstrap_fit_by_config, 
                                sdf=sdf, statdf=statdf, params=params, 
                                create_new=create_new), rdf_list).get() #999999999)

    except KeyboardInterrupt:
        print("**interupt")
        pool.terminate()
        print("***Terminating!")
    finally:
        pool.close()
        pool.join()
  
    if results is not None: 
        #for roi, bootres in results.items():
        #    print(roi, len([k for k, v in bootres.items() if v['fits'] is not None]))

        #print(results)
        bootresults = {k: v for d in results for k, v in d.items()}
        
    return bootresults
    

##%%

def plot_tuning_fits(roi, bootr, df_traces, labels, sdf, trace_type='dff'):
    '''
    Plot raw and fit for 1 roi. Plots 
        a) PSTH of traces, 
        b) linear tuning curve + fit, 
        c) polar plot + fit.
    '''
    responses_df = bootr['data']['responses']
    curr_cfgs = responses_df.columns.tolist()
    fit_success = bootr['fits'] is not None
    
    fig = pl.figure(figsize=(9, 6))
    fig.patch.set_alpha(1)
    nr=2; nc=8;
    
    # Plot original data - PSTH
    fig, ax = plot_psth_roi(roi, df_traces, labels, curr_cfgs, sdf, 
                            trace_type=trace_type,
                            fig=fig, nr=nr, nc=nc, s_row=0)
    ymin = np.min([0, ax.get_ylim()[0]])
    ax.set_ylim([ymin, ax.get_ylim()[1]])
    
    # Plot original data - tuning curves
    curr_oris = np.array([sdf['ori'][c] for c in curr_cfgs])
    sz = np.mean([sdf['size'][c] for c in curr_cfgs])
    sf = np.mean([sdf['sf'][c] for c in curr_cfgs])
    sp = np.mean([sdf['speed'][c] for c in curr_cfgs])
   
    # Correct mean responses to match fitting processing 
    # curr_resps = responses_df.mean()
    mean_responses = responses_df.mean(axis=0)
    #curr_resps = np.abs(mean_responses - mean_responses.mean())
    curr_resps = (mean_responses - mean_responses.min()) #- (mean_responses-mean_responses.mean()).min()
    curr_sems = responses_df.sem(axis=0)
    fig, ax1 = tuning_curve_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
                                     response_type=trace_type,
                                     fig=fig, nr=nr, nc=nc, s_row=1, colspan=5,
                                     marker='o', markersize=5, lw=0)

    fig, ax2 = polar_plot_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
                                     response_type=trace_type,
                                     fig=fig, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)

    if fit_success:
        # Plot bootstrap fits
        oris_interp = bootr['fits']['xv']
        # Mean + sem across iterations
        resps_fit = bootr['fits']['fitv'].mean(axis=1)
        resps_fit_sem = spstats.sem(bootr['fits']['fitv'].values, axis=1) 
        n_intervals_interp = bootr['fits']['n_intervals_interp']
         
        fig, ax1 = tuning_curve_roi(oris_interp[0:-n_intervals_interp], 
                                     resps_fit[0:-n_intervals_interp], 
                                     curr_sems=resps_fit_sem[0:-n_intervals_interp], 
                                     response_type=trace_type,color='cornflowerblue',
                                     markersize=0, lw=1, marker=None,
                                     fig=fig, ax=ax1, nr=nr, nc=nc, s_row=1, colspan=5)

        fig, ax2 = polar_plot_roi(oris_interp, resps_fit, 
                                     curr_sems=resps_fit_sem, 
                                     response_type=trace_type, color='cornflowerblue',
                                     fig=fig, ax=ax2, nr=nr, nc=nc, s_row=1, 
                                     s_col=6, colspan=2) 
        r2_avg = bootr['results']['r2'].mean()
        ax1.plot(0, 0, alpha=0, label='avg r2=%.2f' % r2_avg)
        ax1.legend() #loc='upper left')
        #ax1.text(0, ax1.get_ylim()[-1]*0.75, 'r2=%.2f' % r2_avg, fontsize=6)
    else:
        ax1.plot(0, 0, alpha=0, label='no fit')
        ax1.legend(loc='upper left')
        #ax1.text(0, ax1.get_ylim()[-1]*0.75, 'no fit', fontsize=10)

#    ymin = np.min([0, ax1.get_ylim()[0]])
#    ax1.set_ylim([ymin,  ax1.get_ylim()[1]])
#    ax1.set_yticks([ymin, ax1.get_ylim()[1]])
#    ax1.set_yticklabels([round(ymin, 2), round( ax1.get_ylim()[1], 2)])
    sns.despine(trim=True, offset=4, ax=ax1)
    ax1.set_xticks(curr_oris)
    ax1.set_xticklabels(curr_oris)
    pl.subplots_adjust(hspace=0.5, right=0.95, bottom=0.2) 

    stimkey = 'sf-%.1f-sz-%i-speed-%i' % (sf, sz, sp)
    fig.suptitle('rid %i (sf %.1f, sz %i, speed %i)' % (roi, sf, sz, sp), fontsize=12)

    return fig, stimkey
    
#%%
def get_params_dict(response_type='dff', trial_epoch='stimulus',
                   n_bootstrap_iters=1000, n_intervals_interp=3,
                   responsive_test='nstds', responsive_thr=10, n_stds=2.5,
                   min_cfgs_above=1, min_nframes_above=10, nonori_params=None):
 
    fitparams = {
        'trial_epoch': trial_epoch,
        'response_type': response_type,
        'responsive_test': responsive_test,
        'responsive_thr': responsive_thr \
                    if responsive_test is not None else None,
        'n_stds': n_stds if responsive_test=='nstds' else None,
        'n_bootstrap_iters': n_bootstrap_iters,
        'n_intervals_interp': n_intervals_interp,
        'min_cfgs_above': min_cfgs_above,
        'min_nframes_above': min_nframes_above 
    }
    if nonori_params is not None:
        fitparams.update({'nonori_params': nonori_params})

    return fitparams

def get_tuning(datakey, run_name, return_iters=False,
               traceid='traces001', roi_list=None, statdf=None,
               response_type='dff', trial_epoch='stimulus',
               n_bootstrap_iters=1000, n_intervals_interp=3,
               make_plots=True, responsive_test='nstds', responsive_thr=10, n_stds=2.5,
               create_new=False, redo_cell=False, fmt='svg',
               rootdir='/n/coxfs01/2p-data', n_processes=1,
               min_cfgs_above=1, min_nframes_above=10, verbose=True):
    '''
    Returns:

    bootresults (dict)
        keys: roi ids
        values: dicts, where keys=stim param combos (tuple: sf, size, speed)

        (sf, size, speed): 
              {'data': 
                    {'responses': dataframe, cols=configs, rows=trials,
                     'tested_values': orientation values
                     },
              'results': pd.DataFrame, fit values for all bootstrap iterations, 
              'fits': 
                    {'fitv': pd.DataFrame, values fit for corresponding yvs's,
                     'yvs': bootstrapped values for each iteration,
                     'xvs': x-values that fitv is fit over,
                     'n_intervals_interp': int, # of intervals to interp
                     },
               'stimulus_configs': config names for current stimulus cond
               }
    params (dict)
        Params for fitting procedure
    
    missing_datasets (list)
        List of datakeys with no aggregate traces
    ''' 
    missing_data=[] 
    bootresults=None; fitparams=None;
    # Get tuning dirs
    fitdir, fit_desc = utils.create_fit_dir(datakey, run_name, 
                                traceid=traceid,
                                response_type=response_type, 
                                responsive_test=responsive_test, 
                                n_stds=n_stds, responsive_thr=responsive_thr, 
                                n_bootstrap_iters=n_bootstrap_iters, 
                                rootdir=rootdir)
    traceid_dir =  fitdir.split('/tuning/')[0] 
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'np_subtracted.npz')
    print(traceid_dir)

    do_fits = False
    if not os.path.exists(data_fpath):
        missing_data.append(datakey)
        return None, None, None

#        # Realign traces
#        print("*****corrected offset unfound, running now*****")
#        print("| %s | %s | %s" % (datakey, run_name, traceid))
#
#        aggregate_experiment_runs(animalid, session, fov, 'gratings', traceid=traceid)
#        print("*****corrected offsets!*****")
#        do_fits=True                        
#        create_new=True

    if create_new is False:
        try:
            bootresults, fitparams = load_tuning_results(
                                        datakey=datakey,
                                        traceid_dir=traceid_dir,
                                        fit_desc=fit_desc, verbose=verbose)
            assert fitparams is not None, "None returned"
            assert 'nonori_params' in fitparams.keys(), "Wrong results"
        except Exception as e:
            #traceback.print_exc()
            do_fits = True
    else:
        do_fits=True
    
    data_id = '%s\n%s' % ('|'.join([datakey, run_name, traceid]), fit_desc)

    # Do fits
    # --------------------------------------------------------------
    if do_fits:
        print("Loading data and doing fits")
        if redo_cell:
            print("Refitting ALL cells")
        else:
            print("Only refitting what we need")
        # Select only responsive cells:
        roi_list=None; statdf=None;
        if responsive_test is not None:
            roi_list, nrois_total, roistats = aggr.get_responsive_cells(
                                            datakey, run=run_name, 
                                            traceid=traceid, 
                                            response_type=response_type, 
                                            responsive_test=responsive_test, 
                                            responsive_thr=responsive_thr, 
                                            return_stats=True,
                                            n_stds=n_stds, rootdir=rootdir)
            if responsive_test == 'nstds':
                statdf = roistats['nframes_above']
            else:
                #roi_list = [r for r, res in rstats.items() if res['pval'] < responsive_thr]
                #print("%s -- not implemented for gratings..." % responsive_test)
                statdf = None
   
        print("*%s* -- %i of %i cells responsive" % (responsive_test, len(roi_list), nrois_total))
 
        # Load raw data, calculate metrics 
        raw_traces, labels, sdf, run_info = traceutils.load_dataset(data_fpath, 
                                                trace_type='corrected')
        dff_traces, metrics = aggr.process_traces(raw_traces, labels, 
                                    trace_type='dff', 
                                    response_type=response_type, 
                                    trial_epoch=trial_epoch)
        if roi_list is None:
            roi_list = raw_traces.columns.tolist()

        # Check existing
        fit_basedir = os.path.join(fitdir, 'roi-fits')

        if redo_cell:
            existing_dirs = [os.path.join(fitdir, fd) \
                                for fd in ['files', 'figures']]
            for fdir in existing_dirs:
                if os.path.exists(fdir):
                    shutil.rmtree(fdir) 
        else:
            existing = glob.glob(os.path.join(fit_basedir, 'files', '*.pkl'))
            all_rois = copy.copy(roi_list)
            roi_list = [r for r in all_rois if \
                        os.path.join(fit_basedir, 'files', 'roi%05d.pkl' % int(r+1)) \
                        not in existing]
        # Get cells to fit 
        print("... Fitting %i rois (n=%i procs):" \
                        % (len(roi_list), n_processes))
        if 'trial' not in metrics.columns:
            metrics['trial'] = metrics.index.tolist() 
        roidf_list = [metrics[[roi, 'config', 'trial']] for roi in roi_list]

        # Create dirs
        if not os.path.exists(os.path.join(fit_basedir, 'files')):
            os.makedirs(os.path.join(fit_basedir, 'files'))
        if not os.path.exists(os.path.join(fit_basedir, 'figures')):
            os.makedirs(os.path.join(fit_basedir, 'figures'))

        # Get fit params
        min_cfgs_above = min_cfgs_above if statdf is not None else None
        min_nframes_above = min_nframes_above if statdf is not None else None
        nonori_params = get_non_ori_params(sdf)
        fitparams = get_params_dict(response_type=response_type, 
                                    trial_epoch=trial_epoch,
                                    n_bootstrap_iters=int(n_bootstrap_iters), 
                                    n_intervals_interp=int(n_intervals_interp),
                                    responsive_test=responsive_test, 
                                    responsive_thr=responsive_thr, n_stds=n_stds,
                                    min_cfgs_above=min_cfgs_above, 
                                    min_nframes_above=min_nframes_above,
                                    nonori_params=nonori_params)
        fitparams.update({'directory': fitdir})
        
        # Do boostrap 
        start_t = time.time()
        bootresults = pool_bootstrap(roidf_list, sdf, statdf=statdf, 
                                    params=fitparams, n_processes=n_processes,
                                    create_new=redo_cell)
        #bootresults = bootstrap_osi_mp(roidf_list, sdf, statdf=statdf, 
        #                            params=fitparams, n_processes=n_processes)
        end_t = time.time() - start_t
        print("Multiple processes: {0:.2f}sec".format(end_t)) 

        # Save results
        save_tuning_results(bootresults, fitparams)
        passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
        print("... %i of %i cells fit at least 1 tuning curve." \
                                    % (len(passrois), len(roi_list)))         
        # Plot 
        if make_plots:
            print("Plotting fits")
            for ri, roi in enumerate(passrois):
                if ri%10==0:
                    print("... plotting %i of %i rois" % (int(ri+1), len(passrois)))
                for stimparam, bootr in bootresults[roi].items():
                    fig, stimkey = plot_tuning_fits(roi, bootr, dff_traces, labels, 
                                                    sdf, trace_type='dff')
                    pplot.label_figure(fig, data_id)
                    pl.savefig(os.path.join(fitdir, 'roi-fits', 'figures', \
                                'roi%05d__%s.%s' % (int(roi+1), stimkey, fmt)))
                    pl.close()

    return bootresults, fitparams, missing_data


def evaluate_tuning(datakey, run_name, traceid='traces001', fit_desc='', gof_thr=0.66,
                   create_new=False, rootdir='/n/coxfs01/2p-data', 
                   plot_metrics=True, verbose=True):

    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    data_id = '%s\n%s' % ('|'.join([datakey, run_name]), fit_desc)
    bootresults, fitparams = load_tuning_results(datakey, run_name=run_name, 
                                    fit_desc=fit_desc, rootdir=rootdir, verbose=verbose)
    # Evaluate metric fits
    fitdir = fitparams['directory']
    if not os.path.exists(os.path.join(fitdir, 'evaluation', 'gof-rois')):
        os.makedirs(os.path.join(fitdir, 'evaluation', 'gof-rois'))
    else:
        for f in os.listdir(os.path.join(fitdir, 'evaluation', 'gof-rois')):
            os.remove(os.path.join(fitdir, 'evaluation', 'gof-rois', f))

    # Calculate evaluation metrics over iters
    evaldf = aggregate_evaluation(bootresults, fitparams)
    if evaldf is None:
        return None, None

    # Get only the best (not just passable) cells 
    rmetrics, rmetrics_all_cfgs = get_good_fits(bootresults, fitparams, gof_thr=gof_thr)
    if rmetrics is None:
        print("Nothing to do here, all rois suck!")
        return None, None

    # Filter out ones that don't pass thr.
    plotresults = dict((k, bootresults[k]) for k in rmetrics['cell'].unique())
    # Get boot results for best config for top N cells 
    passdf = evaldf[evaldf['cell'].isin(rmetrics['cell'].unique())].copy()
    df_ = pd.concat([g for (c, stim), g in passdf.groupby(['cell', 'stimulus'])
                    if stim==str(rmetrics[rmetrics['cell']==c]['stimulus'].values[0])])
    # Do some plotting
    if (plot_metrics or create_new):
        print("Plotting bootstrap results ...")
        # Plot all metrics
        plot_bootstrapped_params(df_, fitparams, data_id=data_id)
        # Plot ASI/DSI for top cells
        plot_top_asi_and_dsi(df_, fitparams, topn=5, data_id=data_id) 
        # Plot polar tuning    
        polar_plots_all_configs(rmetrics_all_cfgs, plotresults, 
                            fitparams, gof_thr=gof_thr, data_id=data_id)
        print("*** done! ***")
        if plot_metrics:   
            print("Plotting evaluation metrics...")
            # Visualize fit metrics for each roi's stimulus config
            for (roi, stim), g in rmetrics_all_cfgs.groupby(['cell', 'stimulus']):
                #for skey in g.index.tolist():
                bootr = plotresults[roi][stim]
                stimkey = 'sf-%.1f-sz-%i-speed-%i' % stim #skey
                fig = plot_evaluation_results(roi, bootr, fitparams, 
                            param_str=stimkey)
                pplot.label_figure(fig, data_id)
                fig.suptitle('roi %i (%s)' % (int(roi+1), stimkey))
                pl.savefig(os.path.join(fitdir, 'evaluation', 'gof-rois', \
                        'roi%05d__%s.png' % (int(roi+1), stimkey)))
                pl.close()
        
    return rmetrics, rmetrics_all_cfgs 
    

#%%
# #############################################################################
# EVALUATION:
# #############################################################################

def aggregate_evaluation(bootresults, fitparams): #, gof_thr=0.66):
    '''
    From all bootstrap iterations for all ROIs, calculate combined metrics for
    evaluating the fits. Appends combined metric to EACH iteration entry.
    Call get_good_fits() to get avg across iters.
    ''' 
    aggr_bootdf=None
    niters = fitparams['n_bootstrap_iters']
    interp = fitparams['n_intervals_interp'] > 1
    
    passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
    boot_ = []
    for roi in passrois:
        for stimparam, bootr in bootresults[roi].items():
            if bootr['fits'] is None:
                #print("%s: no fit" % str(stimparam))
                continue
            r2comb, gof, fitr = evaluate_fits(bootr, interp=interp)
            if np.isnan(gof): # or gof < gof_thr:
                #print("%s: bad fit" % str(stimparam))
                continue
            
            rfdf = bootr['results']
            rfdf['r2comb'] = [r2comb for _ in range(niters)]
            rfdf['gof'] = [gof for _ in range(niters)]
            rfdf['stimulus'] = str(stimparam)
            #tmpd = pd.DataFrame(rfdf.mean(axis=0)).T #, index=[roi])
            #stimkey = 'sf-%.2f-sz-%i-sp-%i' % stimparam
            #tmpd['stimconfig'] = str(stimparam)            
            boot_.append(rfdf)       
    if len(boot_)>0:
        aggr_bootdf = pd.concat(boot_, axis=0)
        print("... aggregating (%i of %i attempted fit min. 1 curve)." \
                % (len(aggr_bootdf['cell'].unique()), len(passrois)))
       
    return aggr_bootdf
  
          
def evaluate_fits(bootr, interp=False):
    '''
    From all bootstrap iterations, calculate combined metrics for
    evaluating the fits. 
    ''' 

    # Average fit parameters aross boot iters
    params = [c for c in bootr['results'].columns if 'stim' not in c]
    avg_metrics = average_metrics_across_iters(bootr['results'][params])
    
    orig_ = bootr['data']['responses'].mean(axis=0)
    #orig_data = np.abs(orig_ - np.mean(orig_)) 
    orig_data = (orig_ - orig_.min()) #- (orig_ - orig_.mean()).min()

    # Get combined r2 between original and avg-fit
    if interp:
        origr = interp_values(orig_data)
        thetas = bootr['fits']['xv']
    else:
        origr = orig_data #bootr['data']['responses'].mean(axis=0).values
        thetas = bootr['data']['tested_values']
    #thetas = bootr['fits']['xv'][0:-1]
    #origr = interp_values(origr, n_intervals=3, wrap_value=origr[0])[0:-1]
    
    cpopt = tuple(avg_metrics[['response_pref', 'response_null', 'theta_pref', 'sigma', 'response_offset']].values[0])
    fitr = double_gaussian( thetas, *cpopt)
    r2_comb, _ = coeff_determination(origr, fitr)
    
    # Get Goodness-of-fit
    iqr = spstats.iqr(bootr['results']['r2'])
    gfit = np.mean(bootr['results']['r2']) * (1-iqr) * np.sqrt(r2_comb)
    
    return r2_comb, gfit, fitr
    

def get_good_fits(bootresults, fitparams, gof_thr=0.66, verbose=True):
    '''
    Calculate combined metrics. Return those that pass (if gof_thr)
    ''' 
    rmetrics=None; rmetrics_by_cfg=None; 
    niters = fitparams['n_bootstrap_iters']
    interp = fitparams['n_intervals_interp']>1

    passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
           
    metrics_by_config = []
    roidfs=[]
    goodrois = []
    for roi in passrois:
        
        fitresults = []
        stimkeys = []
        for stimparam, bootr in bootresults[roi].items():
            if bootr['fits'] is None:
                #print("%s: no fit" % str(stimparam))
                continue

            # Evaluate current fits from bootstrapped results
            r2comb, gof, fitr = evaluate_fits(bootr, interp=interp)
            if np.isnan(gof) or (gof_thr is not None and (gof < gof_thr)):
                #print("%s: bad fit" % str(stimparam))
                continue
            
            rfdf = bootr['results'] # All 1000 iterations
            rfdf['r2comb'] = [r2comb for _ in range(niters)] # add combined R2 val
            rfdf['gof'] = [gof for _ in range(niters)] # add GoF metric
            rfdf['stimulus'] = [stimparam]*len(rfdf)

             # Average current roi, current condition results
            tmpd = average_metrics_across_iters(rfdf) 
            stimkey = 'sf-%.2f-sz-%i-sp-%i' % stimparam     
            fitresults.append(tmpd)
            stimkeys.append(stimkey)
        
        if len(fitresults) > 0:
            roif = pd.concat(fitresults, axis=0).reset_index(drop=True)
            roif.index = stimkeys
            gof =  roif.mean()['gof']
            
            if (gof_thr is not None) and (gof >= gof_thr): #0.66:
                goodrois.append(roi) # This is just for repoorting
           
            if gof_thr is not None:
                best_cond_df = pd.DataFrame(roif.sort_values(by='r2comb').iloc[-1]).T
            else:
                best_cond_df = pd.DataFrame(roif.sort_values(by='gof').iloc[-1]).T
            # Select config w/ strongest response 
            roidfs.append(best_cond_df)
            # But also save all results
            metrics_by_config.append(roif)
   
    if len(roidfs) > 0: 
        rmetrics = pd.concat(roidfs, axis=0)
        # rename indices to rois
        new_ixs = [int(i) for i in rmetrics['cell'].values]
        rmetrics.index = new_ixs
        rmetrics_by_cfg = pd.concat(metrics_by_config, axis=0)
      
        if verbose: 
            if gof_thr is not None: 
                print("... %i (of %i) fitable cells pass GoF thr %.2f" \
                            % (len(goodrois), len(passrois), gof_thr))
            else:
                print("... %i (of %i) fitable cells (no GoF thr)" \
                                % (rmetrics.shape[0], len(passrois)))

    return rmetrics, rmetrics_by_cfg


def plot_evaluation_results(roi, bootr, fitparams, param_str='curr params'):
    #%  Look at residuals        
    n_intervals_interp = fitparams['n_intervals_interp']
    
    residuals = bootr['fits']['yv'].subtract(bootr['fits']['fitv'])
    mean_residuals = residuals.mean(axis=0)
    
    fig, axes = pl.subplots(2,3, figsize=(10,6))
    xv = bootr['fits']['xv'][0:-1]
    for fiter in bootr['fits']['yv'].columns:
        yv = bootr['fits']['yv'][fiter][0:-1]
        fitv = bootr['fits']['fitv'][fiter][0:-1]
        axes[0,0].plot(xv, fitv, lw=0.5)
        axes[0,1].scatter(yv, residuals[fiter][0:-1], alpha=0.5)
        
    # ax0: adjust ticks/labels
    ax = axes[0,0]
    ax.set_xticks(xv[0::n_intervals_interp])
    ax.set_xticklabels([int(x) for x in xv[0::n_intervals_interp]])
    ax.set_xlabel('thetas')
    ax.set_ylabel('fit')
    ymin = np.min([0, ax.get_ylim()[0]])
    ax.set_ylim([ymin, ax.get_ylim()[1]])
    ax.tick_params(labelsize=8)
    ax.set_ylabel(fitparams['response_type'])
    sns.despine(ax=ax, trim=True, offset=2)
    ax.set_title('fit iters (n=%i)' % fitparams['n_bootstrap_iters'])

    # ax1: adjust ticks/labels
    ax = axes[0,1]
    ax.axhline(y=0, linestyle=':', color='k')
    ax.set_ylabel('residuals')
    ax.set_xlabel('fitted value')
    ax.tick_params(labelsize=8)
    
    ax = axes[0,2]
    ax.hist(mean_residuals, bins=20, color='k', alpha=0.5)
    ax.set_xlabel('mean residuals')
    ax.set_ylabel('counts of iters')
    ax.set_xticks([ax.get_xlim()[0], ax.get_xlim()[-1]])
    ax.tick_params(labelsize=8)
    sns.despine(ax=ax, trim=True, offset=2)

    # Compare the original direction tuning curve with the fitted curve derived 
    # using the average of each fitting parameter (across 100 iterations)

    # Get residual sum of squares and compare ORIG and AVG FIT:
    thetas = xv[0::n_intervals_interp] #[0:-1]
    orig0 = bootr['data']['responses'].mean(axis=0)
    #origr = np.abs(orig0 - orig0.mean())
    origr = (orig0-orig0.min()) #- (orig0-orig0.mean()).min()
    fitv = bootr['fits']['fitv'].mean(axis=1)[0::n_intervals_interp][0:-1]
 
    r2comb, gof, fitr = evaluate_fits(bootr, interp=False)        
    ax = axes[1,0]
    ax.plot(thetas, origr, 'k', label='orig')
    ax.plot(thetas, fitv, 'b:', label='fit_avgboot')
    ax.plot(thetas, fitr, 'r:', label='fit_avgparams')
    ax.set_title('r2-comb: %.2f' % r2comb)
    ax.legend()
    ax.set_xticks(xv[0::n_intervals_interp])
    ax.set_xticklabels([int(x) for x in xv[0::n_intervals_interp]], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlabel('thetas')
    ax.set_ylabel(fitparams['response_type'])
    sns.despine(ax=ax, trim=True, offset=2)
    
    
    # Compare distN of preferred orientation across all iters
    fparams = bootr['results'] #get_params_all_iters(fitdata['results_by_iter'][roi])
    ax = axes[1,1]
    ax.hist(fparams['theta_pref'], alpha=0.5, bins=20, color='k', )
    #ax.set_xlim([fparams['theta_pref'].min(), fparams['theta_pref'].max()])
    ax.set_xticks(np.linspace(int(np.floor(fparams['theta_pref'].min())), int(np.ceil(fparams['theta_pref'].max())), num=5))
    sns.despine(ax=ax, trim=True, offset=2)
    ax.set_xlabel('preferred theta')
    ax.set_ylabel('counts of iters')
    ax.tick_params(labelsize=8)
    ax.axvline(x=fparams['theta_pref'].median(), linestyle=':', color='k')
    
    # Look at calculated ASI/DSIs across iters:
    ax = axes[1,2]
    ax.scatter(bootr['results']['asi'], bootr['results']['dsi'], c='k', marker='+', alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel('ASI');
    ax.set_ylim([0, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel('DSI');
    ax.set_aspect('equal')
    
    pl.subplots_adjust(hspace=.5, wspace=.5)
    
    fig.suptitle('rid %i, %s' % (int(roi), param_str))

    return fig

def average_metrics_across_iters(fitdf):
    means = {}
    roi = int(fitdf['cell'].unique()[0])
    #print("COLS:", fitdf.columns)
    for param in fitdf.columns:
        if 'stim' in param:
            meanval = [fitdf[param].values[0]]
        elif 'theta' in param:
            # meanval = np.rad2deg(spstats.circmean(np.deg2rad(fitdf[param] % 360.)))
            # Use Median, since could have double-peaks
            #meanval = fitdf[param].median() 
            cnts, bns = np.histogram(fitdf[param] % 360., 
                            bins=np.linspace(0, 360., 50))
            meanval = float(bns[np.where(cnts==max(cnts))[0][0]])
        else:
            meanval = fitdf[param].mean()
        means[param] = meanval
    return pd.DataFrame(means, index=[roi])



def coeff_determination(origr, fitr):
    residuals = origr - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((origr - np.mean(origr))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2, residuals

#%%

# In[21]:

# #############################################################################
# Plotting functions:
# #############################################################################

def cleanup_axes(axes_list, which_axis='y'):    
    for ax in axes_list: 
        if which_axis=='y':
            # get the yticklabels from the axis and set visibility to False
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
        elif which_axis=='x':
            # get the xticklabels from the axis and set visibility to False
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)

import matplotlib.patches as patches

def plot_psth_roi(roi, raw_traces, labels, curr_cfgs, sdf,  trace_type='dff', fig=None, nr=1, nc=1, s_row=0, colspan=1):
    if fig is None:
        fig = pl.figure()

    pl.figure(fig.number)
    #print('plotting roi %i' % roi) 
    # ---------------------------------------------------------------------
    #% plot raw traces:
    mean_traces, std_traces, tpoints = traceutils.get_mean_and_std_traces(roi, 
                                        raw_traces, labels, curr_cfgs, sdf)
    
    stim_on_frame = labels['stim_on_frame'].unique()[0]
    nframes_on = labels['nframes_on'].unique()[0]
    
    ymin = np.nanmin((mean_traces - std_traces )) #.min()
    ymax = np.nanmax((mean_traces + std_traces )) #.max()
    for icfg in range(len(curr_cfgs)):
        ax = pl.subplot2grid((nr, nc), (s_row, icfg), colspan=colspan)
        ax.plot(tpoints, mean_traces[icfg, :], color='k')
        ax.set_xticks([tpoints[stim_on_frame], 
                        round(tpoints[stim_on_frame+nframes_on], 1)])
        ax.set_xticklabels(['', round(tpoints[stim_on_frame+nframes_on], 1)])
        ax.set_ylim([ymin, ymax])
        if icfg > 0:
            ax.set_yticks([]); ax.set_yticklabels([]);
            ax.set_xticks([]); ax.set_xticklabels([]);
            sns.despine(ax=ax, offset=4, trim=True, left=True, bottom=True)
        else:
            ax.set_ylabel(trace_type); ax.set_xlabel('time (s)');
            sns.despine(ax=ax, offset=4, trim=True)
        sem_plus = np.array(mean_traces[icfg,:]) + np.array(std_traces[icfg,:])
        sem_minus = np.array(mean_traces[icfg,:]) - np.array(std_traces[icfg,:])
        ax.fill_between(tpoints, sem_plus, y2=sem_minus, alpha=0.5, color='k')

        ax.axvspan(tpoints[stim_on_frame], tpoints[stim_on_frame+nframes_on], 
                    alpha=0.3, facecolor='gray', edgecolor='none')

    return fig, ax


def tuning_curve_roi(curr_oris, curr_resps, curr_sems=None, 
                    response_type='dff', fig=None, ax=None, 
                    nr=1, nc=1, colspan=1, s_row=0, s_col=0, color='k',
                    marker='o', lw=1, markersize=5):
    '''
    Plot linear tuning curve (prev. called plot_tuning_curve_roi)
    '''
    if fig is None:
        fig = pl.figure()
    
    pl.figure(fig.number)
        
    # Plot tuning curve:
    if ax is None:
        ax = pl.subplot2grid((nr, nc), (s_row, s_col), colspan=colspan)
    ax.plot(curr_oris, curr_resps, color=color, marker=marker, 
                                        markersize=markersize, lw=lw)
    if curr_sems is not None:
        ax.errorbar(curr_oris, curr_resps, yerr=curr_sems, fmt='none', ecolor=color)
    ax.set_xticks(curr_oris)
    ax.set_xticklabels(curr_oris)
    ax.set_ylabel(response_type)
    #ax.set_title('(sz %i, sf %.2f)' % (best_cfg_params['size'], best_cfg_params['sf']), fontsize=8)
    #sns.despine(trim=True, offset=4, ax=ax)
    
    return fig, ax

def polar_plot_roi(curr_oris, curr_resps, curr_sems=None, response_type='dff',
                          fig=None, ax=None, nr=1, nc=1, colspan=1, s_row=0, s_col=0, 
                          color='k', linestyle='-', label=None, alpha=1.0):
    '''
    Plot direction tuning as polar plot. 
    (Prev. called plot_tuning_polar_roi())

    '''
    if fig is None:
        fig = pl.figure()
    
    pl.figure(fig.number)
    
    # Plot polar graph:
    if ax is None:
        ax = pl.subplot2grid((nr,nc), (s_row, s_col), colspan=colspan, polar=True)
    thetas = np.array([np.deg2rad(c) for c in curr_oris])
    radii = curr_resps.copy()
    thetas = np.append(thetas, np.deg2rad(curr_oris[0]))  # append first value so plot line connects back to start
    radii = np.append(radii, curr_resps[0]) # append first value so plot line connects back to start
    ax.plot(thetas, radii, '-', color=color, label=label, linestyle=linestyle, alpha=alpha)
    ax.set_theta_zero_location("N")
    ax.set_yticks([curr_resps.min(), curr_resps.max()])
    ax.set_yticklabels(['', round(curr_resps.max(), 1)])

    
    return fig, ax

# Summary plotting:
def sort_by_selectivity(df, topn=10):
    
    top_asi = df.groupby(['cell']).mean().sort_values(['asi'], ascending=False)
    top_dsi = df.groupby(['cell']).mean().sort_values(['dsi'], ascending=False)
    #top_r2 = df.groupby(['cell']).mean().sort_values(['r2'], ascending=False)
    
    top_asi_cells = top_asi.index.tolist()[0:topn]
    top_dsi_cells = top_dsi.index.tolist()[0:topn]
    #top_r2_cells = top_r2.index.tolist()[0:topn]

    top10_asi = [roi for rank, roi in enumerate(top_asi.index.tolist()) if rank < topn]
    top10_dsi = [roi for rank, roi in enumerate(top_dsi.index.tolist()) if rank < topn]
    
    df.loc[:, 'top_asi'] = np.array([ roi if roi in top10_asi else -10 for roi in df['cell']])
    df.loc[:, 'top_dsi'] = np.array([ roi if roi in top10_dsi else -10 for roi in df['cell']])
    
    #% # Convert to str for plotting:        
    df.loc[:, 'top_asi'] = [str(s) for s in df['top_asi'].values]
    df.loc[:, 'top_dsi'] = [str(s) for s in df['top_dsi'].values]

 
    return df, top_asi_cells, top_dsi_cells


def compare_topn_selective(plotd, color_by='ASI', palette='cubehelix'):
    
    hue = 'top_asi' if color_by in ['ASI', 'asi'] else 'top_dsi'

    g = sns.pairplot(plotd[plotd[hue]!=str(-10)], hue=hue, vars=['asi', 'dsi'], 
                 palette='cubehelix', size=2, markers='+')
       
    g.set(ylim=(0, 1))
    g.set(xlim=(0, 1))
    
    g = g.add_legend(bbox_to_anchor=(1.0,0.2))
#    for li, lh in enumerate(g._legend.legendHandles): 
#        if not all([round(l, 1)==0.5 for l in lh.get_facecolor()[0][0:3]]): 
#            lh.set_alpha(1)
#            lh._sizes = [20] 
#        
    pl.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)
    
    #g.set(xlim=(0,1), ylim=(0,1))
    #g.set(xticks=[0, 1])
    #g.set(yticks=[0, 1])
    #sns.despine(trim=True)
            
    #cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
    #cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
        
    return g.fig

    
def plot_bootstrapped_params(fitdf, fitparams, 
                            data_id='METADATA', return_fig=False):
    '''sns.pairplot() with all metrics,
    Plots bootstrapped values'''
 
    roi_fitdir = fitparams['directory']
    metrics_to_plot = ['asi', 'dsi', 'response_pref', 'theta_pref', 'r2comb', 'gof']
   
    #% PLOT -- plot ALL fits:
    if 'ASI_cv' in fitdf.columns.tolist():
        fitdf['ASI_cv'] = [1-f for f in fitdf['ASI_cv'].values] # Revert to make 1 = very tuned, 0 = not tuned
        metrics_to_plot.append('ASI_cv')
    if 'DSI_cv' in fitdf.columns.tolist():
        fitdf['DSI_cv'] = [1-f for f in fitdf['DSI_cv'].values] # Revert to make 1 = very tuned, 0 = not tuned
        metrics_to_plot.append('DSI_cv')

    #g = sns.PairGrid(fitdf, hue='cell', vars=metrics_to_plot) #=['asi', 'dsi', 'r2comb'])
    plotdf = fitdf.groupby(['cell', 'stimulus']).mean().reset_index()
    g = sns.pairplot(plotdf, height=2, aspect=1, hue='cell', vars=metrics_to_plot)
    g.fig.patch.set_alpha(1) 
    #g = g.map_offdiag(pl.scatter, marker='o',  alpha=0.5, s=1) 
    #g = g.map_diag(pl.hist, normed=True) #histtype="step",  
       
    if plotdf.shape[0] < 10:
        g = g.add_legend(bbox_to_anchor=(1.01,.5)) 
    pl.subplots_adjust(left=0.1, right=0.85)

    pplot.label_figure(g.fig, data_id) 
    #nrois_fit = len(fitdf['cell'].unique())
    #nrois_thr = len(strong_fits)
    n_bootstrap_iters = fitparams['n_bootstrap_iters']
    n_intervals_interp = fitparams['n_intervals_interp']
    
    figname = 'compare-bootstrapped-metrics' 
    pl.savefig(os.path.join(roi_fitdir, '%s.svg' % figname))
    pl.close()
    print("... plotted: %s" % figname)
    
    if return_fig:
        return g.fig
    else:
        return None
 
def plot_top_asi_and_dsi(fitdf, fitparams, topn=10, 
                            data_id='METADATA', return_figs=False):   
    '''Plot ASI/DSI distn (Bootstrapped) for top N cells'''

    # Sort cells by ASi and DSi    
    df, top_asi_cells, top_dsi_cells = sort_by_selectivity(fitdf, topn=topn)
    if df is None:
        return
 
    #% Set color palettes:
    palette = sns.color_palette('cubehelix', len(top_asi_cells))
    main_alpha = 0.8
    sub_alpha = 0.01
    asi_colordict = dict(( str(roi), palette[i]) for i, roi in enumerate(top_asi_cells))
    for k, v in asi_colordict.items():
        asi_colordict[k] = (v[0], v[1], v[2], main_alpha)
        
    dsi_colordict = dict(( str(roi), palette[i]) for i, roi in enumerate(top_dsi_cells))
    for k, v in dsi_colordict.items():
        dsi_colordict[k] = (v[0], v[1], v[2], main_alpha)
          
    asi_colordict.update({ str(-10): (0.8, 0.8, 0.8, sub_alpha)})
    dsi_colordict.update({ str(-10): (0.8, 0.8, 0.8, sub_alpha)})
            
    #% PLOT by ASI:
    roi_fitdir = fitparams['directory']
    n_bootstrap_iters = fitparams['n_bootstrap_iters']
    nrois_fit = len(fitdf['cell'].unique())
    nrois_pass = len(df['cell'].unique())
    
    color_by = 'asi'
    palette = asi_colordict if color_by=='asi' else dsi_colordict
            
    fig1 = compare_topn_selective(df, color_by=color_by, palette=palette)
    pplot.label_figure(fig1, data_id)
    
    figname = 'sort-by-%s_top%i' % (color_by, topn)
    fig1.savefig(os.path.join(roi_fitdir, '%s.svg' % figname))
    print("... plotted: %s" % figname)
    if not return_figs:
        pl.close(fig1)
    
    #% Color by DSI:
    color_by = 'dsi'
    palette = asi_colordict if color_by=='asi' else dsi_colordict

    fig2 = compare_topn_selective(df, color_by=color_by, palette=palette)
    pplot.label_figure(fig2, data_id)
    figname = 'sort-by-%s_top%i' % (color_by, topn)
    fig2.savefig(os.path.join(roi_fitdir, '%s.svg' % figname))
    print("... plotted: %s" % figname)

    if not return_figs:
        pl.close(fig2)

    if return_figs:
        return fig1, fig2


#% Evaluation -- plotting
def compare_all_metrics_for_good_fits(fitdf, good_fits=None):
    
    if good_fits is not None:
        df = fitdf[fitdf['cell'].isin(good_fits)]
    else:
        df = fitdf.copy()
    df['cell'] = df['cell'].astype(str)
    
    g = sns.PairGrid(df, hue='cell', vars=[c for c in fitdf.columns.tolist() if c != 'cell'], palette='cubehelix')
    g.fig.patch.set_alpha(1)
    g = g.map_offdiag(pl.scatter, marker='o', s=5, alpha=0.7)
    g = g.map_diag(pl.hist, normed=True, alpha=0.5) #histtype="step",  
    
    #g.set(ylim=(0, 1))
    #g.set(xlim=(0, 1))
    #g.set(aspect='equal')
    
    return g.fig
    
#%%

def get_non_ori_params(sdf):

    cfgs = list(itertools.product(*[sdf['sf'].unique(), sdf['size'].unique(), sdf['speed'].unique()]))

    return cfgs

def polar_plots_all_configs(rmetrics_all_cfgs, bootresults, fitparams, gof_thr=0.66,
                        return_fig=False, data_id='META'):
    '''Plot raw + fit tuning curve on polar plot for all successfully fit configs''' 
    n_intervals_interp = fitparams['n_intervals_interp']
    # Set figure grid
    n_rois_pass = len(bootresults.keys())
    nr = int(np.ceil(np.sqrt(n_rois_pass))) + 1 if n_rois_pass>=4 else 1 
    nc = int(np.ceil(float(n_rois_pass) / nr)) 
    # add extra row for legend
    nr += 1
    # Set colormap 
    cfgs = [tuple(c) for c in fitparams['nonori_configs']]
    colors = sns.color_palette(palette='cubehelix', n_colors=8) #len(cfgs))  
    # Plot
    fig, axes = pl.subplots(nc, nr, figsize=(nc*1.5,nr*2), 
                            subplot_kw=dict(polar=True), dpi=150) 
    for ax, (roi, g) in zip(axes.flat, rmetrics_all_cfgs.groupby(['cell'])):
        allgofs = []
        for skey in g.index.tolist():
            stimparam = tuple(float(i) for i in skey.split('-')[1::2])
            si = cfgs.index(stimparam) # same color scheme across cells
            # Plot average of bootstrapped data
            bootr = bootresults[roi][stimparam] 
            thetas_interp = bootr['fits']['xv']
            thetas = bootr['fits']['xv'][0::n_intervals_interp] 
            # Get combined tuning across iters for current stim config
            r2comb, gof, fitr = evaluate_fits(bootr, interp=True)
            origr0 = bootr['data']['responses'].mean(axis=0).values
            #origr = np.abs(origr0 - origr0.mean())
            origr = (origr0 - origr0.min()) #- (origr0 - origr0.mean()).min()

            # Plot polar
            origr = np.append(origr, origr[0]) # wrap back around
            polar_plot_roi(thetas, origr, curr_sems=None, response_type='dff',
                        fig=fig, ax=ax, color=colors[si], linestyle='--')
    
            polar_plot_roi(thetas_interp, fitr, curr_sems=None, response_type='dff',
                        fig=fig, ax=ax, color=colors[si], linestyle='-', alpha=0.8,
                        label='gof %.2f\ndff %.2f' % (gof, origr.max()) )
            allgofs.append(gof)
            
        ax.set_title('%i (GoF: %.2f)' % (int(roi), np.mean(allgofs)), fontsize=6, y=1)
        ax.legend(bbox_to_anchor=(0.75, 0), loc='upper right', ncol=1, fontsize=6)
        ax.yaxis.grid(False)
        ax.yaxis.set_ticklabels([])        
        ax.xaxis.grid(True)
        ax.xaxis.set_ticklabels([])
        
    for ax in axes.flat[n_rois_pass:]:
        ax.axis('off')

    pl.subplots_adjust(hspace=0.5, wspace=0.8, left=0.1, 
                                    right=0.95, bottom=0.3, top=0.9) 
    # Custom legend 
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[i], lw=4) \
                                        for i, k in enumerate(cfgs)]
    custom_labels = ['sf-%.2f, sz-%i, speed-%i' % k for i, k in enumerate(cfgs)]
    ax = axes.flat[-1]
    ncol=2 #1 if nc<=3 else 2
    ax.legend(custom_lines, custom_labels, ncol=ncol, fontsize=8,
                bbox_to_anchor=(-0.5, -0.5), loc='upper right')

    pplot.label_figure(fig, data_id)
    pl.savefig(os.path.join(fitparams['directory'], 'evaluation', \
                            'polarplots_thr-%.2f.svg' % (gof_thr)))
     
    print("... plot: polar plots all configs")
    if not return_fig:
        pl.close(fig)
        return 
    else:
        return fig


#%%
# DATA AGGREGATION
def aggregate_ori_fits(CELLS, traceid='traces001', fit_desc=None,
                       response_type='dff', responsive_test='nstds', responsive_thr=10.,
                       n_bootstrap_iters=1000,  verbose=False,
                       return_missing=False, rootdir='/n/coxfs01/2p-data'):
    '''
    assigned_cells:  dataframe w/ assigned cells of dsets that have gratings
    '''
    if fit_desc is None:
        fit_desc = get_fit_desc(response_type=response_type, 
                            responsive_test=responsive_test, 
                            n_stds=n_stds, responsive_thr=responsive_thr, 
                            n_bootstrap_iters=n_bootstrap_iters, 
                            )
    gdata=None
    no_fits=[]; missing_fits=[];
    g_list=[];
    for (va, dk), g in CELLS.groupby(['visual_area', 'datakey']):
        try:
            # Load tuning results
            fitresults, fitparams, is_missing = load_tuning_results(datakey=dk, 
                                            fit_desc=fit_desc, traceid=traceid,
                                            return_missing=True)
            if is_missing or fitresults is None:
                missing_fits.append((va, dk))
                continue
            if len(fitresults.keys())==0:
                no_fits.append((va, dk))
                continue
            # Get OSI results for assigned cells
            rois_ = g['cell'].unique()
            boot_ = dict((k, v) for k, v in fitresults.items() if k in rois_)
        except Exception as e:
            if verbose:
                traceback.print_exc() 
            no_fits.append('%s_%s' % (va, dk))
            continue
        # Aggregate fits
        best_fits, all_fits = get_good_fits(boot_, fitparams, 
                                             gof_thr=None, verbose=verbose)
        if best_fits is None:
            no_fits.append('%s_%s' % (va, dk))
            continue
        if verbose:
            print(va, dk, all_fits.shape)
        all_fits['visual_area'] = va
        all_fits['datakey'] = dk
        g_list.append(all_fits)
    gdata = pd.concat(g_list, axis=0).reset_index(drop=True)
    if verbose:
        print("Datasets with no results found:")
        for s in missing_fits:
            print(s)

    if return_missing:
        return gdata, no_fits, missing_fits
    else:
        return gdata



#%%
def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--datakey', action='store', dest='datakey', default='yyyymmdd_JCxx_fovX', 
                      help='datakey (YYYYMMDD_JCXX_fovX)')

#    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
#                      help='Animal ID')
#    parser.add_option('-S', '--session', action='store', dest='session', default='', 
#                      help='Session (format: YYYYMMDD)')
#
#    # Set specific session/run for current animal:
#    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', 
#                      help="fov name (default: FOV1_zoom2p0x)")
#    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    
    # Responsivity params:
     # Responsivity params:
    choices_resptest = ('ROC','nstds', None)
    default_resptest = None
    
    parser.add_option('-R', '--response-test', type='choice', choices=choices_resptest,
                      dest='responsive_test', default=default_resptest, 
                      help="Stat to get. Valid choices are %s. Default: %s" % (choices_resptest, str(default_resptest)))
    parser.add_option('-r', '--responsive-thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test threshold (default: p<0.05 for responsive_test=ROC)")
    parser.add_option('-s', '--n-stds', action='store', dest='n_stds', default=2.5, 
                      help="[n_stds only] n stds above/below baseline to count frames, if test=nstds (default: 2.5)") 
    parser.add_option('-m', '--min-frames', action='store', dest='min_nframes_above', 
                    default=10, 
                     help="[n_stds only] Min N frames above baseline std (responsive_thr), if responsive_test=nstds (default: 10)")   
    parser.add_option('-c', '--min-configs', action='store', dest='min_cfgs_above', 
                    default=1, 
                     help="[n_stds only] Min N configs in which min-n-frames threshold is met, if responsive_test=nstds (default: 2)")   

    # Tuning params:
    parser.add_option('-b', '--iter', action='store', dest='n_bootstrap_iters', 
                    default=1000, 
                     help="N bootstrap iterations (default: 1000)")
    #parser.add_option('-k', '--samples', action='store', dest='n_resamples', 
    #                default=20, 
    #                  help="N trials to sample w/ replacement (default: 20)")
    parser.add_option('-p', '--interp', action='store', dest='n_intervals_interp', 
                    default=3, 
                      help="N intervals to interp between tested angles (default: 3)")
    
    parser.add_option('-d', '--response-type', action='store', dest='response_type', 
                    default='dff', 
                      help="Trial response measure to use for fits (default: dff)")
    parser.add_option('-n', '--n-processes', action='store', dest='n_processes', 
                    default=1, help="N processes (default: 1)")

    parser.add_option('-G', '--goodness-thr', action='store', dest='goodness_thr', 
                    default=0.66, help="Goodness-of-fit threshold (default: 0.66)")

    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="Create aggregate fit results file")

    parser.add_option('--redo-cell', action='store_true', dest='redo_cell', 
                    default=False, help="Refit all cells")

    parser.add_option('--plots', action='store_true', dest='make_plots', default=False, 
                      help='Flag to plot roi fits')

    parser.add_option('-E', '--epoch', action='store', dest='trial_epoch', 
                    default='stimulus', help='epoch of trial to use for fitting')

    (options, args) = parser.parse_args(options)

    return options

#%%

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC084' 
#session = '20190522' #'20190319'
#fov = 'FOV1_zoom2p0x' 
#run = 'combined_gratings_static'
#traceid = 'traces001' #'traces002'
##trace_type = 'corrected'
#
#response_type = 'dff'
##metric_type = 'dff'
#make_plots = True
#n_bootstrap_iters = 100
#n_intervals_interp = 3
#
#responsive_test = 'ROC'
#responsive_thr = 0.05
#


def fit_and_evaluate(datakey, traceid='traces001', 
                        response_type='dff', trial_epoch='stimulus',
                        responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
                        n_bootstrap_iters=1000, 
                        min_cfgs_above=2, min_nframes_above=10, 
                        n_intervals_interp=3, 
                        goodness_thr = 0.66,
                        n_processes=1, create_new=False, redo_cell=False,
                        rootdir='/n/coxfs01/2p-data',
                        make_plots=True, verbose=True):

    #datakey = '%s_%s_fov%i' % (session, animalid, int(fov.split('_')[0][3:])) 
    run_name = get_run_name(datakey, traceid=traceid, rootdir=rootdir, verbose=verbose) 
    assert run_name is not None, "ERROR: [%s|%s] No gratings run..." % (datakey, traceid)

    # 1. Fit tuning curves
    bootresults, fitparams, missing_data = get_tuning(datakey, run_name, 
                                         traceid=traceid, 
                                         response_type=response_type, 
                                         trial_epoch=trial_epoch,
                                         n_bootstrap_iters=int(n_bootstrap_iters), 
                                         #n_resamples = int(n_resamples),
                                         n_intervals_interp=int(n_intervals_interp),
                                         responsive_test=responsive_test, 
                                         responsive_thr=responsive_thr, n_stds=n_stds,
                                         create_new=create_new, redo_cell=redo_cell,
                                         n_processes=n_processes, 
                                         rootdir=rootdir, verbose=verbose,
                                         min_cfgs_above=min_cfgs_above, 
                                         min_nframes_above=min_nframes_above, 
                                         make_plots=make_plots
    )

    fit_desc = os.path.split(fitparams['directory'])[-1]
    print("----- COMPLETED 1/2: bootstrap tuning! ------")

    # Evaluate fits
    rmetrics, rmetrics_all_cfgs = evaluate_tuning(datakey, run_name, 
                                                traceid=traceid, fit_desc=fit_desc, 
                                                gof_thr=goodness_thr,
                                                create_new=create_new, 
                                                rootdir=rootdir, plot_metrics=make_plots
    ) 
    if rmetrics is None:
        n_goodcells = 0
    else:
        n_goodcells = len(rmetrics.index.tolist())
 
    print("----- COMPLETED 2/2: evaluation (%i good cells)! -----" % n_goodcells)
   
    print("There were %i missing datasets:" % len(missing_data))
    for m in missing_data:
        print('    %s' % m)
 
    return rmetrics_all_cfgs, fitparams, missing_data 
#%%

def main(options):
    opts = extract_options(options)
    rootdir = opts.rootdir
    #animalid = opts.animalid
    #session = opts.session
    #fov = opts.fov
    datakey = opts.datakey

    traceid = opts.traceid
    response_type = opts.response_type
    trial_epoch = opts.trial_epoch

    n_bootstrap_iters = int(opts.n_bootstrap_iters)
    #n_resamples = int(opts.n_resamples)
    n_intervals_interp = int(opts.n_intervals_interp)
    responsive_test = opts.responsive_test
    responsive_thr = float(opts.responsive_thr)
    n_stds = float(opts.n_stds)
    n_processes = int(opts.n_processes)
    min_nframes_above = int(responsive_thr) if responsive_test=='nstds' else int(opts.min_nframes_above)
    min_cfgs_above = int(opts.min_cfgs_above)
    create_new = opts.create_new
    redo_cell = opts.redo_cell

    goodness_thr = float(opts.goodness_thr)
    make_plots = opts.make_plots
 
    rmetrics_all_cfgs, fitparams, missing_data = fit_and_evaluate(
                                             datakey, traceid=traceid,
                                             #animalid, session, fov, traceid=traceid, 
                                             response_type=response_type, 
                                             trial_epoch=trial_epoch,
                                             n_bootstrap_iters=n_bootstrap_iters, 
                                             #n_resamples=n_resamples,
                                             n_intervals_interp=n_intervals_interp,
                                             responsive_test=responsive_test, 
                                             responsive_thr=responsive_thr, 
                                             n_stds=n_stds,
                                             min_nframes_above=min_nframes_above, 
                                             min_cfgs_above=min_cfgs_above,
                                             create_new=create_new, redo_cell=redo_cell,
                                             n_processes=n_processes, 
                                             rootdir=rootdir,
                                             goodness_thr=goodness_thr, make_plots=make_plots)
    print("***** DONE *****")
    
if __name__ == '__main__':
    main(sys.argv[1:])
    


