#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''Functions for analyzing gratings data'''

import os
import glob
import json
import copy
import traceback
import _pickle as pkl
import numpy as np
import pandas as pd
import scipy.stats as spstats

from analyze2p import utils as hutils
import analyze2p.extraction.traces as traceutils
import analyze2p.aggregate_datasets as aggr

def get_fit_desc(response_type='dff', responsive_test=None, 
                 responsive_thr=10, n_stds=2.5,
                 n_bootstrap_iters=1000):# , n_resamples=20):
    '''
    Set standardized naming scheme for ori_fit_desc
    '''
    if responsive_test is None:
        fit_desc = 'fit-%s_all-cells_boot-%i' \
                        % (response_type, n_bootstrap_iters) 
    elif responsive_test == 'nstds':
        fit_desc = 'fit-%s_responsive-%s-%.2f-thr%.2f_boot-%i' \
                        % (response_type, responsive_test, n_stds, responsive_thr, n_bootstrap_iters)
    else:
        fit_desc = 'fit-%s_responsive-%s-thr%.2f_boot-%i' \
                        % (response_type, responsive_test, responsive_thr, n_bootstrap_iters)

    return fit_desc


def get_fit_dir(datakey, traceid='traces001', fit_desc=None, verbose=False,
                rootdir='/n/coxfs01/2p-data'):
    '''Find specified results dir for gratings fits'''
    ori_dir=None
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    try:
        ori_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                         'FOV%i_*' % fovnum,
                         'combined_gratings_static', 'traces', '%s*' % traceid,
                         'tuning', fit_desc))
        assert len(ori_dir)==1,"... [%s]: Ambiguous dir" % (datakey)
        ori_dir = ori_dir[0]

    except AssertionError as e:
        ori_dir=None
        if verbose:
            print(e)
            tdir = glob.glob(os.path.join(rootdir, animalid, session, 
                            'FOV%i_*' % fovnum,
                            'combined_gratings_static', 
                            'traces', '%s*' % traceid,
                            'tuning'))[0]
            edirs = os.listdir(tdir)
            print("--- found dirs: ---")
            for e in edirs:
                print(e)
     
    return ori_dir


def create_fit_dir(datakey, run_name='gratings', 
                   traceid='traces001', response_type='dff', n_stds=2.5,
                   responsive_test=None, responsive_thr=0.05,
                   n_bootstrap_iters=1000, 
                   rootdir='/n/coxfs01/2p-data', traceid_dir=None):

    # Get RF dir for current fit type
    search_str = run_name if 'combined' in run_name else 'combined_%s_' % run_name

    fit_desc = get_fit_desc(response_type=response_type, responsive_test=responsive_test, 
                            n_stds=n_stds,
                            responsive_thr=responsive_thr, n_bootstrap_iters=n_bootstrap_iters)
    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    try:
        traceid_dirs = glob.glob(os.path.join(rootdir, animalid, session, 
                             'FOV%i_*' % fovnum, '%s*' % search_str, 
                             'traces', '%s*' % traceid))
        if len(traceid_dirs) > 1:
            print("More than 1 trace ID found:")
            for ti, traceid_dir in enumerate(traceid_dirs):
                print(ti, traceid_dir)
            sel = input("Select IDX of traceid to use: ")
            traceid_dir = traceid_dirs[int(sel)]
        else:
            traceid_dir = traceid_dirs[0]
    except Exception as e:
        print(traceid_dirs)
        print(datakey, search_str, fit_desc)

    osidir = os.path.join(traceid_dir, 'tuning', fit_desc)
    if not os.path.exists(osidir):
        os.makedirs(osidir)

    return osidir, fit_desc


def load_tuning_results(datakey, run_name='gratings', traceid='traces001',
                        fit_desc=None, rootdir='/n/coxfs01/2p-data', verbose=False):
    '''Load results from bootstrap analysis (fitresults.pkl, fitparams.json)'''
    bootresults=None; fitparams=None;
    try: 
        ori_dir = get_fit_dir(datakey, traceid=traceid, fit_desc=fit_desc,
                                verbose=verbose)
        assert ori_dir is not None, "... [%s] No ori_dir" % datakey
        results_fpath = os.path.join(ori_dir, 'fitresults.pkl')
        params_fpath = os.path.join(ori_dir, 'fitparams.json')
        # open 
        with open(results_fpath, 'rb') as f:
            bootresults = pkl.load(f, encoding='latin1')
        with open(params_fpath, 'r') as f:
            fitparams = json.load(f, encoding='latin1')
    except AssertionError as e:
        print(e)
    except Exception as e:
        print("[ERROR]: NO fits %s" % datakey)
        if verbose:
            traceback.print_exc() 
                        
    return bootresults, fitparams


# Fitting
def interp_values(response_vector, n_intervals=3, as_series=False, wrap_value=None, wrap=True):
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

def angdir180(x):
    '''wraps anguar diff values to interval 0, 180'''
    return min(np.abs([x, x-360, x+360]))

def double_gaussian( x, c1, c2, mu, sigma, C ):
    #(c1, c2, mu, sigma) = params
    x1vals = np.array([angdir180(xi - mu) for xi in x])
    x2vals = np.array([angdir180(xi - mu - 180 ) for xi in x])
    res =   C + c1 * np.exp( -(x1vals**2.0) / (2.0 * sigma**2.0) ) + c2 * np.exp( -(x2vals**2.0) / (2.0 * sigma**2.0) )

#    res =   C + c1 * np.exp( - ((x - mu) % 360.)**2.0 / (2.0 * sigma**2.0) ) \
#            + c2 * np.exp( - ((x + 180 - mu) % 360.)**2.0 / (2.0 * sigma**2.0) )

#        res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) \
#                #+ c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res


# Evaluation
def average_metrics_across_iters(fitdf):
    ''' 
    Average bootstrapped params to get an "average" set of params.
    '''
    means = {}
    roi = int(fitdf['cell'].unique()[0])
    #print("COLS:", fitdf.columns)
    for param in fitdf.columns:
        if 'stim' in param:
            meanval = fitdf[param].values[0]
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
       
def evaluate_fits(bootr, interp=False):
    '''
    Create an averaged param set (from bootstrapped iters). 
    Fit tuning curve, calculate R2 and goodness-of-fit.
    '''
    # Average fit parameters aross boot iters
    params = [c for c in bootr['results'].columns if 'stim' not in c]
    avg_metrics = average_metrics_across_iters(bootr['results'][params])

    # Get mean response (deal with offset)    
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

    # Fit to average, evaluate fit   
    cpopt = tuple(avg_metrics[['response_pref', 'response_null', 'theta_pref', 'sigma', 'response_offset']].values[0])
    fitr = double_gaussian( thetas, *cpopt)
    r2_comb, _ = coeff_determination(origr, fitr) 
    # Get Goodness-of-fit
    iqr = spstats.iqr(bootr['results']['r2'])
    gfit = np.mean(bootr['results']['r2']) * (1-iqr) * np.sqrt(r2_comb)
    
    return r2_comb, gfit, fitr
    


def get_good_fits(bootresults, fitparams, gof_thr=0.66, verbose=True):
    '''
    For all cells, evaluate fits.
    bootresults: dict
        All results from bootstrap analysis, by roi (keys). 
    fitparams:  dict
        Fit info
    
    Returns

    best_rfits: pd.DataFrame
        Best fit stimulus param for each cell.

    all_rfits: pd.DatFrame
        All stimulus conditions with fit for each cell.
    '''
    best_rfits=None; all_rfits=None; 
    niters = fitparams['n_bootstrap_iters']
    interp = fitparams['n_intervals_interp']>1
    passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
           
    all_dfs = []; best_dfs=[];
    goodrois = []
    for roi in passrois: 
        fitresults = []
        stimkeys = []
        for stimparam, bootr in bootresults[roi].items():
            if bootr['fits'] is None:
                continue
            # Evaluate current fits from bootstrapped results
            r2comb, gof, fitr = evaluate_fits(bootr, interp=interp)
            if np.isnan(gof) or (gof_thr is not None and (gof < gof_thr)):
                continue            
            rfdf = bootr['results'] # df with all 1000 iterations
            rfdf['r2comb'] = r2comb # add combined R2 val
            rfdf['gof'] = gof # add GoF metric            
            rfdf['sf'] = float(stimparam[0])
            rfdf['size'] = float(stimparam[1])
            rfdf['speed'] = float(stimparam[2])

            tmpd = average_metrics_across_iters(rfdf) # Average condition results
            stimkey = 'sf-%.2f-sz-%i-sp-%i' % stimparam 
            fitresults.append(tmpd)
            stimkeys.append(stimkey)
 
        if len(fitresults) > 0:
            roif = pd.concat(fitresults, axis=0).reset_index(drop=True)
            roif.index = stimkeys
            roif['cell'] = roi
            gof =  roif.mean()['gof']  
            if (gof_thr is not None) and (gof >= gof_thr): #0.66:
                goodrois.append(roi) # This is just for repoorting 
            # Select the "best" condition, so each cell has 1 fit
            if gof_thr is not None:
                best_cond_df = pd.DataFrame(roif.sort_values(by='r2comb').iloc[-1]).T
            else:
                best_cond_df = pd.DataFrame(roif.sort_values(by='gof').iloc[-1]).T
            best_dfs.append(best_cond_df)
            # But also save all results
            all_dfs.append(roif)

    # Save fit info for each stimconfig   
    if len(best_dfs) > 0: 
        best_rfits = pd.concat(best_dfs, axis=0)
        # rename indices to rois
        new_ixs = [int(i) for i in best_rfits['cell'].values]
        best_rfits.index = new_ixs
        # and all configs
        all_rfits = pd.concat(all_dfs, axis=0) 
        if verbose: 
            if gof_thr is not None: 
                print("... %i (of %i) fitable cells pass GoF thr %.2f" \
                                % (len(goodrois), len(passrois), gof_thr))
            else:
                print("... %i (of %i) fitable cells (no GoF thr)" \
                                % (best_rfits.shape[0], len(passrois)))

    return best_rfits, all_rfits #rmetrics_by_cfg


def aggregate_ori_fits(CELLS, traceid='traces001', fit_desc=None,
                       response_type='dff', responsive_test='nstds', responsive_thr=10.,
                       n_bootstrap_iters=1000, verbose=False,
                       return_missing=False, create_new=False,
                       rootdir='/n/coxfs01/2p-data',
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    assigned_cells:  dataframe w/ assigned cells of dsets that have gratings
    '''
    gdata=None
    no_fits=[]

    if fit_desc is None:
        fit_desc = get_fit_desc(response_type=response_type, 
                            responsive_test=responsive_test, 
                            n_stds=n_stds, responsive_thr=responsive_thr, 
                            n_bootstrap_iters=n_bootstrap_iters)

    aggr_fits_fpath = os.path.join(aggregate_dir, 'gratings-tuning', 'dataframes',
                               '%s.pkl' % fit_desc)
 
    if not create_new:
        try:
            with open(aggr_fits_fpath, 'rb') as f:
                res = pkl.load(f, encoding='latin1')
            gdata = res['fits']
            no_fits = res['no_fits']
        except Exception as e:
            traceback.print_exc()
            create_new=True

    if create_new:
        g_list=[];
        for (va, dk), g in CELLS.groupby(['visual_area', 'datakey']):
            try:
                # Load tuning results
                fitresults, fitparams = load_tuning_results(dk, 
                                                fit_desc=fit_desc, traceid=traceid)
                assert fitresults is not None, "ERROR: [%s] No fit results" % dk
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
            all_fits['visual_area'] = va
            all_fits['datakey'] = dk
            g_list.append(all_fits)
        gdata = pd.concat(g_list, axis=0).reset_index(drop=True)
        if verbose:
            print("Datasets with NO fits found:")
            for s in no_fits:
                print(s)
        
        with open(aggr_fits_fpath, 'wb') as f:
            pkl.dump({'fits': gdata, 'no_fits': no_fits}, f, protocol=2)

    if return_missing:
        return gdata, no_fits
    else:
        return gdata


## Plotting.

def plot_tuning_polar_roi(curr_oris, curr_resps, curr_sems=None, response_type='dff',
                          fig=None, ax=None, nr=1, nc=1, colspan=1, s_row=0, s_col=0,
                          color='k', linestyle='-', label=None, alpha=1.0):
    import pylab as pl
    if fig is None:
        fig = pl.figure()

    pl.figure(fig.number)

    # Plot polar graph:
    if ax is None:
        ax = pl.subplot2grid((nr,nc), (s_row, s_col), colspan=colspan, polar=True)
    thetas = np.array([np.deg2rad(c) for c in curr_oris])
    radii = curr_resps.copy()
    t3hetas = np.append(thetas, np.deg2rad(curr_oris[0]))  # append first value so plot line connects back to start
    radii = np.append(radii, curr_resps[0]) # append first value so plot line connects back to start
    ax.plot(thetas, radii, '-', color=color, label=label, linestyle=linestyle, alpha=alpha)
    ax.set_theta_zero_location("N")
    ax.set_yticks([curr_resps.min(), curr_resps.max()])
    ax.set_yticklabels(['', round(curr_resps.max(), 1)])


    return fig, ax


# NON ORI PARAMS 
def calculate_nonori_index(meanr, sdf, offset='minsub', at_best_ori=True,
                    param_list=['sf', 'size', 'speed']):
    '''
    Calculate pref index for SF, SIZE, SPEED.
    meanr: (pd.Series)
        Average response vec to all configs
    offset: str or None ('none', 'minsub', 'rectify')
        How to deal with negative values
    at_best_ori: (bool)
        Only calculate metric at best/most responsive ori 
    Returns:
    ixs: (pd.Series) NEG prefers lower, POS prefers higher
    '''
    if meanr.min()<0 and offset not in ['none', None]:
        if offset=='minsub':
            meanr -= meanr.min()
        elif offset=='rectify':
            meanr[meanr<0] = 0

    if at_best_ori:
        meanr_c = meanr.copy()
        meanr_c.index = sdf.loc[meanr.index]['ori'].values
        best_ori = meanr_c.groupby(meanr_c.index).mean().idxmax()
        sdf_= sdf[sdf['ori']==best_ori].copy()
    else:
        sdf_ = sdf.copy()
        best_ori='all'
    ixs={}
    for par in param_list:
        v_lo, v_hi = sdf_[par].unique().min(), sdf_[par].unique().max()
        resp_lo = meanr.loc[sdf_[sdf_[par]==v_lo].index].mean()
        resp_hi = meanr.loc[sdf_[sdf_[par]==v_hi].index].mean()
        val = (resp_hi - resp_lo) / (resp_hi+resp_lo)
        ixs.update({par: val})
    ixs.update({'ori': best_ori})
    
    return pd.Series(ixs)

def check_null(iterd, col_order=['sf', 'size', 'speed'], ci=95):
    '''
    Given bootstrapped metrics for sf, size, speed, do 2-tailed test for rejecting null,
    i.e., greater or less than 0 or nah.
    '''
    reject_or_no={}
    lo_ = np.percentile(iterd[col_order], (100-ci)/2, axis=0)
    hi_ = np.percentile(iterd[col_order], ci+((100-ci)/2), axis=0)
    paramcis = [(l, h) for l, h in zip(lo_, hi_)]
    for par, ci_ in zip(col_order, paramcis):
        reject_null = ~(ci_[0] < 0 < ci_[1])
        reject_or_no.update({par: reject_null})
        
    d1 = pd.DataFrame(iterd.median(axis=0), columns=['value'])
    d2 = pd.DataFrame(reject_or_no, index=['reject_null']).T
    df_ = pd.merge(d1, d2, left_index=True, right_index=True)
    df_.index.name = 'param'
    return df_

def count_preference_metrics(test, param_list=['sf', 'size', 'speed']):
    '''
    Count all the values and metrics for nonori bootstrap results
    for each param, returns n_pass, n_total, fraction preferring the LOW vs HIGH, etc.

    '''
    pref_df=None
    pass_ = test[test.reject_null].copy()
    n_total = len(test['cell'].unique()) #.shape[0]
    p_=[]
    for par in param_list:
        n_pass = pass_[pass_.index.get_level_values('param')==par].shape[0]
        #frac = n_pass/n_total
        # low vs high preference
        pref_lo = pass_[(pass_.index.get_level_values('param')==par) \
                      & (pass_['value']<0)]
        pref_hi = pass_[(pass_.index.get_level_values('param')==par) \
                      & (pass_['value']>0)]
        n_pref_lo = pref_lo.shape[0]
        n_pref_hi = pref_hi.shape[0]
        #frac_hi = n_pref_hi/n_pass
        #frac_lo = n_pref_lo/n_pass
        prefs_ = pd.DataFrame({'n_pass': n_pass, 'n_total': n_total,
                               'n_pref_low': n_pref_lo, 'n_pref_high': n_pref_hi},
                                index=[par])
                               #'frac_pref': frac, 
                               #'frac_pref_low': frac_lo, 'frac_pref_high': frac_hi,
        prefs_['param'] = par
        p_.append(prefs_)
    pref_df = pd.concat(p_, axis=0, ignore_index=True)
    return pref_df

def bootstrap_nonori_index(roi_resp, sdf, n_iterations=100, ci=95, 
                           at_best_ori=True, offset='minsub',
                           param_list=['sf', 'size', 'speed']):

    '''
    Do bootstrap estimation of nonori params. 

    roi_resp: (pd.DataFrame)
        roi df from NDATA (stacked). Should have columns: cell, config
    sdf:  (pd.DataFrame)
        stimconfig df for current roi's datakey

    Returns:
    
    res_: (pd.DataFrame)
        Multiindex: (cell, param) -- params listed in param_list
        Columns:  'value' and 'reject_null' (True if should reject null)
         
    '''
    if at_best_ori:
        param_list.append('ori')
    trialdf = pd.concat([pd.Series(g['response'], name=c)\
                                      .reset_index(drop=True)\
                              for c, g in roi_resp.groupby(['config'])], axis=1)
    n_resamples = trialdf.shape[0]
    iterd = pd.concat([calculate_nonori_index(\
                    trialdf.sample(n_resamples, replace=True).mean(axis=0), sdf,\
                    param_list=param_list, offset=offset)\
                    for _ in range(n_iterations)], axis=1).T
    res_ = check_null(iterd, col_order=param_list, ci=ci)
    res_['cell'] = int(roi_resp['cell'].unique())

    return res_

def fit_nonori_params(va, dk, responsive_test='ROC', responsive_thr=0.05,
                      trial_epoch='stimulus', n_iterations=100,
                      param_list=['sf', 'size', 'speed'], offset='minsub',
                      n_bootstrap_iters=500, ci=95, at_best_ori=True,
                      response_type='dff', traceid='traces001', n_processes=1,
                      visual_areas=['V1', 'Lm', 'Li']):
    # Get fit dir
    ori_fit_desc = get_fit_desc(response_type=response_type,
                            responsive_test=responsive_test, 
                            responsive_thr=responsive_thr, 
                            n_bootstrap_iters=n_bootstrap_iters)
    traceid_dir = traceutils.get_traceid_dir(dk, 'gratings', traceid='traces001')
    fitdirs = glob.glob(os.path.join(traceid_dir, 'tuning*', ori_fit_desc))
    if len(fitdirs)==0:
        print("no fits: %s" % dk)
        return None
    # outfile
    fitdir=fitdirs[0]
    tmp_fov_outfile = os.path.join(fitdir, 'results_nonori_params.pkl')
    
    # Get cells in area
    sdata, cells0 = aggr.get_aggregate_info(visual_areas=visual_areas, 
                                            return_cells=True)
    curr_cells = cells0[(cells0.visual_area==va) & (cells0.datakey==dk)]
    # Get stimuli
    sdf = aggr.get_stimuli(dk, experiment='gratings')
    # Get neuraldata
    ndf_wide = aggr.get_neuraldf(dk, experiment='gratings', traceid=traceid,
                       epoch=trial_epoch, response_type=response_type,
                       responsive_test=responsive_test, responsive_thr=responsive_thr)
    ndf_long = pd.melt(ndf_wide, id_vars=['config'], 
                  var_name='cell', value_name='response')
    all_cells = curr_cells['cell'].unique()
    ndf = ndf_long[ndf_long['cell'].isin(all_cells)]
    
    # Get preference index for all cells in FOV that pass
    ixs_ = ndf.groupby('cell').apply(bootstrap_nonori_index,\
                             sdf, param_list=param_list, offset=offset, at_best_ori=at_best_ori,
                             n_iterations=n_iterations, ci=ci)
    # Save iter results
    with open(tmp_fov_outfile, 'wb') as f:
        pkl.dump(ixs_, f, protocol=2)
    print("   saved.")
    
    return ixs_


#% misc.

def check_high_low_param_values(iterdf):
    '''
    Checks which datakeys have high/low value for each parameter (size, sf, speed).
    
    Returns:
    --------
    err_config: (dict)
        For each param, list of datakeys with too few or two many tested values

    iterdf: (pd.DataFrame)
        Same as input, but with added columns: size_rel, sf_rel, and speed_rel,
        which take values 'high' or 'low'

    '''
    err_configs = dict((k, []) for k in ['size', 'sf', 'speed'])
    if 'train_transform' in iterdf.columns:
        for param in ['size', 'sf', 'speed']:
            iterdf['%s_rel' % param] = None
        for (va, dk), g in iterdf.groupby(['visual_area','datakey']):
            for param in ['size', 'sf', 'speed']:
                if len(g[param].unique())!=2:
                    err_configs[param].append(dk)
                    continue
                minv, maxv = sorted(g[param].unique())
                iterdf.loc[g[g[param]==minv].index, '%s_rel' % param] = 'low'
                iterdf.loc[g[g[param]==maxv].index, '%s_rel' % param] = 'high'

    return iterdf, err_configs

def assign_theta_bin(fits):
    '''find tested theta value for fit theta_pref
    fits: pd.DataFrame (fit params)
    '''
    # create bin intervals
    tested_thetas = np.arange(0, 360, 45)
    half_bin = 45/2.
    bin_tuples = [] # [(0, 1), (2, 3), (4, 5)]
    for i in tested_thetas:
        print(i)
        bin_tuples.append((i-half_bin, i+half_bin))
    bins = pd.IntervalIndex.from_tuples(bin_tuples)
    bin_labels = dict((k, v) for k, v in zip(bins, tested_thetas))
    # wrap theta values >largest interval
    ixs = fits[fits['theta_pref'] > (360-half_bin)].index.tolist()
    new_vs = fits.loc[ixs]['theta_pref'].values-360.
    fits.loc[ixs, 'theta_pref'] = new_vs
    # cut into  bins
    fits['theta_interval'] = pd.cut(fits['theta_pref'], bins=bins, labels=tested_thetas)
    # assign tested theta val label
    fits['theta_bin'] = [bin_labels[i] for i in fits['theta_interval']]
    return fits

