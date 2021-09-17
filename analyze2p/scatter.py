#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 14:14:07 2021

@author: julianarhee
"""
import glob
import os
import sys
import optparse
import cv2
import glob
import importlib
import h5py
import json
import copy
import traceback
import _pickle as pkl
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import pylab as pl
import matplotlib as mpl
import scipy.stats as spstats
import analyze2p.receptive_fields.utils as rfutils
import analyze2p.extraction.rois as roiutils

import analyze2p.utils as hutils
import analyze2p.plotting as pplot
import analyze2p.aggregate_datasets as aggr
import analyze2p.retinotopy.utils as retutils
import analyze2p.retinotopy.segment as seg


from mpl_toolkits.axes_grid1 import make_axes_locatable
# In[3]:

warnings.simplefilter(action='ignore', category=FutureWarning)

# plotting
def regplot(
    x=None, y=None,
    data=None,ci=95, fit_regr=True,
    lowess=False, x_partial=None, y_partial=None,
    order=1, robust=False, dropna=True, label=None, color=None,
    scatter_kws=None, line_kws=None, ax=None, truncate=False):

    plotter = sns.regression._RegressionPlotter(x, y, data, ci=ci,
                                 order=order, robust=robust,
                                 x_partial=x_partial, y_partial=y_partial,
                                 dropna=dropna, color=color, label=label, truncate=truncate)
    if ax is None:
        ax = pl.gca()
    # Calculate the residual from a linear regression
    #_, yhat, _ = plotter.fit_regression(grid=plotter.x)
    #plotter.y = plotter.y - yhat
    #print(len(plotter.x))

    # Set the regression option on the plotter
    if lowess:
        plotter.lowess = True
    else:
        plotter.fit_reg = False
    # Draw the scatterplot
    scatter_kws = {} if scatter_kws is None else scatter_kws.copy()
    line_kws = {} if line_kws is None else line_kws.copy()
    plotter.plot(ax, scatter_kws, line_kws)
    # unfortunately the regression results aren't stored, so we rerun
    grid, yhat, err_bands = plotter.fit_regression(ax) #, grid=plotter.x)
    # also unfortunately, this doesn't return the parameters, so we infer them
    slope = (yhat[-1] - yhat[0]) / (grid[-1] - grid[0])
    intercept = yhat[0] - slope * grid[0]
    if fit_regr:
        plotter.lineplot(ax, line_kws)
    
    return slope, intercept, err_bands, plotter

def fit_with_deviants(boot_, cis_, rfs_, xname='ml_proj', yname='x0', ax=None,
                        scatter_kws={'s': 5, 'marker': 'o'}, 
                        line_kws={'lw': 0.5}, deviant_color='magenta',
                        lw=0.25, marker='o', fontsize=6, legend=True):
    if ax is None:
        fig, ax = pl.subplots()

    rois_=rfs_['cell'].unique()
    x_bins = sorted(rfs_[xname].values)
    sort_rois_by_x = np.array(rfs_[xname].argsort().values)

    # 2a. Get mean and upper/lower CI bounds of bootstrapped distn for each cell
    bootc = cis_[['%s_lower' % yname, '%s_upper' % yname]].copy()
    boot_meds = np.array([g[yname].mean() for k, g in boot_.groupby(['cell'])])
    boot_medians_df = pd.DataFrame(boot_meds,index=rois_)
    # Get YERR for plotting, (2, N), row1=lower errors, row2=upper errors
    lo = boot_meds - cis_['%s_lower' % yname].values.astype(float),
    hi = cis_['%s_upper' % yname].values.astype(float) - boot_meds
    boot_errs = np.vstack([lo, hi])
    booterrs_df = pd.DataFrame(boot_errs, columns=rois_)
    # Fit regression
    slope, intercept, err_bands, plotter = regplot(x=xname, y=yname, 
                                                data=rfs_, ax=ax, 
                                                color='k', fit_regr=True,
                                                scatter_kws=scatter_kws,line_kws=line_kws)
    eq_str = 'y=%.2fx + %.2f' % (slope, intercept)
    # Refit line to get correct xbins
    grid, yhat, err_bands = plotter.fit_regression(ax, grid=x_bins)

    # 2b. Get CIs from linear fit (fit w/ reliable rois only)
    # grid, yhat, err_bands = plotter.fit_regression(grid=plotter.x)
    e1 = err_bands[0, :] # err_bands[0, np.argsort(xvals)] <- sort by xpos to plot
    e2 = err_bands[1, :] #err_bands[1, np.argsort(xvals)]
    regr_cis = pd.DataFrame({'regr_lower': e1, 'regr_upper': e2},
                           index=rois_[sort_rois_by_x])

    # Calculate deviants
    deviants = []
    for ri, (roi, g) in enumerate(rfs_.reset_index(drop=True).groupby(['cell'])):
        (bootL, bootU) = bootc[['%s_lower' % yname, '%s_upper'% yname]].loc[roi]
        (regrL, regrU) = regr_cis[['regr_lower', 'regr_upper']].loc[roi]
        pass_boot = (bootL <= float(g[yname]) <= bootU) # estimate falls within 95 CI 
        pass_regr = ( (regrL > bootU) or (regrU < bootL) ) # 95CI of estimate falls outside of fit CIs
        if pass_regr and pass_boot:
            deviants.append(roi)

    #plot deviants
    yerrs = booterrs_df[deviants].values
    ax.scatter(rfs_[xname][deviants], rfs_[yname][deviants], 
               label='deviant', marker=marker,
               s=scatter_kws['s'], facecolors=deviant_color, 
               edgecolors=deviant_color, alpha=1.0)
    #     if plot_boot_med:
    #         ax.scatter(xv, boot_meds[dev_ixs], c=deviant_color, marker='_', alpha=1.0)
    if len(deviants)>0:
        ax.errorbar(rfs_[xname][deviants], 
                    boot_medians_df.loc[deviants].values, yerr=yerrs,
                    fmt='none', color=deviant_color, alpha=1, lw=lw)
    #ax.set_title(eq_str, loc='left', fontsize=fontsize)
    # # Old way (unprojected)
    # sns.regplot(x='ml_pos', y=yname, data=rfs_.reset_index(drop=True), ax=ax,
    #             scatter=False, color='c')
    # ax.scatter(x='ml_pos', y=yname, data=rfs_.reset_index(drop=True), s=2, color='c')
  
    if legend: 
        pl.subplots_adjust(left=0.1, right=0.75, bottom=0.25, top=0.85, wspace=0.3)

        leg_h = pplot.custom_legend_markers(colors=['k', 'k', deviant_color],
                            labels=['reliable', 'linear fit (95% CI)', 'deviant (95% CI)'],
                            markers=[marker, '_', marker], lws=[0, 0.5, 0.5])
        ax.legend(handles=leg_h, bbox_to_anchor=(1,1), loc='upper left',
                  frameon=False, fontsize=6, markerscale=0.5)
 
    return ax, deviants




def plot_scatter_and_marginals(axdf, regr_, roi_to_label=None, cond='az', 
                    xlim=None, ylim=None, 
                    lw=0.5, sz=3, nbins=10, color1='purple', color2='green'):
#    sz = 3
#    lw=0.5
#    nbins=10
#    color1='purple'
#    color2='green'

    rf_label = 'x0' if cond=='az' else 'y0'
    ctx_label = 'ml' if cond=='az' else 'ap'

    fig, scatterax = pl.subplots(figsize=(6,6))
    # Do scatter plot
    #axdf = rfs_.copy()
    sns.scatterplot(x='%s_proj' % ctx_label, y=rf_label, data=axdf, ax=scatterax,
                    color='k', edgecolor='k', s=sz)
    if xlim is None:
        xlim = np.ceil(axdf['%s_proj' % ctx_label].max())
    if ylim is None:
        ylim = np.ceil(axdf[rf_label].max())
    scatterax.set_xlim([0, xlim])
    scatterax.set_ylim([0, ylim])

    slope = float(regr_[regr_['cond']==cond]['coefficient'])
    intercept = float(regr_[regr_['cond']==cond]['intercept'])
    r2_v = float(regr_[regr_['cond']==cond]['R2'])
    label_prefix='R2=%.2f' % (r2_v)
    ls = '-' if r2_v > 0.5 else ':'

    # Plot Vert/Horz lines showing deg_scatter or dist_scatter
    npts = axdf.shape[0]
    #if npts>20:
    #pt_ixs = [int(npts/5.)] #np.arange(0, npts, 10)
    #for ii, (xi, yi) in enumerate(axdf[['%s_proj' % ctx_label, rf_label]].values):
    #    if ii not in pt_ixs:
    #        continue
    # Do DEG scatter
    if roi_to_label is None:
        med_val = abs(axdf[rf_label]).max()/2
        ii = abs(abs(axdf[rf_label])-med_val).argmin()
    else:
        ii = int(np.where(axdf['cell']==roi_to_label)[0]) #int(axdf[axdf['cell']==roi_to_label].index[0])
    xi = float(axdf['%s_proj' % ctx_label].iloc[ii])
    yi = float(axdf[rf_label].iloc[ii])
    pred_deg = axdf['predicted_%s' % rf_label].iloc[ii]
    offset_deg = axdf['deg_scatter_%s' % rf_label].iloc[ii] #*-1 if yi>pred_deg \
                        #else axdf['deg_scatter_%s' % rf_label].iloc[ii]
    scatterax.plot([xi,xi], [yi, yi-offset_deg], color2, alpha=1, lw=lw)
    # Do DIST scatter
    pred_dist = axdf['predicted_%s_proj' % ctx_label].iloc[ii]
    offset_dist = axdf['dist_scatter_%s' % ctx_label].iloc[ii]*-1 if xi>pred_dist \
                        else axdf['dist_scatter_%s' % ctx_label].iloc[ii]    
    scatterax.plot([xi, xi+offset_dist], [yi, yi], color1, alpha=1, lw=lw)
    # Draw regr line
    scatterax = pplot.abline(slope, intercept, ax=scatterax, ls=ls, lw=0.2,
                       color='k', label=False, label_prefix=label_prefix)
    scatterax.set_ylim([axdf[rf_label].min()-20, axdf[rf_label].max()+20])

    # Create top/right histograms
    divider = make_axes_locatable(scatterax)
    histax_x = divider.append_axes("top", 0.5, pad=0.5, sharex=None) #scatterax) #None)
    histax_y = divider.append_axes("right", 0.5, pad=0.5, sharey=None) #scatterax)
    # plot the marginals
    histax_x.tick_params(labelright=False, labelleft=True, 
                        bottom=True, labelbottom=True, top=False, labeltop=False)
    #sns.distplot(axdf['deg_scatter_%s' % rf_label], color=color2, ax=histax_x, 
    #             vertical=False, kde=False, bins=nbins)
    sns.histplot(x='deg_scatter_%s' % rf_label, data=axdf, color=color2, ax=histax_x, 
                 kde=False, bins=nbins, stat='count', edgecolor='w')

    histax_x.set_xlabel('visual field scatter (deg.)')

    #sns.distplot(axdf['dist_scatter_%s' % ctx_label], color=color1, ax=histax_y, vertical=True,
    #             kde=False, bins=nbins)
    sns.histplot(y='dist_scatter_%s' % ctx_label, data=axdf, color=color1, ax=histax_y, 
                 kde=False, bins=nbins, stat='count', edgecolor='w')

    histax_y.tick_params(labelright=False, labelleft=True, 
                        bottom=True, labelbottom=True, top=False, labeltop=False)
    histax_y.set_ylabel('cortical scatter (um)')
    clean_up_marginal_axes(histax_x, histax_y)
    
    return fig

def clean_up_marginal_axes(histax_x, histax_y):
    xlim, ylim=histax_x.get_ylim()
    yticks = np.linspace(xlim, ylim, 3)
    histax_x.set_yticks(yticks)
    histax_x.set_yticklabels([ int(round(i)) for i in yticks])
     #round(xlim), round(ylim)])
    histax_x.spines["right"].set_visible(False)
    histax_x.spines["left"].set_visible(True)
    histax_x.spines["top"].set_visible(False)
    histax_x.spines["bottom"].set_visible(True)

    xlim, ylim=histax_y.get_xlim()
    xticks = np.linspace(xlim, ylim, 3)
    histax_y.set_xticks(xticks)
    histax_y.set_xticklabels([int(round(i)) for i in xticks])
    #round(xlim), round(ylim)])
    histax_y.spines["right"].set_visible(False)
    histax_y.spines["left"].set_visible(True)
    histax_y.spines["top"].set_visible(False)
    histax_y.spines["bottom"].set_visible(True)


# scatter
def plot_linear_fit_and_scatter(aligned_pix, regr_meas, x_var='ml_pos', y_var='x0',
                                cond_name=None,
                                spacing=500, markers='.', s=1, mc='k', 
                                lc='m', lw=1, ax=None):
    '''
    Plot pixel coords and fov coords, with linear fit.
    '''
    if ax is None:
        fig, ax = pl.subplots()
    if cond_name is None:
        cond_name = 'az' if ('ml' in x_var or 'x' in y_var) else 'el'
    sns.scatterplot(x=x_var, y=y_var, data=aligned_pix.iloc[0::spacing], ax=ax, 
               markers=markers, s=s, color=mc)
    R2 = float(regr_meas[regr_meas.cond==cond_name]['R2'])
    RMSE = float(regr_meas[regr_meas.cond==cond_name]['RMSE'])
    prefix = '[%s] R2=%.2f, RMSE=%.2f\n' % (cond_name, R2, RMSE)
    slope = float(regr_meas[regr_meas.cond==cond_name]['coefficient'])
    intercept = float(regr_meas[regr_meas.cond==cond_name]['intercept'])
    ax = pplot.abline(slope, intercept, ax=ax, lw=lw, color=lc, 
                      label=x_var, label_prefix=prefix)
    return ax



# Save deviants
def get_deviants_in_fov(dk, va, experiment='rfs',  traceid='traces001', 
                 redo_fov=False, response_type='dff',  ecc_center=(0, 0), abs_value=False,
                 do_spherical_correction=False, verbose=False, save_plots=False):
    '''
    Finds true deviants per axis condition -- CIs per ROI vs. CIs for linear fit. 
    Only considered as deviant if this is true, and also, is "reliable" for position.
    '''
    # Get reliable c
    deviants={}
    eval_results, eval_params = rfutils.load_eval_results(dk,
                                experiment=experiment, 
                                traceid=traceid, 
                                response_type=response_type,
                                do_spherical_correction=do_spherical_correction)
                                    #fit_desc=fit_desc)   
    if eval_params is None:
        return None
    # try loading
    eval_dir = os.path.join(eval_params['rfdir'], 'evaluation')
    dev_fpath = os.path.join(eval_dir, 'deviants.json')
    if os.path.exists(dev_fpath) and redo_fov is False:
        try:
            with open(dev_fpath, 'r') as f:
                deviants = json.load(f)
            assert isinstance(deviants, dict) and va in list(deviants.keys())
            # assert va in deviants.keys(), "No deviant results found: %s, %s" % (dk, va)
        except Exception as e:
            traceback.print_exc()
            print("    REDOING.")
            #print(e)
            redo_fov=True

    cond_dict = {}
    if not isinstance(deviants, dict):
        deviants={}
    fit_desc = rfutils.get_fit_desc(response_type=response_type, 
                                    do_spherical_correction=do_spherical_correction)
    data_id = '%s|%s|%s, %s' % (fit_desc, va, dk, experiment)
    if redo_fov:
        if verbose:
            print("    identifying deviants (%s, %s)" % (dk, va))
        # Get list of all reliable fits
        #reliable_ = rfutils.get_reliable_fits(eval_results['pass_cis'],
        #                                    pass_criterion='position')
        # Get rf fits for cells in visual area
        fitrf_ = project_soma_position_in_fov(dk, va, experiment=experiment, 
                                traceid=traceid, response_type=response_type,
                                do_spherical_correction=do_spherical_correction, 
                                ecc_center=ecc_center, abs_value=abs_value)
        rfs_ = fitrf_[fitrf_.reliable].copy()
        rfs_.index = rfs_['cell'].values
        reliable_ = rfs_['cell'].unique()        
        if len(reliable_)>0: #itrf_.shape[0]>0:
            bootdata = eval_results['bootdf'].copy()
            #rois_ = np.intersect1d(reliable_, fitrf_['cell'].unique()) # reliable + in area
            boot_ = bootdata[bootdata['cell'].isin(reliable_)]
            cis_ = eval_results['cis'].loc[reliable_]
            #rfs_ = fitrf_[fitrf_['cell'].isin(reliable_)]
            
            for cond in ['az', 'el']:
                xname = 'ml_proj' if cond=='az' else 'ap_proj'
                yname = 'x0' if cond=='az' else 'y0'
                try:
                    fig, ax = pl.subplots(figsize=(5,5))                    
                    if rfs_.shape[0]>0:
                        # Get projected cortical position
                        ax, devs_ = fit_with_deviants(boot_, cis_, rfs_, 
                                                     xname=xname, yname=yname, ax=ax)
                        cond_dict.update({cond: devs_})
                        if save_plots:
                            pplot.label_figure(fig, '%s|%s (%s)' % (fit_desc, dk, va))
                            figname = 'deviants_%s_%s' % (va, cond)
                            pl.savefig(os.path.join(eval_dir, '%s.svg' % figname))
                            #print("    saved: %s" % figname)
                        pl.close('all')
                except Exception as e:
                    print("ERROR: %s, %s -- cond=%s" % (va, dk, cond))
                    traceback.print_exc()
        # save
        deviants.update({va: cond_dict}) 
        with open(dev_fpath, 'w') as f:
            json.dump(deviants, f, indent=4)                                         
    d_=[]
    for k, v in deviants[va].items():
        df_ = pd.DataFrame({'deviants': v})
        df_['axis'] = k
        d_.append(df_)
    if len(d_)>0:
        dev_df = pd.concat(d_, axis=0)
    else:
        dev_df = pd.DataFrame({'deviants': [None, None], 'axis':['az', 'el']})
    return dev_df


def aggregate_deviant_cells(response_type='dff', do_spherical_correction=False,
                            meta=None, traceid='traces001', ecc_center=(0, 0),
                            create_new=False, redo_fov=False, verbose=False, save_plots=False,
                            aggr_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Cycle thru all datasets (meta) with rfs/rfs10, and calculate deviants
    Main function is get_deviants_in_fov(). Assumes basis is RFs.

    Returns:
    
    deviants: (pd.DataFrame)
        Cell (rids of deviants), cond (az/el), datakey, visual_area
    no_results: (list)
        List of datasets (va, dk) where no deviant results found.
        
    Args:
    
    fit_desc: (str)
        Standard ID name for RFs (response_type, do_spherical_correction)
    create_new: (bool)
        Recreate main aggr. file,
        (<aggr_dir>/receptive-fields/scatter/deviants_<fit_desc>.pkl)
    redo_fov: (bool)
        Recalculate deviants (and save results to json) for each dataset.
    '''
    fit_desc = rfutils.get_fit_desc(response_type=response_type,
                                  do_spherical_correction=do_spherical_correction)

    src_dir = os.path.join(aggr_dir, 'receptive-fields/scatter')
    aggr_deviants_fpath = os.path.join(src_dir, 'deviants__%s.pkl' % fit_desc)
    if redo_fov:
        create_new=True
    if not create_new:
        try:
            with open(aggr_deviants_fpath, 'rb') as f:
                res = pkl.load(f, encoding='latin1')
            deviants = res['deviants']
            no_deviants = res['no_deviants']
        except Exception as e:
            traceback.print_exc()
            create_new=True
    #print(create_new)

    if create_new:
        if meta is None:
            sdata = aggr.get_aggregate_info(visual_areas=visual_areas)
            meta = sdata[sdata.experiment.isin(['rfs', 'rfs10'])].copy()
        d_=[]
        no_deviants=[]
        for (va, dk, exp), g in meta.groupby(['visual_area', 'datakey', 'experiment']):
            df_ = get_deviants_in_fov(dk, va, experiment=exp, traceid=traceid,
                                   response_type=response_type,
                                   do_spherical_correction=do_spherical_correction,
                                    ecc_center=ecc_center, 
                                   redo_fov=redo_fov, verbose=verbose, save_plots=save_plots)
            if not df_['deviants'].any(): # is None:
                no_deviants.append((va, dk, exp))
                continue
            df_['visual_area'] = va
            df_['datakey'] = dk
            df_['experiment'] = exp
            d_.append(df_)
        deviants = pd.concat(d_, axis=0).reset_index(drop=True)
        with open(aggr_deviants_fpath, 'wb') as f:
            pkl.dump({'deviants': deviants, 'no_deviants': no_deviants}, f, protocol=2)
            
    return deviants, no_deviants


# --------------------------------------------------------------------
# Background retino maps
# --------------------------------------------------------------------
def get_background_maps(dk, experiment='rfs', traceid='traces001',
                        response_type='dff', is_neuropil=True, 
                        do_spherical_correction=False, 
                        create_new=False, redo_smooth=False, 
                        desired_radius_um=20,
                        target_sigma_um=20, smooth_spline_x=1, smooth_spline_y=1,
                        ds_factor=1, fit_thr=0.5):
    '''
    Create smoothed background maps for AZ and EL (like done w BAR, but for TILE protocol).
    This can be funky. Better to use global gradient estimation, as with Widefield.
    Saves to: <rfdir>/neuropil_maps.pkl

    Loads RF fitting results from TILES protocol.
   
    Returns:
        res: (dict)
            Keys: 'azim_orig', 'azim_final', 'elev_orig', 'elev_final', 'fitdf'

    Args:
    redo: (bool)
        Redo the whole background smoothing and segmentation (mask dilation, etc.)
    
    redo_smooth: (bool)
        Only adjust smoothing params.


    '''
    res=None
    try:
        fit_results, fit_params = rfutils.load_fit_results(dk, experiment=experiment,
                                traceid=traceid, response_type=response_type,
                                is_neuropil=is_neuropil,
                                do_spherical_correction=do_spherical_correction)
    except FileNotFoundError as e:
        print(" skipping %s" % dk)
        return None
    maps_outfile = os.path.join(fit_params['rfdir'], 'neuropil_maps.pkl')
    redo = create_new is True
    if not create_new:
        try:
            with open(maps_outfile, 'rb') as f:
                res = pkl.load(f, encoding='latin1')
            azim_np = res['azim_orig']
            elev_np = res['elev_orig']
        except Exception as e:
            redo=True
    if redo:
        redo_smooth=True
        fitdf_all = rfutils.rfits_to_df(fit_results, fit_params, 
                               convert_coords=True, scale_sigma=True)
        fitdf = fitdf_all[fitdf_all['r2']>fit_thr].copy()
        roi_list = fitdf.index.tolist()
        # Get masks
        zproj, dilated_masks, centroids = retutils.dilate_centroids(dk,
                                            desired_radius_um=desired_radius_um,
                                            traceid=traceid)
        ixs = np.sum(dilated_masks, axis=0)
        # Get maps
        azim_ = np.array([dilated_masks[i]*v for i, v \
                                    in enumerate(fitdf['x0'].values)])
        azim_np = np.true_divide(np.nansum(azim_, axis=0), ixs)
        elev_ = np.array([dilated_masks[i]*v for i, v \
                                    in enumerate(fitdf['y0'].values)])
        elev_np = np.true_divide(np.nansum(elev_, axis=0), ixs)
    
    if redo_smooth:
        # Smmooth
        pixel_size = hutils.get_pixel_size()
        sm_azim, sm_elev = seg.smooth_maps(azim_np, elev_np, 
                                target_sigma_um=target_sigma_um,  
                                smooth_spline=(smooth_spline_x, smooth_spline_y),
                                fill_nans=True,
                                start_with_transformed=False, 
                                use_phase_smooth=False, ds_factor=ds_factor)
        sm_azim.update({'input': azim_np})
        sm_elev.update({'input': elev_np})
        fig, axn = seg.plot_smoothing_results(sm_azim, sm_elev)
        fig.text(0.01, 0.9, dk)
        pl.savefig(os.path.join(fit_params['rfdir'], 'neuropil_maps.svg'))
        pl.close()
        res = {'azim_orig': azim_np, 
               'azim_final': sm_azim['final'],
               'elev_orig': elev_np, 
               'elev_final': sm_elev['final'],
               'fitdf': fitdf,
               'zproj': zproj}
        with open(maps_outfile, 'wb') as f:
            pkl.dump(res, f, protocol=2)

    return res

def cycle_and_load_maps(dk_list, experiment='rfs', traceid='traces001',
                        response_type='dff', do_spherical_correction=False,
                        is_neuropil=True,
                        target_sigma_um=20, desired_radius_um=20,
                        smooth_spline_x=1, smooth_spline_y=1, ds_factor=1,
                        create_new=False, redo_smooth=False,
                        verbose=False,
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Quick func to cycle dsets to create neuropil maps using TILE experiments.
    '''
#    target_sigma_um=40
#    desired_radius_um=20
#    smooth_spline_x=1
#    smooth_spline_y=1
#    create_new=False
#    redo_smooth=True
#    is_neuropil=True

    basedir = os.path.join(aggregate_dir, 'receptive-fields', 'neuropil')
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    tmp_masks_fpath = os.path.join(basedir, 'np_maps.pkl')

    try_loading = (create_new is False) and (redo_smooth is False)
    if try_loading:
        try:
            with open(tmp_masks_fpath, 'rb') as f:
                MAPS = pkl.load(f, encoding='latin1')
        except Exception as e:
            if verbose:
                traceback.print_exc()
            print("Cycling thru to get background maps.")
            create_new=True

    if create_new or redo_smooth:
        MAPS = dict() #dict((k, ) for k in dk_list)
        for dk in dk_list:
            res = get_background_maps(dk, experiment=experiment, traceid=traceid,
                            response_type=response_type, is_neuropil=is_neuropil,  
                            do_spherical_correction=do_spherical_correction,    
                            create_new=create_new, redo_smooth=redo_smooth, 
                            target_sigma_um=target_sigma_um, 
                            smooth_spline_x=smooth_spline_x, 
                            smooth_spline_y=smooth_spline_y, ds_factor=ds_factor)
            MAPS[dk] = res

        with open(tmp_masks_fpath, 'wb') as f:
            pkl.dump(MAPS, f, protocol=2)
            
    return MAPS

#---------------------------------------------------------------------
# Functions to load NP from MOVING BAR
# --------------------------------------------------------------------
def get_best_retinorun(datakey):
    all_retinos = retutils.get_average_mag_across_pixels(datakey)     
    retinorun = all_retinos.iloc[all_retinos[1].idxmax()][0]
    #retinorun = all_retinos.loc[all_retinos[1].idxmax()][0] 
    return retinorun

def load_movingbar_results(dk, retinorun, traceid='traces001',
                        rootdir='/n/coxfs01/2p-data'):
    # load retinodata
    retinoid, RETID = retutils.load_retino_analysis_info(
                        dk, run=retinorun, use_pixels=False)
    data_id = '_'.join([dk, retinorun, retinoid])
    #print("DATA ID: %s" % data_id)
    scaninfo = retutils.get_protocol_info(dk, run=retinorun)

    # Image dimensions
    d2_orig = scaninfo['pixels_per_line']
    d1_orig = scaninfo['lines_per_frame']
    #print("Original dims: [%i, %i]" % (d1_orig, d2_orig))
    ds_factor = int(RETID['PARAMS']['downsample_factor'])
    #print('Data were downsampled by %i.' % ds_factor)
    # Get pixel size
    pixel_size = hutils.get_pixel_size()
    pixel_size_ds = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)
    d1 = int(d1_orig/ds_factor)
    d2 = int(d2_orig/ds_factor)
    #print(d1, d2)
    # Load fft 
    fft_results = retutils.load_fft_results(dk,
                                    retinorun=retinorun, traceid=traceid, 
                                    rootdir=rootdir, create_new=False,
                                    use_pixels=False)
    fft_soma = fft_results['fft_soma']
    fft_np = fft_results['fft_neuropil']
    # Create dataframe of magratios -- each column is a condition
    magratios_soma, phases_soma = retutils.extract_from_fft_results(fft_soma)
    magratios_np, phases_np = retutils.extract_from_fft_results(fft_np)
    dims = (d1_orig, d2_orig)
    
    return magratios_soma, phases_soma, magratios_np, phases_np, dims


def adjust_retinodf(mvb_np, mag_thr=0.02):
    '''
    Filter out cells that are not responsive (mag_thr),
    and convert phase to centered linear coords of screen.
    '''
    if mag_thr is None:
        return mvb_np

    # Filter
    pass_mag_rois = mvb_np[(mvb_np.mag_az>mag_thr) 
                          & (mvb_np.mag_el>mag_thr)].index.tolist()
    retinodf_np = mvb_np.loc[pass_mag_rois]
    # Get screen info
    screen = hutils.get_screen_dims()
    screen2p_x = screen['azimuth_deg'] # 119.5564
    screen2p_y = screen['altitude_deg'] #67.323
    resolution2p = screen['resolution'] #[1920, 1080] #[1024, 768]
    # Convert to screen coords
    abs_vmin, abs_vmax = (-np.pi, np.pi)
    lmax_az_2p = screen2p_x #/2.
    lmin_az_2p = 0 #-screen2p_x #-lmax_az_2p
    lmax_el_2p = screen2p_y #/2.
    lmin_el_2p = 0 #-screen2p_y# 0 #-lmax_el_2p
    retinodf_np['az_lin'] = hutils.convert_range(retinodf_np['phase_az'], 
                                       newmin=lmin_az_2p, newmax=lmax_az_2p, 
                                       oldmin=abs_vmin, oldmax=abs_vmax)
    retinodf_np['el_lin'] = hutils.convert_range(retinodf_np['phase_el'], 
                                       newmin=lmin_az_2p, newmax=lmax_az_2p, 
                                       oldmin=abs_vmin, oldmax=abs_vmax)
    retinodf_np['x0'] = retinodf_np['az_lin'] - (lmax_az_2p/2.)
    retinodf_np['y0'] = retinodf_np['el_lin'] - (lmax_az_2p/2.)
    
    return retinodf_np


def load_neuropil_data(dk, retinorun, mag_thr=0.001, delay_map_thr=1.0, ds_factor=2,
                    visual_areas=['V1', 'Lm', 'Li']):
    '''
    Wrapper for loading neuropil data from movingbar.
    Loads FFT results and calculates final retino pref. estimates for each NP mask.
    Converts phase to linear (screen) coords in DEG visual angle.
    Adds CTX position info for each centroid (Assumes single VA for all cells,
    so should filter with seg.load_roi_assignments() after).
    Filters out poorly responding cells.

    Returns:  retinodf_np (pd.DataFrame)
    '''
    df = None
    # Load FFT results
    mags_soma, phases_soma, mags_np, phases_np, dims = load_movingbar_results(dk, 
                                                                              retinorun)
    # Get maps:  abs_vmin, abs_vmax = (-np.pi, np.pi)
    mvb_np = retutils.get_final_maps(mags_np, phases_np, 
                        trials_by_cond=None,
                        mag_thr=None, dims=dims,
                        ds_factor=ds_factor, use_pixels=False)
    # Filter bad responses
    df = adjust_retinodf(mvb_np.dropna().copy(), mag_thr=mag_thr)
    # Add cell position info
    df = add_position_info(df, dk, 'retino', retinorun=retinorun)

    return df

# --------------------------------------------------------------------
# Gradient functions
# --------------------------------------------------------------------
def get_smoothed_area_map(dk, va, create_new=False, map_type='final',
                        rootdir='/n/coxfs01/2p-data'):

    '''
    Load or create smoothed background map from pixels (segemented area).
    Saves maps, if create_new.

    Mimics the processing steps in ./retinotopy/segmentation.py(), which is 
    used to segment areas (identify_visual_areas.ipynb)
    
    Returns:
    
    AZMAP, ELMAP
    
    Args:
    
    map_type (str) 
        'final' -- final maps (smoothed and nan-filled, over phase)
        'smoothed' -- intermediate step
        'start' -- input map (filtered, no smooth)

    '''
    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
    gradients_basedir = os.path.join(fovdir, 'segmentation')

    maps={} 
    map_file = os.path.join(gradients_basedir, 'smoothed_area_maps.pkl')
    if not create_new:
        try:
            with open(map_file, 'rb') as f:
                maps = pkl.load(f, encoding='latin1')
                assert va in maps.keys(), "Area --%s-- not found (%s)." % (va, dk)
        except Exception as e:
            #trackeback.print_exc()
            create_new=True 
            print("    found maps: %s" % str(list(maps.keys()))) 
     
    if create_new:
        retinorun = get_best_retinorun(dk)
        sm_azim, sm_elev = smooth_within_area_mask(dk, va, retinorun)
        # Save for visual area
        new_maps = {'final_az': sm_azim['final'], 'final_el': sm_elev['final'], 
                    'start_az': sm_azim['input'], 'start_el': sm_elev['input'],
                    'smoothed_az': sm_azim['smoothed'], 'smoothed_el': sm_elev['smoothed']}
        maps.update({va: new_maps})
        with open(map_file, 'wb') as f:
            pkl.dump(maps, f, protocol=2)

    AZMAP_NP = maps[va]['%s_az' % map_type]
    ELMAP_NP = maps[va]['%s_el' % map_type]

    return AZMAP_NP, ELMAP_NP

def smooth_within_area_mask(dk, va, retinorun):
    '''
    Load original pixel maps, do smoothing WITHIN area (segmentation results).
    '''
    # Load original image
    pmaps, pparams = seg.get_processed_maps(dk, retinorun=retinorun, create_new=False)

    # Load segmentation and mask for current visual area
    seg_results, seg_params = seg.load_segmentation_results(dk, retinorun=retinorun)
    segmented_areas = seg_results['areas']
    area_results = segmented_areas[va].copy()
    curr_segmented_mask = area_results['mask']

    # Smooth within VA to avoid artefacts from spline>1
    start_az = pmaps['filtered_az_scaled']
    start_el = pmaps['filtered_el_scaled']

    thr_img_az = start_az.copy() 
    thr_img_az[curr_segmented_mask==0] = np.nan     

    thr_img_el = start_el.copy() 
    thr_img_el[curr_segmented_mask==0] = np.nan     
    # -------
    smooth_spline=1
    target_sigma_um=25 # 
    sm_azim, sm_elev = seg.smooth_maps(thr_img_az, thr_img_el, 
                                target_sigma_um=target_sigma_um, #smooth_fwhm=smooth_fwhm, 
                                smooth_spline=(smooth_spline, smooth_spline), 
                                fill_nans=True,
                                start_with_transformed=True, 
                                use_phase_smooth=False)
    sm_azim['input'] = thr_img_az
    sm_elev['input'] = thr_img_el

    return sm_azim, sm_elev

 
    return AZMAP_NP, ELMAP_NP, thr_img_az, thr_img_el


def load_neuropil_background(datakey, retinorun, map_type='final', protocol='BAR',
                        rootdir='/n/coxfs01/2p-data'):
    '''
    Load NP background (MOVINGBAR) from:
        <retinorun_dir>/retino_analysis/segmentation/smoothed_maps.npz 
    Saved by notebook (interactive for manual check):
        analyze2p/retinotopy/identify_visual_areas.ipynb.

    TODO:  Clean up, so all this stuff in same place 
    (retino_structure - has projections_, vectors_.pkl
    vs. segmentation - has smoothed/processed maps)

    Args.

    map_type (str) 
        'final' -- final maps (smoothed and nan-filled, over phase)
        'smoothed' -- intermediate step
        'start' -- input map (filtered, no smooth)

    '''

    session, animalid, fovnum = hutils.split_datakey_str(datakey)
    maps_fpaths = glob.glob(os.path.join(rootdir, animalid, 
                        session, 'FOV%i_*' % fovnum, '%s*' % retinorun, 
                        'retino_analysis/segmentation/smoothed_maps.npz'))[0]
    maps = np.load(maps_fpaths)
    if map_type=='final':
        az_ = maps['azimuth']
        el_ = maps['elevation']
    else:
        az_ = maps['start_az']
        el_ = maps['start_el']

    # screen info
    screen = hutils.get_screen_dims()
    screen_max = screen['azimuth_deg']/2.
    screen_min = -screen_max
    # Convert to screen units
    vmin, vmax = (-np.pi, np.pi)
    az_map = hutils.convert_range(az_, oldmin=vmin, oldmax=vmax, 
                                  newmin=screen_min, newmax=screen_max)
    el_map = hutils.convert_range(el_, oldmin=vmin, oldmax=vmax, 
                                  newmin=screen_min, newmax=screen_max)

    return az_map, el_map

def load_pixel_maps(dk, va, create_new=False, map_type='final',
                        rootdir='/n/coxfs01/2p-data'):
    '''
    create_new: (bool)
        Set to re-create smoothed image (i.e., if re-segmented)

    map_type (str) 
        'final' -- final maps (smoothed and nan-filled, over phase)
        'smoothed' -- intermediate step
        'start' -- input map (filtered, no smooth)
    '''
    # Get smoothed background from pixel map (within area)
    az_, el_ = get_smoothed_area_map(dk, va, create_new=create_new, map_type=map_type)
        
    # screen info
    screen = hutils.get_screen_dims()
    screen_max = screen['azimuth_deg']/2.
    screen_min = -screen_max
    # Convert to screen units
    vmin, vmax = (-np.pi, np.pi)
    az_map = hutils.convert_range(az_, oldmin=vmin, oldmax=vmax, 
                                  newmin=screen_min, newmax=screen_max)
    el_map = hutils.convert_range(el_, oldmin=vmin, oldmax=vmax, 
                                  newmin=screen_min, newmax=screen_max)

    return az_map, el_map


def load_gradients(dk, va, retinorun='retino_run1', create_new=False,
                    rootdir='/n/coxfs01/2p-data'):
    '''
    Load gradient results for specified visual area and run.
    Results are in: <FOVDIR>/segmentation/results_VAREA.pkl

    Returns:
    
    results: (dict)
        'az/el_gradients': {
            image: gradient image (az)
            magnitude: np.sqrt(gdx**2 + gdy**2)
            gradient_x: gdx values
            gradient_y: gdy values
            direction: direction of gradient at each point
            mean_deg/_direction:  mean direction in DEG or RAD
            vhat: unit vector 
        }
        area_mask:  binary mask with 1's corresponding to specified visual area (va)
        retinorun:  best run (used to calculate grads)
        visual_area: area specified in fov
    '''
    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
    gradients_basedir = os.path.join(fovdir, 'segmentation')
    if not os.path.exists(gradients_basedir):
        os.makedirs(gradients_basedir) 
    gradients_fpath = os.path.join(gradients_basedir, 'gradient_results.pkl')
    if not create_new:
        #print("Loading (%s, %s)." % (dk, va))
        try: 
            with open(gradients_fpath, 'rb') as f:
                results = pkl.load(f, encoding='latin1')
            if len(list(results.keys()))>1:
                print('    found: %s, %s' % (dk, va), results.keys())
            curr_results = results[va].copy()

            #assert va in list(results.keys()), "Area <%s> not in results. creating now." % va
        except Exception as e:
            create_new=True
            traceback.print_exc()
        
    if create_new:
        print("... calculating global gradients (%s, %s)" % (dk, va))
        # Load area segmentation results 
        seg_results, seg_params = seg.load_segmentation_results(dk, retinorun=retinorun)
        segmented_areas = seg_results['areas']
        region_props = seg_results['region_props']
        assert va in segmented_areas.keys(), \
            "Visual area <%s> not in region. Found: %s" % (va, str(segmented_areas.keys())) 
        curr_area_mask = segmented_areas[va]['mask'].copy()

        # Load pixel map
        AZMAP_NP, ELMAP_NP = load_pixel_maps(dk, va, create_new=create_new, 
                                            map_type='final')
        # Load NP masks
#        AZMAP_NP, ELMAP_NP = load_neuropil_background(dk, retinorun,
#                                            map_type='final', protocol='BAR')
        # Calculate gradients
        grad_az, grad_el = seg.calculate_gradients(curr_area_mask, AZMAP_NP, ELMAP_NP)

        # Save
        results={}
        if os.path.exists(gradients_fpath):
            with open(gradients_fpath, 'rb') as f:
                results = pkl.load(f, encoding='latin1')
       
        curr_results = {'az_gradients': grad_az,
                        'el_gradients': grad_el,
                        'area_mask': curr_area_mask,
                        'retinorun': retinorun}
        results.update({va: curr_results})
        with open(gradients_fpath, 'wb') as f:
            pkl.dump(results, f, protocol=2)
           
         
    return curr_results #results[va]


# --------------------------------------------------------------------
# ALIGNMENT functions.
# --------------------------------------------------------------------
def apply_projection(df_, u1, u2, xlabel='ml_proj', ylabel='ap_proj', abs_value=False):
    '''Project data position (ml_pos, al_pos) onto direction of gradients,
    specified by u1 (azimuth) and u2 (elevation)'''
    M = np.array([[u1[0], u1[1]],
                  [u2[0], u2[1]]])
   # transf_vs = [M.dot(np.array([x, y])) for (x, y) \
   #                  in df_[['ml_pos', 'ap_pos']].values]

    transf_vs = [M@np.array([x, y]) for (x, y) \
                     in df_[['ml_pos', 'ap_pos']].values]

    t_df = pd.DataFrame(transf_vs, columns=[xlabel, ylabel], index=df_.index)
    if abs_value:
        t_df = t_df.abs()

    return t_df, M


def project_onto_gradient(df, gvectors, xlabel='ml_pos', ylabel='ap_pos', abs_value=False):
    '''
    Align FOV to gradient vector direction w transformation matrix.
    Use gvectors to align coordinates specified in df.
    Note: calculate separate for each axis. (prev. called align_cortex_to_gradient)
    
    gvectors: dict()
        keys/vals: 'az': [v1, v2], 'el': [w1, w2]
    df: pd.DataFrame()
        coordi
    '''
    # Transform predicted-ctx pos back to FOV coords
    u1 = gvectors['az']
    u2 = gvectors['el']

    # Transform FOV coords to lie alone gradient axis
    transf_df, M = apply_projection(df, u1, u2, 
                                    xlabel='ml_proj', ylabel='ap_proj', abs_value=abs_value)
    # rename

    return transf_df, M

def regress_cortex_and_retino_pos(df, xvar='pos', model='ridge'):
    '''
    Linear regression for each condition (az, el). 
    Return as dataframe.
    '''
    r_=[]
    for ai, cond in enumerate(['az', 'el']):
        ctx_label = 'ml' if cond=='az' else 'ap'
        ret_label = 'x0' if cond=='az' else 'y0'

        xvs = df['%s_%s' % (ctx_label, xvar)].values
        yvs = df['%s' % ret_label].values
        regr_, linmodel = do_linear_fit(xvs, yvs, model=model)
        regr_['cond'] = cond
        r_.append(regr_)
    regr_tiles = pd.concat(r_).reset_index(drop=True)

    return regr_tiles

# ALIGNMENT:  plotting -----------------------
def scatter_ctx_vs_retino_by_cond(df_, 
                             az_x='ml_pos', az_y='x0', el_x='ap_pos', el_y='y0',
                             xlabel='meas. CTX pos (um)', 
                             ylabel='meas. RET pos (deg)', scatter_kws={'s': 5}):
    '''Plot regression for azimuth (left) and elevation (right) for Y on X
    az_x:  x-axis for Azimuth plot.
    az_y:  y-axis for Azimuth plot.
    el_x = x-axis for Elevation plot
    el_y:  y-axis for Elevation plot.
    (Called: plot_regression_az_and_el in Nb)
    '''
    fig, axn = pl.subplots(1,2, figsize=(6.5,3))
    ax=axn[0]; ax.set_title('Azimuth');
    sns.regplot(az_x, az_y, df_, ax=ax, scatter_kws=scatter_kws)

    ax=axn[1]; ax.set_title('Elevation');
    sns.regplot(el_x, el_y, df_, ax=ax, scatter_kws=scatter_kws)
    for ax in axn:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    pl.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.8, wspace=0.5)
    return fig

def plot_regression_ctx_vs_retino_pos(df, regr):
    '''Plot AZ and EL results for ctx vs. retino pos'''
    fig, axn = pl.subplots(1,2, figsize=(8,4))
    for ai, (cond, regr_) in enumerate(regr.groupby(['cond'])):
        ax=axn[ai]
        ctx_label = 'ml' if cond=='az' else 'ap'
        ret_label = 'x0' if cond=='az' else 'y0'
        xvs = df['%s_proj' % ctx_label].values
        yvs = df['%s' % ret_label].values

        sns.regplot(xvs, yvs, ax=ax)
        fit_str = '(R2=%.2f) y=%.1fx + %.1f' \
                % (regr_['R2'], regr_['coefficient'], regr_['intercept'])
        ax.set_title('%s\n%s' %(cond, fit_str), loc='left', fontsize=12)
        ax.set_ylabel('retino pos')
        ax.set_xlabel('ctx pos')
    pl.subplots_adjust(left=0.1, right=0.8, bottom=0.3, hspace=0.5, top=0.8,
                      wspace=0.5)
    return fig

def plot_measured_and_aligned(aligned_np, REGR_NP, REGR_MEAS):
    fig, axn = pl.subplots(2,2, figsize=(6.5,6))
    for ai, cond in enumerate(['az', 'el']):
        regr_ = REGR_NP[REGR_NP.cond==cond].copy()
        ctx_ = 'ml' if cond=='az' else 'ap'
        ret_ = 'x' if cond=='az' else 'y'
        ax=axn[0, ai]
        sns.regplot('%s_pos' % ctx_, '%s0' % ret_, data=aligned_np, 
                    ax=ax, color='k', scatter_kws={'s':2}, label='measured')
        regr_meas_ = REGR_MEAS[REGR_MEAS.cond==cond].copy()
        regr_meas_str = 'y=%.2fx+%.2f (R2=%.2f)\npearson r=%.2f, p=%.2f'\
                        % (regr_meas_['coefficient'], regr_meas_['intercept'], 
                           regr_meas_['R2'], 
                           regr_meas_['pearson_r'], regr_meas_['pearson_p'])
        ax.set_title(regr_meas_str, loc='left', fontsize=8)
        ax=axn[1, ai]
        sns.regplot('%s_proj' % ctx_, '%s0' % ret_, data=aligned_np, 
                    ax=ax, color='m', scatter_kws={'s':2}, label='aligned')
        # show linear fit
        (slope, intercept), = regr_[['coefficient', 'intercept']].values
        xvs = aligned_np['%s_pos' % ctx_].values
        yvs = xvs*slope + intercept
        regr_str = 'y=%.2fx+%.2f (R2=%.2f)\npearson r=%.2f, p=%.2f'\
                        % (slope, intercept, regr_['R2'], 
                           regr_['pearson_r'], regr_['pearson_p'])
        ax.plot(xvs, yvs, 'r:', label='regression')
        ax.set_title(regr_str, loc='left', fontsize=8)
    pl.subplots_adjust(bottom=0.2, wspace=0.5, hspace=0.8, right=0.75, top=0.85)
    pl.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=6)
    return fig

# --------------------------------------------------------------------
# Model CTX vs RETINOTOPIC POS.
# --------------------------------------------------------------------
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.model_selection import train_test_split
import scipy.stats as spstats
import sklearn.metrics as skmetrics #import mean_squared_error

def fit_linear_regr(xvals, yvals, return_regr=False, model='ridge'):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    if model=='ridge':
        regr = Ridge()
    elif model=='Lasso':
        regr = Lasso()
    else:
        model = 'ols'
        regr = LinearRegression()
    if len(xvals.shape) == 1:
        xvals = np.array(xvals).reshape(-1, 1)
        yvals = np.array(yvals).reshape(-1, 1)
    else:
        xvals = np.array(xvals)
        yvals = np.array(yvals)
    if any([np.isnan(x) for x in xvals]) or any([np.isnan(y) for y in yvals]):
        print("NAN")
    regr.fit(xvals, yvals)
    fitv = regr.predict(xvals)
    if return_regr:
        return fitv.reshape(-1), regr
    else:
        return fitv.reshape(-1)

def do_linear_fit(xvs, yvs, model='ridge', di=0, verbose=False):
    fitv, regr = fit_linear_regr(xvs, yvs,
                            return_regr=True, model=model)
     
    rmse = np.sqrt(skmetrics.mean_squared_error(yvs, fitv))
    r2 = skmetrics.r2_score(yvs, fitv)
    pearson_r, pearson_p = spstats.pearsonr(xvs, yvs) 
    slope = float(regr.coef_)
    intercept = float(regr.intercept_)

    model_results = {'fitv': fitv, 
                     'regr': regr}

    regr_df = pd.DataFrame({ 
                      'R2': r2,
                      'RMSE': rmse,
                      'pearson_p': pearson_p,
                      'pearson_r': pearson_r,
                      'coefficient': slope, # float(regr.coef_), 
                      'intercept': intercept, #float(regr.intercept_)
                      }, index=[di])
    if verbose:
        print("~~~regr results: y = %.2f + %.2f (R2=%.2f)" % (slope, intercept, r2))

    return regr_df, model_results

def load_models(dk, va, return_best=False, return_all=False, rootdir='/n/coxfs01/2p-data'):
    '''
    Load saved REGR results (linear fit for ctx vs. retino position).

    return_best: (bool)
        Set flag to return UNALIGNED, if it's better. Otherwise, just always return corrected.

    '''
    REGR_NP=None; REGR_=None;
    try:
        session, animalid, fovn = hutils.split_datakey_str(dk)
        fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
        curr_dst_dir = os.path.join(fovdir, 'segmentation')
        alignment_fpath = os.path.join(curr_dst_dir, 'linear_fits.pkl')
        with open(alignment_fpath, 'rb') as f:
            regr_results = pkl.load(f, encoding='latin1')
        assert va in regr_results, "Visual area not found"
        REGR_ = regr_results[va].copy()

        if return_all:
            return REGR_

        if return_best:
            pre = float(REGR_[~REGR_.aligned]['R2'].mean())
            post = float(REGR_[REGR_.aligned]['R2'].mean())
            if pre>post:
                REGR_NP = REGR_[~REGR_.aligned]
            else:
                REGR_NP = REGR_[REGR_.aligned]
        else:
            REGR_NP = REGR_[REGR_.aligned]

    except Exception as e:
        traceback.print_exc() 
        return None
    
    return REGR_NP

def update_models(dk, va, REGR_NP, #create_new=False, 
                    rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
    curr_dst_dir = os.path.join(fovdir, 'segmentation')

    alignment_fpath = os.path.join(curr_dst_dir, 'linear_fits.pkl')
    results={}
    if os.path.exists(alignment_fpath): #and (create_new is False):
        with open(alignment_fpath, 'rb') as f:
            results = pkl.load(f, encoding='latin1')
        if not isinstance(results, dict):
            results={}

    results.update({va: REGR_NP})
    with open(alignment_fpath, 'wb') as f:
        pkl.dump(results, f, protocol=2)

    return

def add_position_info(df, dk, experiment, retinorun='retino_run1'):
    '''
    Correctly assign visual area to each cell based on segmentation results,
    specify RETINORUN for loading roi assignemnts.
    
    '''
    # Add pos info to NP masks
    df['cell'] = df.index.tolist()
    # Assign va to each cell
    roi_assignments = seg.load_roi_assignments(dk, retinorun=retinorun)
    df['visual_area'] = None
    for va, rois_ in roi_assignments.items():
        df.loc[df['cell'].isin(rois_), 'visual_area'] = str(va)
    # Add other meta info
    df = hutils.add_meta_to_df(df, {'datakey': dk,
                                    'experiment': experiment})
    df = aggr.add_roi_positions(df)

    return df

# --------------------------------------------------------------------
# Scatter analysis functions
# --------------------------------------------------------------------
def project_soma_position_in_fov(dk, va, experiment='rfs', traceid='traces001', 
        response_type='dff', do_spherical_correction=False, 
        return_transformation=False, ecc_center=(0, 0), abs_value=False):
    '''
    Simplified function: load G-vectors only, align soma coords. 
    Prev called  get_projected_soma_rfs()
    Returns:

    aligned_soma: (pd.DataFrame)
        Return aligned_soma (soma pos projected with gradient dir).

    return_transformation: (bool)
        Set to also return transformation matrix to go bw projected and true coords.

    '''
    # Load gradient vectors
    GVECTORS = load_vectors(dk, va, create_new=False)
    # Load soma data w/ RF fits
    df_soma = load_soma_data(dk, experiment=experiment,
                                protocol='TILE', traceid=traceid,
                                response_type=response_type,
                                do_spherical_correction=do_spherical_correction, 
                                ecc_center=ecc_center)
    if df_soma is None:
        return None

    # Align soma coords to gradient
    curr_somas = df_soma[df_soma.visual_area==va].copy()
    aligned_, M = project_onto_gradient(curr_somas, GVECTORS,
                                      xlabel='ml_pos', ylabel='ap_pos', abs_value=abs_value)
    aligned_soma = pd.concat([curr_somas, aligned_], axis=1).dropna()\
                        .reset_index(drop=True)

    if return_transformation:
        return aligned_soma, M
    else:
        return aligned_soma


def load_soma_data(dk, experiment='rfs', retinorun='retino_run1', 
                        protocol='TILE', traceid='traces001',
                        response_type='dff', 
                        do_spherical_correction=False, fit_thr=0.5,
                        mag_thr=0.01, ecc_center=(0, 0), verbose=False):
    '''
    Load SOMA data (and visual_area assignemnts) -- if TILE, includes reliable or not.
    **Specify RETINORUN for correct roi assignments .
    (even if protocol is TILE).
    ''' 
    df=None
    if protocol=='BAR':
        magthr_2p=0.001
        delay_map_thr=1.0
        ds_factor=2
        mags_soma, phases_soma, mags_np, phases_np, dims = load_movingbar_results(dk, 
                                                                    retinorun)
        # #### Get maps:  abs_vmin, abs_vmax = (-np.pi, np.pi)
        mvb_soma = retutils.get_final_maps(mags_soma, phases_soma, 
                            trials_by_cond=None,
                            mag_thr=magthr_2p, dims=dims,
                            ds_factor=ds_factor, use_pixels=False)
        # add pos
        mvb_soma['cell'] = mvb_soma.index.tolist()
        # adjust to LIN coords
        mvb_soma = adjust_retinodf(mvb_soma.dropna(), mag_thr=mag_thr)
        df = mvb_soma.copy()
    else:
        fit_desc = rfutils.get_fit_desc(response_type=response_type,
                                do_spherical_correction=do_spherical_correction)
        fitdf_soma = rfutils.load_rf_fits(dk, experiment=experiment, fit_desc=fit_desc,
                                        ecc_center=ecc_center)
        if fitdf_soma is None:
            return None

        fitdf_soma['cell'] = fitdf_soma.index.tolist()
        assert 'eccentricity' in fitdf_soma.columns
        # Get reliable
        reliable_rois=[]
        try: # Load eval results 
            eval_results, eval_params = rfutils.load_eval_results(dk,
                                            experiment=experiment, 
                                            traceid=traceid, 
                                            fit_desc=fit_desc)   
            if eval_results is not None:                
                # check if all params within 95% CI
                reliable_rois = rfutils.get_reliable_fits(eval_results['pass_cis'],
                                                     pass_criterion='position')
        except Exception as e: 
            raise e
            
        cells_ = fitdf_soma['cell'].unique()
        if verbose:
            print("    %s, %s: %i of %i reliable" % (va, dk, len(reliable_rois), len(cells_)))
        fitdf_soma['reliable'] = False 
        fitdf_soma.loc[reliable_rois, 'reliable'] = True
        
    # Add pos info to NP masks
    df = add_position_info(fitdf_soma.copy(), dk, experiment, retinorun=retinorun)

    return df.reset_index(drop=True)

# predicted_rf_locs = slope*proj_locs + intercept
# predicted_ctx_locs = (actual_rf_locs - intercept) / slope
def predict_cortex_position(regr, cond='az', points=None):
    g_intercept = float(regr[regr.cond==cond]['intercept'])
    g_slope = float(regr[regr.cond==cond]['coefficient'])
    predicted_ctx_x = (points - g_intercept) / g_slope

    return predicted_ctx_x

def predict_retino_position(regr, cond='az', points=None):
    g_intercept = float(regr[regr.cond==cond]['intercept'])
    g_slope = float(regr[regr.cond==cond]['coefficient'])
    predicted_ret_x = (points * g_slope) + g_intercept

    return predicted_ret_x

def check_inbounds(df):
    # TMP -- not always 512x512..
    (pix_ap_um, pix_ml_um) = hutils.get_pixel_size()
    ap_lim = 512*pix_ap_um
    ml_lim = 512*pix_ml_um
    inbounds = df[(df.ml_proj<=ml_lim) & (df.ap_proj<=ap_lim)].copy()
    df['inbounds'] = False
    df.loc[inbounds.index, 'inbounds'] = True

    return df

def combined_axis_colum_names(x_cols):
    name_lut={}
    for name in x_cols:
        if  'scatter' in name or 'gradient' in name:
            if '0' in name:
                new_name = name.split('_x0')[0] if 'x0' in name \
                            else name.split('_y0')[0] 
            else:
                new_name = name.split('_ml')[0] if 'ml' in name \
                            else name.split('_ap')[0] 
        elif 'x0' in name or 'y0' in name:
            new_name = name.replace('x0', 'rf_pos') if 'x0' in name \
                            else name.replace('y0', 'rf_pos')
        elif 'x' in name or 'y' in name:
            new_name = name.split('_x')[0] if 'x' in name \
                            else name.split('_y')[0] 
        elif 'ml' in name or 'ap' in name:
            new_name = name.replace('ml', 'ctx') if 'ml' in name \
                            else name.replace('ap', 'ctx')
        name_lut.update({name: new_name})
    return name_lut

def stack_axes(df):
    '''
    Stack so that "axis" is a column for az/el. Rename variables appropriately.
    '''
    ignore_cols = ['fov_xpos', 'fov_ypos', 'fov_xpos_pix', 'fov_ypos_pix']
    non_cond_cols = ['experiment', 'visual_area', 'datakey', 'cell', 'r2',
                    'theta', 'offset', 'amplitude', 'aniso_index',
                    'ratio_xy', 'major_axis', 'minor_axis', 'anisotropy', 
                    'eccentricity', 'eccentricity_ctr', 'area', 
                    'rf_theta_deg', 'reliable', 'inbounds']
    df = df.rename(columns={'fx': 'vectorproj_x', 'fy': 'vectorproj_y'})
    if 'cell' not in df.columns:
        df['cell'] = np.arange(0, df.shape[0])
    if 'inbounds' not in df.columns:
        df = check_inbounds(df)
        
    x_cols = [c for c in df.columns if ('x' in c or 'ml' in c) \
                  and (c not in non_cond_cols) and (c not in ignore_cols)]
    y_cols = [c for c in df.columns if ('y' in c or 'ap' in c) \
                  and (c not in non_cond_cols) and (c not in ignore_cols)]

    curr_cols = non_cond_cols.copy()
    curr_cols.extend(x_cols)
    name_lut_x =  combined_axis_colum_names(x_cols)
    x_df = df[curr_cols].rename(columns=name_lut_x)
    x_df['axis'] = 'az'

    curr_cols = non_cond_cols.copy()
    curr_cols.extend(y_cols)
    name_lut_y =  combined_axis_colum_names(y_cols)
    y_df = df[curr_cols].rename(columns=name_lut_y)
    y_df['axis'] = 'el'

    projdf = pd.concat([x_df, y_df], axis=0, ignore_index=True)
    return projdf

# def calculate_scatter(df):
#     '''
#     Calculate scatter (in degrees for VF and distance for CTX). 
#     Redundant with 'aligned_soma' but just creates simplified dataframe that is stacked
#     prev. called get_deviations()
#     '''
#     if 'cell' not in df.columns:
#         df['cell'] = np.arange(0, df.shape[0])
#     if 'inbounds' not in df.columns:
#         df = check_inbounds(df)
#     projdf = stack_axes(df)
# #     df['deg_scatter_x0'] = abs(df['x0']-df['predicted_x0'])
# #     df['deg_scatter_y0'] = abs(df['y0']-df['predicted_y0'])
# #     df['dist_scatter_ml'] = abs(df['ml_proj']-df['predicted_ml_proj'])
# #     df['dist_scatter_ap'] = abs(df['ap_proj']-df['predicted_ap_proj'])
# #     dev_az = df[['cell', 'deg_scatter_x0', 'dist_scatter_ml', 'inbounds', 'reliable']].copy()\
# #                     .rename(columns={'deg_scatter_x0': 'deg_scatter', 
# #                                      'dist_scatter_ml': 'dist_scatter'})
# #     dev_az['axis'] = 'az'

# #     dev_el = df[['cell', 'deg_scatter_y0', 'dist_scatter_ap', 'inbounds']].copy()\
# #                     .rename(columns={'deg_scatter_y0': 'deg_scatter', 
# #                                      'dist_scatter_ap': 'dist_scatter'})
# #     dev_el['axis'] = 'el'
# #     deviations = pd.concat([dev_az, dev_el], axis=0).reset_index(drop=True)

#     #deviations = check_inbounds(deviations)

#     return deviations


#---------------------------------------------------------------------
# MAIN STEPS FOR SCATTER ANALYSIS 
#---------------------------------------------------------------------
def load_vectors(dk, va, create_new=False, 
                rootdir='/n/coxfs01/2p-data'):
    '''
    Load gradient vectors (az and el). If not found or create_new,
    loads gradient results, and saves a gradient_vectors.pkl file. 
    A bit redundant, but faster.

    Returns:

    GVECTORS (dict) 
        Vector shape (2,) for each direction (azimuth and elevation) for specified visual area.

    '''
    GVECTORS=None
    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
    gradients_basedir = os.path.join(fovdir, 'segmentation')
    if not os.path.exists(gradients_basedir):
        os.makedirs(gradients_basedir) 
    gradients_fpath = os.path.join(gradients_basedir, 'gradient_vectors.pkl')
    all_vectors={}
    if not create_new:
        #print("... loading vectors (%s, %s)" % (dk, va))
        try: 
            with open(gradients_fpath, 'rb') as f:
                all_vectors = pkl.load(f, encoding='latin1')
            assert va in all_vectors.keys(),\
                "Area <%s> not in results." % va
            GVECTORS = all_vectors[va].copy()
            assert 'az' in list(GVECTORS.keys()), "grad vectors, found keys: %s" % str(list(GVECTORS.keys()))
        except Exception as e:
            create_new=True
            traceback.print_exc()

    if not isinstance(all_vectors, dict):
        all_vectors={}

    if create_new: 
        #gresults={}
        gpath = gradients_fpath.replace('vectors', 'results')
        with open(gpath, 'rb') as f:
            results = pkl.load(f, encoding='latin1') 
        GVECTORS = {'az': results[va]['az_gradients']['vhat'], 
                    'el': results[va]['el_gradients']['vhat']}
        all_vectors.update({va: GVECTORS})

        with open(gradients_fpath, 'wb') as f:
            pkl.dump(all_vectors, f, protocol=2)
        print("... saved: %s" % gradients_fpath)
 
    return GVECTORS

def load_vectors_and_maps(dk, va, create_new=False):
    '''
    If create_new, re-calculates gradients from saved image (loads gradients_results.pkl).
    
    Returns:

    retinorun: (str)
        Best retino run for specified datakey and visual area (max. mag_thr)
    
    AZMAP_NP: (np.ndarray)
        Smoothed background map for azimuth

    ELMAP_NP: (np.ndarray)
        (same)

    GVECTORS: (dict)
        Gradient vectors for azimuth and elevation

    '''
    retinorun = get_best_retinorun(dk)
    gresults = load_gradients(dk, va, retinorun, create_new=create_new)
    
    AZMAP_NP = gresults['az_gradients']['image']
    ELMAP_NP = gresults['el_gradients']['image']
    GVECTORS = {'az': gresults['az_gradients']['vhat'], 
                'el': gresults['el_gradients']['vhat']}

    return retinorun, AZMAP_NP, ELMAP_NP, GVECTORS



def plot_gradients(dk, va, retinorun, cmap='Spectral'):
    # Gradient plot
    spacing =200
    scale = 0.0001 #0.0001
    width = 0.01 #1 #0.01
    headwidth=20
    contour_lc='w'
    contour_lw=1

    # load
    gresults = load_gradients(dk, va, retinorun, create_new=False)
    grad_az = gresults['az_gradients']
    grad_el = gresults['el_gradients']
    AZMAP_NP = grad_az['image']
    ELMAP_NP = grad_el['image']
    area_mask = gresults['area_mask']

    #### Plot gradients
    fig = seg.plot_gradients_in_area(area_mask, AZMAP_NP, ELMAP_NP, 
                         grad_az, grad_el, cmap_phase=cmap,
                         contour_lc=contour_lc, contour_lw=contour_lw, 
                         spacing=spacing, 
                         scale=scale, width=width, headwidth=headwidth)
    pl.subplots_adjust(left=0.1, right=0.9, bottom=0.2, hspace=0.8, top=0.8)

    return fig

def plot_scatter_on_fov(df_, AZMAP_NP, ELMAP_NP, zimg_r=None, single_axis=True,
                cmap='Spectral', markersize=50, lw=0.6, alpha=1., 
                plot_true=True, plot_predicted=True, plot_lines=True):
    
    '''
    Overlay soma on smoothed background. Visualize scatter for each axis.

    single_axis: (bool)
        Only show scatter along one axis at a time.
    '''
    # Make sure we are in bounds of FOV
    max_ypos, max_xpos = AZMAP_NP.shape
    ap_lim, ml_lim = AZMAP_NP.shape
    incl_plotdf = df_[(df_['predicted_ml_pos']>=0) \
                    & (df_['predicted_ml_pos']<=ml_lim)\
                    & (df_['predicted_ap_proj']>=0) \
                    & (df_['predicted_ap_proj']<=ap_lim)].copy()
    excl_ixs = [i for i in df_.index.tolist() if i not in incl_plotdf.index]
    #plotdf = df_.loc[excl_ixs].copy()
    #plotdf = incl_plotdf.iloc[0::].copy()
    plotdf=df_.copy()

    fig, axn = pl.subplots(1, 2, figsize=(7,5))
    for ax, cond in zip(axn, ['azimuth', 'elevation']):
        neuropil_map = AZMAP_NP.copy() if cond=='azimuth' else ELMAP_NP.copy()
        retino_label='x0' if cond=='azimuth' else 'y0'
        # Set color limits
        vmin = min([np.nanmin(neuropil_map), plotdf[retino_label].min()])
        vmax = max([np.nanmax(neuropil_map), plotdf[retino_label].max()])
        normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        ax.set_title(cond)
        if zimg_r is not None:
            ax.imshow(zimg_r, cmap='gray') #, vmin=abs_vmin, vmax=abs_vmax)
        ax.imshow(neuropil_map, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

        if single_axis:
            pred_x = 'predicted_ml_pos' if cond=='azimuth' else 'ml_pos'
            pred_y = 'predicted_ap_pos' if cond=='elevation' else 'ap_pos'
        else:
            pred_x = 'predicted_ml_pos' 
            pred_y = 'predicted_ap_pos'
        if plot_true:
            # Plot soma
            sns.scatterplot(x='ml_pos', y='ap_pos', data=plotdf, ax=ax,
                    alpha=alpha, hue=retino_label, hue_norm=normalize, 
                    palette=cmap,
                    s=markersize, linewidth=lw, edgecolor='k', zorder=1000) 
        if plot_predicted:
            # Plot soma
            sns.scatterplot(x=pred_x, y=pred_y, 
                    data=plotdf, ax=ax,
                    alpha=alpha, hue=retino_label, hue_norm=normalize, 
                    palette=cmap,
                    s=markersize, linewidth=lw, edgecolor='w', zorder=1000) 
        if plot_lines:
            # Plot connecting line
            for (x1, y1), (x2, y2) in zip(plotdf[[pred_x, \
                                                  pred_y]].values,
                           plotdf[['ml_pos', 'ap_pos']].values):
                ax.plot([x1, x2], [y1, y2], lw=0.5, markersize=0, color='k')
    for ax in axn:
        ax.legend_.remove()
        ax.axis('off')

    return fig


def get_gradient_results(dk, va, return_best=False, 
                    do_gradients=False, do_model=False, plot=True,
                    np_mag_thr=0.02, np_delay_map_thr=1., np_ds_factor=2,
                    cmap='Spectral', plot_dst_dir=None, verbose=False,
                    create_new=False, rootdir='/n/coxfs01/2p-data'):  
    '''
    Do gradient analysis and fit linear model for retino pref. and cortical position.
    Uses NEUROPIL preferences (assumes RF tiling experiment) to calculate global background.

    Args:
    do_gradients: (bool)
        Re-calculate gradients (first step, return GVECTORS).
    do_model: (bool)
        From GVECTORS, fit linear model to adjusted/aligned background.
    create_new: (bool)
        Completely overwrite ALL found gradient results
    '''
    print("Do model?", do_model)

    retinorun, AZMAP_NP, ELMAP_NP, GVECTORS, REGR_NP = None, None, None, None, None

    if do_gradients or do_model:
        plot=True

    if plot_dst_dir is None:
        session, animalid, fovn = hutils.split_datakey_str(dk)
        fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
        plot_dst_dir = os.path.join(fovdir, 'segmentation')

    if plot:
        if not os.path.exists(plot_dst_dir):
            os.makedirs(plot_dst_dir)

    #### Load NEUROPIL BACKGROUND and GRADIENTS
    print("... loading gradient vectors (%s, %s)" % (dk, va))
    retinorun, AZMAP_NP, ELMAP_NP, GVECTORS = load_vectors_and_maps(dk, va, create_new=do_gradients)
    if plot:
        fig = plot_gradients(dk, va, retinorun, cmap=cmap)
        fig.text(0.05, 0.95, 'Gradients, est. from MOVINGBAR\n(%s, %s)' % (dk, va), fontsize=8)
        pl.savefig(os.path.join(plot_dst_dir, 'pixelmap_gradients_%s.svg' % va))
        #pl.savefig(os.path.join(plot_dst_dir, 'np_gradients.svg'))
        pl.close()

    #### Use NEUROPIL to estimate linear model
    if not do_model:
        try:
            REGR_NP = load_models(dk, va, return_best=return_best, rootdir=rootdir)
            assert REGR_NP is not None, "NO REGR"
        except Exception as e:
            traceback.print_exc()
            do_model=True
 
    #print("REGR", REGR_NP)
    if do_model:
        print("... estimating linear fit (%s, %s)" % (dk, va))
        # Align NP to gradient vectors in current visual area
        aligned_np, regr_np_post, regr_np_meas = transform_and_fit_pixels(dk, va, retinorun, 
                                                            GVECTORS, create_new=True)
#        aligned_np, regr_np_post, regr_np_meas = transform_and_fit_neuropil(dk, va, retinorun,
#                                                GVECTORS,
#                                                mag_thr=np_mag_thr, 
#                                                delay_map_thr=np_delay_map_thr, 
#                                                ds_factor=np_ds_factor) 
        # Save
        regr_np_meas['aligned'] = False
        regr_np_post['aligned'] = True
        REGR_ = pd.concat([regr_np_post, regr_np_meas])
        update_models(dk, va, REGR_) #, create_new=create_new)
        if aligned_np is not None:
            if verbose:
                print("NEUROPIL, MEASURED:")
                print(regr_np_meas.to_markdown())
                print("NEUROPIL, ALIGNED:")
                print(regr_np_post.to_markdown())
            fig = plot_pre_and_post_pixel_alignment(aligned_np, regr_np_post, regr_np_meas)
            fig.text(0.05, 0.95, 'Pre/Post-aligned pixel map correlation\n[%s] %s, %s' \
                            % (va, dk, retinorun), fontsize=8)
            # save
            figname = 'pixelmap_pre_post_alignment_%s' % va
            pl.savefig(os.path.join(plot_dst_dir, '%s.svg' % figname))
            pl.close()

#            fig = plot_measured_and_aligned(aligned_np, regr_np_post, regr_np_meas)
#            fig.text(0.01, 0.95, 'Aligned CTX. vs retino\n(BAR, Neuropil, %s)' % dk)
#            figname = 'measured_vs_aligned_NEUROPIL'
#            pl.savefig(os.path.join(plot_dst_dir, '%s.svg' % figname))
#            pl.close()

        if return_best:
            pre = float(regr_np_meas['R2'].mean())
            post = float(regr_np_post['R2'].mean())
            if pre>post:
                REGR_NP = regr_np_meas.copy()
            else:
                REGR_NP = regr_np_post.copy()
        else:
            REGR_NP = regr_np_post.copy() 


    return retinorun, AZMAP_NP, ELMAP_NP, GVECTORS, REGR_NP



def transform_and_fit_pixels(dk, va, retinorun, GVECTORS, create_new=False,
                    rootdir='/n/coxfs01/2p-data'):

    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, \
                    'FOV%i_*' % fovn))[0]
    dst_dir = os.path.join(fovdir, 'segmentation')
    alignment_file = os.path.join(dst_dir, 'projected_pixels.pkl')

    res={}
    if not create_new:
        try:
            with open(alignment_file, 'rb') as f:
                res = pkl.load(f, encoding='latin1')
            aligned_pix = res[va]['aligned_pixels']
            regr_proj = res[va]['regr_projected']
            regr_meas = res[va]['regr_measured']
        except Exception as e:
            # traceback.print_exc()
            print("    error loading alignment results (%s, %s)" %(dk, va))
            create_new=True

    if create_new: 
        # Get smoothed background maps for area
        AZMAP_NP, ELMAP_NP = load_pixel_maps(dk, va, create_new=False, map_type='final')

        # Get coordinates of each pixel
        d1, d2 = ELMAP_NP.shape
        Y, X = np.mgrid[0:d1, 0:d2]
        positions = np.array([(i, j) for i, j in zip(X.ravel(),Y.ravel())])
        assert d1*d2 == len(positions)
        # Create dataframe of pixel coords
        pixel_pos = pd.DataFrame({'ml_pos': positions[:, 0], 'ap_pos': positions[:, 1]})
        # Project pixels onto gradient vector
        pixel_proj, M = project_onto_gradient(pixel_pos, GVECTORS)
        # Combine true and projected pixels
        pixel_proj['ml_proj_ix'] = [d2+int(round(i)) if i<0 else int(round(i))\
                                    for i in pixel_proj['ml_proj']]
        pixel_proj['ap_proj_ix'] = [d1+int(round(i)) if i<0 else int(round(i))\
                                    for i in pixel_proj['ap_proj']]
        pix_df = pd.concat([pixel_proj, pixel_pos], axis=1)

        # Get retinomap values for pixels
        retino_val = pd.DataFrame({'x0': [AZMAP_NP[j, i] for (i,j) in positions],
                                   'y0': [ELMAP_NP[j, i] for (i, j) in positions]})
        aligned_pix = pd.merge(retino_val, pix_df, left_index=True, right_index=True)

        # do linear fit
        regr_meas = regress_cortex_and_retino_pos(aligned_pix.dropna(),
                                          xvar='pos', model='ridge')
        regr_proj = regress_cortex_and_retino_pos(aligned_pix.dropna(),
                                          xvar='proj', model='ridge')
        # Get predictions
        aligned_pix = predict_positions(aligned_pix, M, regr_proj)
        # Convert to indices
        aligned_pix['ml_proj_ix'] = [i if np.isnan(i) else int(round(i))\
                                    for i in aligned_pix['ml_proj'].values]
        aligned_pix['ap_proj_ix'] = [i if np.isnan(i) else int(round(i))\
                                    for i in aligned_pix['ap_proj'].values]
        aligned_pix['predicted_ml_pos_ix'] = [d2+int(round(i)) if i<0 else i\
                                     for i in aligned_pix['predicted_ml_pos']]
        aligned_pix['predicted_ap_pos_ix'] = [d1+int(round(i)) if i<0 else i\
                                     for i in aligned_pix['predicted_ap_pos']]

        curr_res = {'aligned_pixels': aligned_pix, 
               'regr_projected': regr_proj, 'regr_measured': regr_meas}

        res.update({va: curr_res})
        with open(alignment_file, 'wb') as f:
            pkl.dump(res, f, protocol=2)

    return aligned_pix, regr_proj, regr_meas


def plot_pre_and_post_pixel_alignment(aligned_pix, regr_proj, regr_meas,
                                    lc='m', lw=1):

    fig, axn = pl.subplots(2,2, figsize=(5,5))
    # plot measured
    ax=axn[0,0]
    ax.set_title('Unaligned', loc='left')
    ax = plot_linear_fit_and_scatter(aligned_pix, regr_meas, ax=ax,
                                     x_var='ml_pos', y_var='x0', lc=lc, lw=lw)
    ax=axn[0,1]
    ax = plot_linear_fit_and_scatter(aligned_pix, regr_meas, ax=ax,
                                     x_var='ap_pos', y_var='y0', lc=lc, lw=lw)
    # plot fit
    ax=axn[1,0]
    ax.set_title('After alignment', loc='left')
    ax = plot_linear_fit_and_scatter(aligned_pix, regr_proj, ax=ax,
                                     x_var='ml_proj', y_var='x0', lc=lc, lw=lw)
    ax=axn[1,1]
    ax = plot_linear_fit_and_scatter(aligned_pix, regr_proj, ax=ax,
                                     x_var='ap_proj', y_var='y0', lc=lc, lw=lw)
    pl.subplots_adjust(left=0.1, right=0.9, wspace=0.5, hspace=0.5)

    #pplot.label_figure(fig, data_id)
    return fig



def transform_and_fit_neuropil(dk, va, retinorun, GVECTORS,abs_value=False,
                        mag_thr=0.001, delay_map_thr=1.0, ds_factor=2):
    '''
    Align NEUROPIL retino preferences (each ROI's neuropil) to gradient vectors
    calculated for curent visual area. 
    prev. called 'get_aligned_neuropil'
    
    Returns:

    aligned_np: (pd.DataFrame)
        Receptive field fit params for each cell's neuropil, plus ml/ap_proj coordinates

    REGR_NP: (pd.DataFrame)
        Results of linear regression (x0 vs. ml_proj, y0 vs. ap_proj)

    regr_np_meas (pd.DataFrame)
        Same as REGR_NP, but without correcting/aligning FOV to gradient direction.

    '''
    aligned_np, REGR_NP, regr_np_meas = None, None, None

    # 1. Get retino data for NEUROPIL (background)
    retinodf_np = load_neuropil_data(dk, retinorun, mag_thr=mag_thr, 
                                        delay_map_thr=delay_map_thr, 
                                        ds_factor=ds_factor)

    if retinodf_np is None:
        return None

    # Only select ROIs within current visual_area:
    curr_np = retinodf_np[retinodf_np.visual_area==va].copy()

    # 2. Align FOV to gradient vector direction 
    aligned_, M = project_onto_gradient(curr_np, GVECTORS,
                                      xlabel='ml_pos', ylabel='ap_pos', abs_value=abs_value)
    aligned_np = pd.concat([curr_np, aligned_], axis=1).dropna()

    # 3. Fit model
    REGR_NP = regress_cortex_and_retino_pos(aligned_np, \
                    xvar='proj', model='ridge')
    regr_np_meas = regress_cortex_and_retino_pos(aligned_np, \
                    xvar='pos', model='ridge')


    return aligned_np, REGR_NP, regr_np_meas


def predict_soma_from_gradient(dk, va, REGR_NP, experiment='rfs',
                     traceid='traces001', protocol='TILE',
                    response_type='dff', do_spherical_correction=False,
                    ecc_center=(0, 0), abs_value=False,
                    verbose=False, plot=False, plot_dst_dir='/tmp'):
    '''
    Using gradient vectors, project FOV coords along retino gradients.
    If protocol=='TILE', then will also calculate deg/dist scatter.
    Prev. called:  get_aligned_soma()
    '''
    # Get soma fits (aligned to gradient) 
    aligned_soma, M = project_soma_position_in_fov(dk, va, experiment=experiment, 
                            traceid=traceid, 
                            response_type=response_type, 
                            do_spherical_correction=do_spherical_correction,
                            return_transformation=True, ecc_center=ecc_center, 
                            abs_value=abs_value)
    if aligned_soma.shape[0]<2:
        return None

    # Do linear fit (proj should be better) 
    regr_soma_meas = regress_cortex_and_retino_pos(aligned_soma, 
                                                       xvar='pos', model='ridge')
    regr_soma_proj = regress_cortex_and_retino_pos(aligned_soma, 
                                                        xvar='proj', model='ridge')
    if verbose:
        print("SOMA, MEASURED:")
        print(regr_soma_meas.to_markdown())
        print("SOMA, ALIGNED:")
        print(regr_soma_proj.to_markdown())
    if plot:
        # PLOT, soma
        fig = plot_measured_and_aligned(aligned_soma, 
                            regr_soma_proj, regr_soma_meas)
        fig.text(0.01, 0.95, 'Measured vs Aligned CTX to RETINO pos (%s)' % dk)
        figname = 'measured_vs_aligned_SOMA'
        pl.savefig(os.path.join(plot_dst_dir, '%s.svg' % figname))
        pl.close()

    # Predict positions and calculate scatter
    aligned_soma = predict_positions(aligned_soma, M, REGR_NP)

    # Check inbounds
    aligned_soma = check_inbounds(aligned_soma)
        
    return aligned_soma.reset_index(drop=True)


def predict_positions(aligned_soma, M, REGR_NP):
    '''
    Predict cortical and retino positions, based on the other (using REGR_NP).
    M is the transformation matrix (apply_projection())

    '''
    #### Predict CORTICAL position (from retino position)
    p_x = predict_cortex_position(REGR_NP, cond='az', 
                              points=aligned_soma['x0'].values)
    p_y = predict_cortex_position(REGR_NP, cond='el', 
                              points=aligned_soma['y0'].values)
    aligned_soma['predicted_ml_proj'] = p_x
    aligned_soma['predicted_ap_proj'] = p_y

    #### Predict RETINO position (from cortical position)
    r_x = predict_retino_position(REGR_NP, cond='az', 
                              points=aligned_soma['ml_proj'].values)
    r_y = predict_retino_position(REGR_NP, cond='el', 
                              points=aligned_soma['ap_proj'].values)
    aligned_soma['predicted_x0'] = r_x
    aligned_soma['predicted_y0'] = r_y

    #### Calculate inverse for visualizing on FOV
    pred_INV = [np.linalg.inv(M).dot(np.array([x, y])) for (x, y) \
                in aligned_soma[['predicted_ml_proj', 'predicted_ap_proj']].values]
    pred_inv_df = pd.DataFrame(pred_INV, columns=['pred_inv_x', 'pred_inv_y'], 
                          index=aligned_soma.index)
    aligned_soma['predicted_ml_pos'] = pred_inv_df['pred_inv_x']
    aligned_soma['predicted_ap_pos'] = pred_inv_df['pred_inv_y']

      # Calculate scatter
    if 'x0' in aligned_soma.keys():
        aligned_soma.loc[:, 'deg_scatter_x0'] = aligned_soma['x0']-aligned_soma['predicted_x0']
        aligned_soma.loc[:, 'deg_scatter_y0'] = aligned_soma['y0']-aligned_soma['predicted_y0']

    aligned_soma.loc[:, 'dist_scatter_ml'] = abs(aligned_soma['ml_proj']-aligned_soma['predicted_ml_proj'])
    aligned_soma.loc[:, 'dist_scatter_ap'] = abs(aligned_soma['ap_proj']-aligned_soma['predicted_ap_proj'])
    
    return aligned_soma


def overlay_scatter(dk, va, df_, AZMAP_NP, ELMAP_NP, experiment='rfs',
                     traceid='traces001', single_axis=True,
                    markersize=50, lw=0.5, alpha=1, cmap='Spectral', 
                    plot_true=True, plot_predicted=True, plot_lines=True,
                    return_fig=False, plot_dst_dir=None, data_id=None):
    '''
    Scatter overlay on top of smoothed background.
    Prev. called do_visualization()
    '''
    if data_id is None:
        data_id = '%s|%s|%s, %s' % (traceid, va, dk, experiment)
    # # Visualization
    zimg, masks, ctrs = roiutils.get_masks_and_centroids(dk, traceid=traceid)
    pixel_size = hutils.get_pixel_size()
    zimg_r = retutils.transform_2p_fov(zimg, pixel_size)
    fig = plot_scatter_on_fov(df_, AZMAP_NP, ELMAP_NP, zimg_r=zimg_r,
                    cmap=cmap, markersize=markersize, lw=lw, alpha=alpha,
                    plot_true=plot_true, plot_predicted=plot_predicted, 
                    plot_lines=plot_lines, single_axis=single_axis)
    fig.text(0.01, 0.95, 'CTX vs RETINO positions - MEASURED (%s)' % dk)
    pplot.label_figure(fig, data_id)

    if plot_dst_dir is not None:
        pl.savefig(os.path.join(plot_dst_dir, 'fov_true_v_predicted_scatter_%s.svg' % va))

    if return_fig:
        return fig
    else:
        pl.close()
        return



def do_scatter_analysis(dk, va, experiment='rfs', 
                        do_gradients=False, do_model=False,
                        np_mag_thr=0.001, np_delay_map_thr=1.0, 
                        np_ds_factor=2., return_best_model=False,
                        response_type='dff', do_spherical_correction=False, 
                        ecc_center=(0, 0), abs_value=False,
                        traceid='traces001',
                        cmap='Spectral', plot=True,
                        rootdir='/n/coxfs01/2p-data', verbose=False,
                        create_new=False):
    '''
    create_new to completely overwrite scatter analysis results
    '''
    scatter_df=None
    scatter_kws={'s':2}


    if not create_new:
        try:
            scatter_df = load_scatter_results(dk, va, experiment=experiment,verbose=verbose)
        except Exception as e:
            # traceback.print_exc()
            create_new=True

    if create_new:
        #### Select output dirs
        session, animalid, fovn = hutils.split_datakey_str(dk)
        fovdir = glob.glob(os.path.join(rootdir, animalid, session, \
                        'FOV%i_*' % fovn))[0]
        curr_dst_dir = os.path.join(fovdir, 'segmentation')
        if not os.path.exists(curr_dst_dir):
            os.makedirs(curr_dst_dir)

        fit_desc = rfutils.get_fit_desc(response_type=response_type,
                                        do_spherical_correction=do_spherical_correction)
        data_id = '%s|%s|%s, %s' % (fit_desc, va, dk, experiment)
        has_retino=True
        has_rfs=True
        try:
            retinorun, AZMAP_NP, ELMAP_NP, GVECTORS, REGR_NP = get_gradient_results(dk, va, 
                                    return_best=return_best_model,
                                    do_gradients=do_gradients, do_model=do_model, 
                                    np_mag_thr=np_mag_thr, 
                                    np_delay_map_thr=np_delay_map_thr, 
                                    np_ds_factor=np_ds_factor,
                                    plot=plot, cmap=cmap, plot_dst_dir=curr_dst_dir,
                                    verbose=verbose)  
            if REGR_NP is None:
                has_retino=False

            if has_retino:
                protocol = 'TILE' if 'rfs' in experiment else 'BAR'
                soma_dst_dir = os.path.join(curr_dst_dir, 'scatter_%s' % experiment)
                if not os.path.exists(soma_dst_dir):
                    os.makedirs(soma_dst_dir)
                aligned_soma = predict_soma_from_gradient(dk, va, REGR_NP, 
                                        experiment=experiment,
                                        traceid=traceid, protocol=protocol, 
                                        response_type=response_type,
                                        do_spherical_correction=do_spherical_correction,
                                        ecc_center=ecc_center,  abs_value=abs_value,
                                        verbose=verbose, plot=plot, 
                                        plot_dst_dir=soma_dst_dir)
                if aligned_soma is None:
                    has_rfs=False
                    
                if has_rfs and plot:
                    # FOV scatter coords 
                    overlay_scatter(dk, va, aligned_soma, AZMAP_NP, ELMAP_NP, 
                                    experiment=experiment, traceid=traceid, 
                                    markersize=30, lw=0.5, alpha=1, cmap=cmap, 
                                    single_axis=True,
                                    plot_true=True, plot_predicted=True, plot_lines=True,
                                    plot_dst_dir=soma_dst_dir, data_id=data_id)
                if has_rfs:
                    # Calculate scatter
                    #print("... calculating deviations")
                    #scatter_df = calculate_scatter(aligned_soma)
                    scatter_df = stack_axes(aligned_soma)
                if has_rfs and plot:
                    # Plot
                    fig, axn = pl.subplots(1,2, figsize=(6.5, 3))
                    ax=axn[0]
                    sns.histplot(scatter_df, x='deg_scatter', hue='axis', ax=ax,
                                stat='probability', cumulative=False )
                    ax.set_title('visual field scatter (deg)')
                    ax=axn[1]
                    sns.histplot(scatter_df, x='dist_scatter', hue='axis', ax=ax,
                                stat='probability', cumulative=False)
                    ax.set_title('cortical scatter (um)')
                    pl.subplots_adjust(left=0.1, right=0.8, bottom=0.25, top=0.85, 
                                        wspace=0.5, hspace=0.5)
                    if os.path.exists(os.path.join(soma_dst_dir, 'deviations.svg')):
                        os.remove(os.path.join(soma_dst_dir, 'deviations.svg'))
                    pplot.label_figure(fig, data_id)
                    pl.savefig(os.path.join(soma_dst_dir, 'scatter_%s.svg' % va))
                    pl.close()
                # save                
                update_scatter_results(dk, va, scatter_df, soma_dst_dir)

        except Exception as e:
            print("ERROR in %s, %s (%s)" % (dk, va, experiment))
            traceback.print_exc()

        if not has_retino:
            print("ERROR: no retino")
        if not has_rfs:
            print("ERROR: no rfs")
      
    return scatter_df

def update_scatter_results(dk, va, soma_results, soma_dst_dir): #, create_new=False):
    scatter_fpath = os.path.join(soma_dst_dir, 'scatter_results.pkl')

    results={}
    if os.path.exists(scatter_fpath): #and (create_new is False):

        with open(scatter_fpath, 'rb') as f:
            results = pkl.load(f, encoding='latin1')
        if not isinstance(results, dict):
            results = {}

    results.update({va: soma_results})
    with open(scatter_fpath, 'wb') as f:
        pkl.dump(results, f, protocol=2)

    return

       
def load_scatter_results(dk, va, experiment='rfs', verbose=False,
                    rootdir='/n/coxfs01/2p-data'):
    '''
    Loads dataframe: 'cell', 'deg_scatter', 'dist_scatter', 'axis'.
    Faster (but otherwise, is same data, but with calcualtions as project_soma_position_in_fov)

     soma_dst_dir = os.path.join(curr_dst_dir, 'scatter_%s' % experiment)
    '''
    currdf=None
    try:
        session, animalid, fovn = hutils.split_datakey_str(dk)
        fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
        soma_dst_dir = os.path.join(fovdir, 'segmentation', 'scatter_%s' % experiment)
        results_fpath = os.path.join(soma_dst_dir, 'scatter_results.pkl')
        with open(results_fpath, 'rb') as f:
            results = pkl.load(f, encoding='latin1')
        assert va in results.keys(), 'Visual area not found in scatter analysis'
        currdf = results[va].copy()
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return None
    
    return currdf


def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', 
                      default='/n/coxfs01/2p-data', 
                      help='root dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-G', '--aggr', action='store', dest='aggregate_dir', 
                      default='/n/coxfs01/julianarhee/aggregate-visual-areas', 
                      help='aggregate analysis dir [default: aggregate-visual-areas]')
    parser.add_option('--zoom', action='store', dest='fov_type', default='zoom2p0x', 
                      help="fov type (zoom2p0x)") 
    parser.add_option('--state', action='store', dest='state', default='awake', 
                      help="animal state (awake)") 
  
    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default=None, 
                      help="experiment to calculate scatter (e.g,. rfs, rfs10, retino), set None to cycycle thru all") 
    parser.add_option('-i', '--datakey', action='store', dest='datakey', default='', 
                      help='datakey (YYYYMMDD_JCxx_fov1)')
    parser.add_option('-V', '--area', action='store', dest='visual_area', default=None, 
                      help='visual area to process (None, to do all)')

    parser.add_option('-t', '--traceid', action='store', dest='traceid', 
                      default='traces001', \
                      help="name of traces ID [default: traces001]")
       
    parser.add_option('-M', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")
    parser.add_option('--sphere', action='store_true', 
                        dest='do_spherical_correction', default=False, 
                        help="Flag to do fit on spherically-corrected response arrays")
    parser.add_option('--rf-thr', action='store', 
                        dest='rf_fit_thr', default=0.5, 
                        help="Fit thr for RF fits (default: 0.5)")
    # Neuropil 
    parser.add_option('--np-mag', action='store', 
                        dest='np_mag_thr', default=0.001,
                        help="Mag thr for neuropil retino (default: 0.001)")
    parser.add_option('--np-delay', action='store', 
                        dest='np_delay_map_thr', default=1., 
                        help="Delay map thr for neuropil retino (default: 1)")
    parser.add_option('--np-downsample', action='store', 
                        dest='np_ds_factor', default=2., 
                        help="Downsample factor for retino maps (default: 2)")

    parser.add_option('--gradients',  action='store_true', dest='do_gradients', default=False,
                      help="Recalculate gradients from NP image")
    parser.add_option('--model',  action='store_true', dest='do_model', default=False,
                      help="Refit model for retino-pos and ctx-pos on NP image")
    parser.add_option('--new',  action='store_true', dest='create_new', default=False,
                      help="Create new scatter_results file")


    parser.add_option('--plot',  action='store_true', dest='plot', default=False,
                      help="plot and save figures")
    parser.add_option('--cmap',  action='store', dest='cmap', default='Spectral',
                      help="Colormap to use for background img (default: Spectral)")
    parser.add_option('-v', '--verbose',  action='store_true', dest='verbose', default=False,
                      help="verbose")


    parser.add_option('--all',  action='store_true', dest='cycle_all', default=False,
                      help="Set flag to cycle thru ALL dsets")
    parser.add_option('--best',  action='store_true', dest='return_best_model', default=False,
                      help="Set to return BEST linear model (i.e., don't use ALIGNED regr if it is worse")

    (options, args) = parser.parse_args(options)

    return options



#### Select dataset and create output dirs
#dk = '20190617_JC097_fov1'
#va = 'V1'

#### RFs, Select parameters
#fit_thr=0.5
## NP background calc.
#np_mag_thr=0.001
#np_delay_map_thr=1.0
#np_ds_factor=2
#
#do_gradients=False
#do_model=False
#plot=True
#
#### Some plotting stuff
#cmap ='Spectral'

def main(options):

    optsE = extract_options(options)
    
    rootdir = optsE.rootdir
    datakey = optsE.datakey
    traceid = optsE.traceid
    
    dk = optsE.datakey
    va = None if optsE.visual_area in [None, 'None'] else optsE.visual_area
    experiment = optsE.experiment
    cmap= optsE.cmap

    # fit params
    create_new= optsE.create_new
    do_gradients = optsE.do_gradients
    do_model = optsE.do_model
    plot = optsE.plot
    if do_gradients:
        #do_gradients=True
        do_model=True
        plot=True
    
    # RF params
    response_type = optsE.response_type
    do_spherical_correction = optsE.do_spherical_correction
   
    # NP background
    np_mag_thr = float(optsE.np_mag_thr) 
    np_delay_map_thr = float(optsE.np_delay_map_thr) 
    np_ds_factor = float(optsE.np_ds_factor) 

    verbose = optsE.verbose
    cycle_all = optsE.cycle_all

    return_best_model = optsE.return_best_model

    if experiment is None:
        exp_list=['rfs', 'rfs10']
    else:     
        exp_list = [experiment]

    if cycle_all:
        sdata = aggr.get_aggregate_info(visual_areas=['V1', 'Lm', 'Li'], 
                                        return_cells=False)
        if va is not None:
            meta = sdata[(sdata.experiment.isin(exp_list)) & (sdata.visual_area==va)]
        else:
            meta = sdata[sdata.experiment.isin(exp_list)]
        for (va, dk, experiment), g in meta.groupby(['visual_area', 'datakey', 'experiment']):
            print("Area: %s - %s (%s) " % (va, dk, experiment))
            deviants = do_scatter_analysis(dk, va, experiment=experiment, 
                            return_best_model=return_best_model,
                            do_gradients=do_gradients, do_model=do_model,
                            np_mag_thr=np_mag_thr, np_delay_map_thr=np_delay_map_thr, 
                            np_ds_factor=np_ds_factor,
                            response_type=response_type, 
                            do_spherical_correction=do_spherical_correction,
                            traceid=traceid,
                            cmap=cmap, plot=plot, verbose=verbose, create_new=create_new)
    else:
        assert dk is not None, "Must specify datakey" 
        sdata = aggr.get_aggregate_info(visual_areas=['V1', 'Lm', 'Li'], 
                                        return_cells=False)
        if (va is None):
            meta = sdata[(sdata.datakey==dk) & (sdata.experiment.isin(exp_list))].drop_duplicates()
        else:
            meta = sdata[(sdata.datakey==dk) & (sdata.experiment.isin(exp_list)) \
                       & (sdata.visual_area==va)].drop_duplicates()
        for (va, experiment), g in meta.groupby(['visual_area', 'experiment']):
            print("Processing: %s, %s" % (va, dk))
            deviants = do_scatter_analysis(dk, va, experiment=experiment,
                            return_best_model=return_best_model,
                            traceid=traceid,
                            do_gradients=do_gradients, do_model=do_model,
                            np_mag_thr=np_mag_thr, np_delay_map_thr=np_delay_map_thr, 
                            np_ds_factor=np_ds_factor,
                            response_type=response_type, 
                            do_spherical_correction=do_spherical_correction,
                            cmap=cmap, plot=plot, verbose=verbose, create_new=create_new)
            if deviants is not None:
                print("    N deviants:", deviants.shape)

    print("Done.")

if __name__ == '__main__':

    main(sys.argv[1:])


#def get_transformation_matrix(u1, u2):
#    '''this is not the one to use'''
#    a, b = u1[0], u1[1]
#    c, d = u2[0], u2[1]
#
#    a_21 = 1./ ( (-a*d/b) + c)
#    a_11 = (-d*b) * a_21
#
#    a_22 = 1./ ((-c*b/a)+d)
#    a_12 = (-c*a) * a_22
#
#    M = np.array( [ [a_11, a_12], [a_21, a_22] ])
#    
#    return M
#
#def X_align_cortex_to_gradient(df, gvectors, xlabel='ml_pos', ylabel='ap_pos'):
#    '''
#    Align FOV to gradient vector direction w transformation matrix.
#    Use gvectors to align coordinates specified in df.
#    Note: calculate separate for each axis.
#    
#    gvectors: dict()
#        keys/vals: 'az': [v1, v2], 'el': [w1, w2]
#    df: pd.DataFrame()
#        coordi
#    '''
#    # Transform predicted-ctx pos back to FOV coords
#    u1 = (gvectors['az'])
#    u2 = (gvectors['el'])
#    # Cartesian normal (plus error)
#    o1 = np.array([1, 0]) + np.finfo(np.float32).eps
#    o2 = np.array([0, 1]) + np.finfo(np.float32).eps
#    # x-axis -- transform FOV coords to lie alone gradient axis
#    T1 = roiutils.get_transformation_matrix(u1, o2)
#    transf_x = [T1.dot(np.array([x, y])) for (x,y) \
#                        in df[[xlabel, ylabel]].values]
#    proj_ctx_x = np.array([p[0] for p in transf_x])
#    # y-axis -- same
#    T2 = roiutils.get_transformation_matrix(o1, u2)
#    transf_y = [T2.dot(np.array([x, y])) for (x,y) \
#                        in df[[xlabel, ylabel]].values]
#    proj_ctx_y = np.array([p[1] for p in transf_y])
#    # rename
#    new_xlabel = '%s_proj' % xlabel.split('_')[0]
#    new_ylabel = '%s_proj' % ylabel.split('_')[0]
#    df_ = pd.DataFrame({new_xlabel: proj_ctx_x, 
#                        new_ylabel: proj_ctx_y}, index=df.index)
#    
#    return df_, T1, T2
