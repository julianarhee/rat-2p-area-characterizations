#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import seaborn as sns
import pylab as pl
import matplotlib as mpl
import scipy.stats as spstats

import analyze2p.utils as hutils
import analyze2p.plotting as pplot
import analyze2p.receptive_fields.utils as rfutils
import analyze2p.aggregate_datasets as aggr
import analyze2p.extraction.rois as roiutils
import analyze2p.retinotopy.utils as retutils
import analyze2p.retinotopy.segment as seg
import analyze2p.gradients as grd


# In[3]:
def load_vectors(dk, va, create_new=False):
    retinorun = grd.get_best_retinorun(dk)
    gresults = grd.load_gradients(dk, va, retinorun, create_new=create_new)
    
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
    gresults = grd.load_gradients(dk, va, retinorun, create_new=False)
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


def visualize_scatter_on_fov(df_, AZMAP_NP, ELMAP_NP, zimg_r=None,
                cmap='Spectral', markersize=50, lw=0.6, alpha=1., 
                plot_true=True, plot_predicted=True, plot_lines=True):

    # Make sure we are in bounds of FOV
    max_ypos, max_xpos = AZMAP_NP.shape
    incl_plotdf = df_[(df_['predicted_ml_pos']>=0) \
                    & (df_['predicted_ml_pos']<=max_xpos)\
                    & (df_['predicted_ap_proj']>=0) \
                    & (df_['predicted_ap_proj']<=max_ypos)].copy()
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
        if plot_true:
            # Plot soma
            sns.scatterplot(x='ml_pos', y='ap_pos', data=plotdf, ax=ax,
                    alpha=alpha, hue=retino_label, hue_norm=normalize, 
                    palette=cmap,
                    s=markersize, linewidth=lw, edgecolor='k', zorder=1000) 
        if plot_predicted:
            # Plot soma
            sns.scatterplot(x='predicted_ml_pos', y='predicted_ap_pos', 
                    data=plotdf, ax=ax,
                    alpha=alpha, hue=retino_label, hue_norm=normalize, 
                    palette=cmap,
                    s=markersize, linewidth=lw, edgecolor='w', zorder=1000) 
        if plot_lines:
            # Plot connecting line
            for (x1, y1), (x2, y2) in zip(plotdf[['predicted_ml_pos', \
                                                  'predicted_ap_pos']].values,
                           plotdf[['ml_pos', 'ap_pos']].values):
                ax.plot([x1, x2], [y1, y2], lw=0.5, markersize=0, color='k')
    for ax in axn:
        ax.legend_.remove()
        ax.axis('off')

    return fig


def get_gradient_results(dk, va, do_gradients=False, do_model=False, plot=True,
                    np_mag_thr=0.001, np_delay_map_thr=1., np_ds_factor=2,
                    cmap='Spectral', plot_dst_dir='/tmp', verbose=False,
                    create_new=False):  
    '''
    create_new to completely overwrite ALL found gradient results
    '''
    if plot:
        if not os.path.exists(plot_dst_dir):
            os.makedirs(plot_dst_dir)

    #### Load NEUROPIL BACKGROUND and GRADIENTS
    retinorun, AZMAP_NP, ELMAP_NP, GVECTORS = load_vectors(dk, va, create_new=do_gradients)
    if plot:
        fig = plot_gradients(dk, va, retinorun, cmap=cmap)
        fig.text(0.05, 0.95, 'Gradients, est. from MOVINGBAR (%s)' % dk)
        pl.savefig(os.path.join(plot_dst_dir, 'np_gradients.svg'))
        pl.close()

    #### Use NEUROPIL to estimate linear model
    try:
        REGR_NP = grd.load_models(dk, va, rootdir=rootdir)
        assert REGR_NP is not None
    except Exception as e:
        plot=True
        do_model=True

    if do_model:
        # 1. Get retino data for NEUROPIL (background)
        retinodf_np = grd.get_neuropil_data(dk, retinorun, mag_thr=np_mag_thr, 
                                            delay_map_thr=np_delay_map_thr, 
                                            ds_factor=np_ds_factor)
        assert retinodf_np is not None, 'ERROR: %s, %s - no df' % (dk, retinorun)

        # 2. Align FOV to gradient vector direction 
        aligned_, M = grd.align_cortex_to_gradient(retinodf_np, GVECTORS,
                                          xlabel='ml_pos', ylabel='ap_pos')
        aligned_np = pd.concat([retinodf_np, aligned_], axis=1).dropna()
        # 3. Fit model
        REGR_NP = grd.regress_cortex_and_retino_pos(aligned_np, \
                        xvar='proj', model='ridge')
        regr_np_meas = grd.regress_cortex_and_retino_pos(aligned_np, \
                        xvar='pos', model='ridge')
        # Save
        grd.update_models(dk, va, REGR_NP, create_new=create_new)
        if verbose:
            print("NEUROPIL, MEASURED:")
            print(regr_np_meas.to_markdown())
            print("NEUROPIL, ALIGNED:")
            print(REGR_NP.to_markdown())
        fig = grd.plot_measured_and_aligned(aligned_np, REGR_NP, regr_np_meas)
        fig.text(0.01, 0.95, 'Aligned CTX. vs retino\n(BAR, Neuropil, %s)' % dk)
        figname = 'measured_vs_aligned_NEUROPIL'
        pl.savefig(os.path.join(plot_dst_dir, '%s.svg' % figname))
        pl.close()

    return retinorun, AZMAP_NP, ELMAP_NP, GVECTORS, REGR_NP

def get_aligned_soma(dk, retinorun, GVECTORS, REGR_NP, experiment='rfs',
                     traceid='traces001', protocol='TILE',
                    response_type='dff', do_spherical_correction=False,
                    verbose=False, plot=False, plot_dst_dir='/tmp'):
    #### Load soma
    df_soma = grd.get_soma_data(dk, experiment=experiment, retinorun=retinorun, 
                                protocol=protocol, traceid=traceid,
                                response_type=response_type,
                                do_spherical_correction=do_spherical_correction)
    #### Align soma coords to gradient
    aligned_, M = grd.align_cortex_to_gradient(df_soma, GVECTORS,
                                      xlabel='ml_pos', ylabel='ap_pos')
    aligned_soma = pd.concat([df_soma, aligned_], axis=1).dropna()\
                        .reset_index(drop=True)
    #### Align SOMA coords
    regr_soma_meas = grd.regress_cortex_and_retino_pos(aligned_soma, 
                                                       xvar='pos', model='ridge')
    regr_soma_proj = grd.regress_cortex_and_retino_pos(aligned_soma, 
                                                        xvar='proj', model='ridge')
    if verbose:
        print("SOMA, MEASURED:")
        print(regr_soma_meas.to_markdown())
        print("SOMA, ALIGNED:")
        print(regr_soma_proj.to_markdown())
    if plot:
        # PLOT, soma
        fig = grd.plot_measured_and_aligned(aligned_soma, 
                            regr_soma_proj, regr_soma_meas)
        fig.text(0.01, 0.95, 'Measured vs Aligned CTX to RETINO pos (%s)' % dk)
        figname = 'measured_vs_aligned_SOMA'
        pl.savefig(os.path.join(plot_dst_dir, '%s.svg' % figname))
        pl.close()

    #### Predict CORTICAL position (from retino position)
    p_x = grd.predict_cortex_position(REGR_NP, cond='az', 
                              points=aligned_soma['x0'].values)
    p_y = grd.predict_cortex_position(REGR_NP, cond='el', 
                              points=aligned_soma['y0'].values)
    aligned_soma['predicted_ml_proj'] = p_x
    aligned_soma['predicted_ap_proj'] = p_y

    #### Predict RETINO position (from cortical position)
    r_x = grd.predict_retino_position(REGR_NP, cond='az', 
                              points=aligned_soma['ml_proj'].values)
    r_y = grd.predict_retino_position(REGR_NP, cond='el', 
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
    
    return aligned_soma.reset_index(drop=True)

def do_visualization(dk, df_, AZMAP_NP, ELMAP_NP, traceid='traces001', 
                    markersize=50, lw=0.5, alpha=1, cmap='Spectral', 
                    plot_true=True, plot_predicted=True, plot_lines=True,
                    plot_dst_dir='/tmp'):
    # # Visualization
    zimg, masks, ctrs = roiutils.get_masks_and_centroids(dk, traceid=traceid)
    pixel_size = hutils.get_pixel_size()
    zimg_r = retutils.transform_2p_fov(zimg, pixel_size)
    fig = visualize_scatter_on_fov(df_, AZMAP_NP, ELMAP_NP, zimg_r=zimg_r,
                    cmap=cmap, markersize=markersize, lw=lw, alpha=alpha,
                    plot_true=plot_true, plot_predicted=plot_predicted, 
                    plot_lines=plot_lines)
    fig.text(0.01, 0.95, 'CTX vs RETINO positions - MEASURED (%s)' % dk)
    pl.savefig(os.path.join(plot_dst_dir, 'fov_true_v_predicted_scatter.svg'))
    pl.close()
    return



def do_scatter_analysis(dk, va, do_gradients=False, do_model=False,
                        np_mag_thr=0.001, np_delay_map_thr=1.0, 
                        np_ds_factor=2., 
                        response_type='dff', do_spherical_correction=False, 
                        experiment='rfs', traceid='traces001',
                        cmap='Spectral', plot=True,
                        rootdir='/n/coxfs01/2p-data', verbose=False,
                        create_new=False):
    '''
    create_new to completely overwrite scatter analysis results
    '''
    deviations=None

    scatter_kws={'s':2}

    #### Select output dirs
    session, animalid, fovn = hutils.split_datakey_str(dk)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, \
                    'FOV%i_*' % fovn))[0]
    curr_dst_dir = os.path.join(fovdir, 'segmentation')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)

    try:
        retinorun, AZMAP_NP, ELMAP_NP, GVECTORS, REGR_NP = get_gradient_results(dk, va, 
                                do_gradients=do_gradients, do_model=do_model, 
                                np_mag_thr=np_mag_thr, 
                                np_delay_map_thr=np_delay_map_thr, 
                                np_ds_factor=np_ds_factor,
                                plot=plot, cmap=cmap, plot_dst_dir=curr_dst_dir,
                                verbose=verbose)  
        protocol = 'TILE' if 'rfs' in experiment else 'BAR'
        soma_dst_dir = os.path.join(curr_dst_dir, 'scatter_%s' % experiment)
        if not os.path.exists(soma_dst_dir):
            os.makedirs(soma_dst_dir)
        aligned_soma = get_aligned_soma(dk, retinorun, GVECTORS, REGR_NP, 
                                experiment=experiment,
                                traceid=traceid, protocol=protocol, 
                                response_type=response_type,
                                do_spherical_correction=do_spherical_correction,
                                verbose=verbose, plot=plot, plot_dst_dir=soma_dst_dir)

        if plot:
            # FOV scatter coords 
            markersize=50
            lw=0.6
            alpha=1
            plot_true=True
            plot_predicted=True
            plot_lines=True
            do_visualization(dk, aligned_soma, AZMAP_NP, ELMAP_NP, traceid=traceid, 
                            markersize=50, lw=0.5, alpha=1, cmap=cmap,
                            plot_true=True, plot_predicted=True, plot_lines=True,
                            plot_dst_dir=soma_dst_dir)


        # # Calculate scatter
        deviations = grd.get_deviations(aligned_soma)
        if plot:
            # Plot
            fig, axn = pl.subplots(1,2, figsize=(6.5, 3))
            ax=axn[0]
            sns.histplot(deviations, x='deg_scatter', hue='axis', ax=ax,
                        stat='probability', cumulative=False )
            ax.set_title('Retino scatter (deg)')
            ax=axn[1]
            sns.histplot(deviations, x='dist_scatter', hue='axis', ax=ax,
                        stat='probability', cumulative=False)
            ax.set_title('Cortical scatter (um)')
            pl.subplots_adjust(left=0.1, right=0.8, bottom=0.25, top=0.85, 
                                wspace=0.5, hspace=0.5)
            pl.savefig(os.path.join(soma_dst_dir, 'deviations.svg'))
            pl.close()

        update_results(dk, va, deviations, soma_dst_dir, create_new=create_new)

    except Exception as e:
        print("ERROR in %s, %s" % (dk, retinorun))

        traceback.print_exc()
        
    return deviations

def update_results(dk, va, soma_results, soma_dst_dir, create_new=False):
    scatter_fpath = os.path.join(soma_dst_dir, 'scatter_results.pkl')

    results={}
    if os.path.exists(scatter_fpath) and (create_new is False):
        with open(scatter_fpath, 'rb') as f:
            results = pkl.load(f)
        if not isinstance(results, dict):
            results = {}

    results.update({va: soma_results})
    with open(scatter_fpath, 'wb') as f:
        pkl.dump(results, f, protocol=2)

    return

def load_results(dk, va, rootdir='/n/coxfs01/2p-data'):
    currdf=None
    try:
        session, animalid, fovn = hutils.split_datakey_str(dk)
        fovdir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn))[0]
        curr_dst_dir = os.path.join(fovdir, 'segmentation')
        results_fpath = os.path.join(curr_dst_dir, 'scatter_results.pkl')
        with open(results_fpath, 'rb') as f:
            results = pkl.load(f)
        assert va in results.keys(), 'Visual area not found in scatter analysis'
        currdf = results[va].copy()
    except Exception as e:
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
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='rfs', 
                      help="experiment to calculate scatter (e.g,. rfs, rfs10, retino)") 
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

    parser.add_option('--plot',  action='store_true', dest='plot', default=False,
                      help="plot and save figures")
    parser.add_option('--cmap',  action='store', dest='cmap', default='Spectral',
                      help="Colormap to use for background img (default: Spectral)")
    parser.add_option('-v', '--verbose',  action='store_true', dest='verbose', default=False,
                      help="verbose")


    parser.add_option('--all',  action='store_true', dest='cycle_all', default=False,
                      help="Set flag to cycle thru ALL dsets")

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
    do_gradients = optsE.do_gradients
    do_model = optsE.do_model
    plot = optsE.plot
    if do_gradients:
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

    if cycle_all:
        sdata = aggr.get_aggregate_info(visual_areas=['V1', 'Lm', 'Li'], 
                                        return_cells=False)
        meta = sdata[(sdata.experiment==experiment)]
        for (va, dk, experiment), g in meta.groupby(['visual_area', 'datakey', 'experiment']):
            print("Area: %s, all <%s> datasets" % (va, experiment))
            deviants = do_scatter_analysis(dk, va, experiment=experiment, 
                            do_gradients=do_gradients, do_model=do_model,
                            np_mag_thr=np_mag_thr, np_delay_map_thr=np_delay_map_thr, 
                            np_ds_factor=np_ds_factor,
                            response_type=response_type, 
                            do_spherical_correction=do_spherical_correction,
                            traceid=traceid,
                            cmap=cmap, plot=plot, verbose=verbose)
    else:
        assert dk is not None, "Must specify datakey" 
        if (va is None):
            sdata = aggr.get_aggregate_info(visual_areas=['V1', 'Lm', 'Li'], 
                                        return_cells=False)
            meta = sdata[(sdata.datakey==dk) & (sdata.experiment==experiment)]
            found_areas = meta['visual_area'].unique()
        else:
            found_areas = [va]
        for va in found_areas:
            print("Processing: %s, %s" % (va, dk))
            deviants = do_scatter_analysis(dk, va, 
                            do_gradients=do_gradients, do_model=do_model,
                            np_mag_thr=np_mag_thr, np_delay_map_thr=np_delay_map_thr, 
                            np_ds_factor=np_ds_factor,
                            response_type=response_type, 
                            do_spherical_correction=do_spherical_correction,
                            experiment=experiment, traceid=traceid,
                            cmap=cmap, plot=plot, verbose=verbose)


    print("Done.")

if __name__ == '__main__':

    main(sys.argv[1:])


