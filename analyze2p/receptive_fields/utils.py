import os
import itertools
import glob
import traceback
import json
import cv2
import math
import time
import shutil
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

import seaborn as sns
import dill as pkl
import scipy.stats as spstats
import numpy as np
import pandas as pd
import scipy.optimize as opt

import analyze2p.aggregate_datasets as aggr 
import analyze2p.plotting as pplot
import analyze2p.utils as hutils
import analyze2p.extraction.rois as roiutils

from matplotlib.patches import Ellipse, Rectangle, Polygon
from shapely.geometry.point import Point
from shapely.geometry import box
from shapely import affinity

from scipy import interpolate

from mpl_toolkits.axes_grid1 import make_axes_locatable

import multiprocessing as mp
import copy

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
        ax = plt.gca()
    # Calculate the residual from a linear regression
    #_, yhat, _ = plotter.fit_regression(grid=plotter.x)
    #plotter.y = plotter.y - yhat
    print(len(plotter.x))

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
                      scatter_kws=None, line_kws=None, deviant_color='magenta',
                      lw=0.25, marker='o', fontsize=6):
    rois_=rfs_['cell'].unique()
    x_bins = sorted(rfs_[xname].values)
    sort_rois_by_x = np.array(rfs_[xname].argsort().values)
    # Get mean and upper/lower CI bounds of bootstrapped distn for each cell
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
    print(eq_str)
    #Refit line to get correct xbins
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
        pass_boot = (bootL <= float(g[yname]) <= bootU)
        pass_regr = ( (regrL > bootU) or (regrU < bootL) )
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
    ax.errorbar(rfs_[xname][deviants], boot_medians_df.loc[deviants].values, yerr=yerrs,
                    fmt='none', color=deviant_color, alpha=1, lw=lw)
    #ax.set_title(eq_str, loc='left', fontsize=fontsize)
    # # Old way (unprojected)
    # sns.regplot(x='ml_pos', y=yname, data=rfs_.reset_index(drop=True), ax=ax,
    #             scatter=False, color='c')
    # ax.scatter(x='ml_pos', y=yname, data=rfs_.reset_index(drop=True), s=2, color='c')
    
    return ax




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

# plotting
def anisotropy_polarplot(rdf, metric='anisotropy', cmap='spring_r', alpha=0.5, 
                            marker='o', markersize=30, ax=None, 
                            hue_param='aniso_index', cbar_bbox=[0.4, 0.15, 0.2, 0.03]):

    vmin=0; vmax=1;
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    iso_cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    if ax is None:
        fig, ax = pl.subplots(1, subplot_kw=dict(projection='polar'), figsize=(4,3))

    thetas = rdf['theta_Mm_c'].values #% np.pi # all thetas should point the same way
    ratios = rdf[metric].values
    ax.scatter(thetas, ratios, s=markersize, c=ratios, cmap=cmap, alpha=alpha)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticklabels(['0$^\circ$', '', '90$^\circ$', '', '', '', '-90$^\circ$', ''])
    ax.set_rlabel_position(135) #315)
    ax.set_xlabel('')
    ax.set_yticklabels(['', 0.4, '', 0.8])
    ax.set_ylabel(metric, fontsize=12)

    # Grid lines and such
    ax.spines['polar'].set_visible(False)
    pl.subplots_adjust(left=0.1, right=0.9, wspace=0.2, bottom=0.3, top=0.8, hspace=0.5)

    # Colorbar
    iso_cmap._A = []
    cbar_ax = ax.figure.add_axes(cbar_bbox)
    cbar = ax.figure.colorbar(iso_cmap, cax=cbar_ax, orientation='horizontal', ticks=[0, 1])
    if metric == 'anisotropy':
        xlabel_min = 'Iso\n(%.1f)' % (vmin) 
        xlabel_max= 'Aniso\n(%.1f)' % (vmax) 
    else:             
        xlabel_min = 'H\n(%.1f)' % (vmin) if hue_param in ['angle', 'aniso_index'] else '%.2f' % vmin
        xlabel_max= 'V\n(%.1f)' % (vmax) if hue_param in ['angle', 'aniso_index'] else '%.2f' % vmax
    cbar.ax.set_xticklabels([xlabel_min, xlabel_max])  # horizontal colorbar
    cbar.ax.tick_params(which='both', size=0)

    return ax



# Ellipse fitting and formatting
def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)

    return ellr

def rfs_to_polys(rffits, sigma_scale=2.35):
    '''
    rffits (pd dataframe)
        index : roi indices (same as gdf.rois)
        columns : r2, sigma_x, sigma_y, theta, x0, y0 (already converted) 
    
    returns list of polygons to do calculations with
    '''
    sz_param_names = [f for f in rffits.columns if '_' in f]
    sz_metrics = np.unique([f.split('_')[0] for f in sz_param_names])
    sz_metric = sz_metrics[0]
    assert sz_metric in ['fwhm', 'std'], "Unknown size metric: %s" % str(sz_metrics)

    sigma_scale = 1.0 if sz_metric=='fwhm' else sigma_scale
    roi_param = 'cell' if 'cell' in rffits.columns else 'rid'

    rf_columns=[roi_param, '%s_x' % sz_metric, '%s_y' % sz_metric, 'theta', 'x0', 'y0']
    rffits = rffits[rf_columns]
    rf_polys=dict((rid, 
        create_ellipse((x0, y0), (abs(sx)*sigma_scale, abs(sy)*sigma_scale), np.rad2deg(th))) \
        for rid, sx, sy, th, x0, y0 in rffits.values)

    return rf_polys

def stimsize_poly(sz, xpos=0, ypos=0):
    from shapely.geometry import box
 
    ry_min = ypos - sz/2.
    rx_min = xpos - sz/2.
    ry_max = ypos + sz/2.
    rx_max = xpos + sz/2.
    s_blobs = box(rx_min, ry_min, rx_max, ry_max)
    
    return s_blobs

def calculate_overlap(poly1, poly2, r1='poly1', r2='poly2'):
    #r1, poly1 = poly_tuple1
    #r2, poly2 = poly_tuple2

    #area_of_smaller = min([poly1.area, poly2.area])
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    area_overlap = intersection_area/union_area 

    area_of_smaller = min([poly1.area, poly2.area])
    overlap_area = poly1.intersection(poly2).area
    perc_overlap = overlap_area/area_of_smaller


    odf = pd.DataFrame({'poly1':r1,
                        'poly2': r2,
                        'area_overlap': area_overlap, #overlap_area,
                        'perc_overlap': perc_overlap}, index=[0])
    
    return odf


def get_proportion_overlap(poly_tuple1, poly_tuple2):
    r1, poly1 = poly_tuple1
    r2, poly2 = poly_tuple2

    area_of_smaller = min([poly1.area, poly2.area])
    overlap_area = poly1.intersection(poly2).area
    perc_overlap = overlap_area/area_of_smaller

    odf = pd.DataFrame({'row':r1,
                        'col': r2,
                        'area_overlap': overlap_area,
                        'perc_overlap': perc_overlap}, index=[0])
    
    return odf


def get_rf_overlaps(rf_polys):
    '''
    tuning_ (pd.DataFrame): nconds x nrois.
    Each entry is the mean response (across trials) for a given stim condition.
    '''
    # Calculate signal corrs
    o_=[]
    rois_ = sorted(rf_polys.keys())
    # Get unique pairs, then iterate thru and calculate pearson's CC
    for col_a, col_b in itertools.combinations(rois_, 2):
        df_ = calculate_overlap(rf_polys[col_a], rf_polys[col_b], \
                                  r1=col_a, r2=col_b)
        o_.append(df_)
    overlapdf = pd.concat(o_)
                   
    return overlapdf


# Data processing
def rfits_to_df(fitr,  fit_params={}, roi_list=None,
                scale_sigma=True, sigma_scale=2.35, convert_coords=True):
    '''
    Takes each roi's RF fit results, converts to screen units, and return as dataframe.
    Scale to make size FWFM if scale_sigma is True.
    '''
    if roi_list is None:
        roi_list = sorted(fitr.keys())
       
    sigma_scale = sigma_scale if scale_sigma else 1.0

    fitdf = pd.DataFrame({'x0': [fitr[r]['x0'] for r in roi_list],
                          'y0': [fitr[r]['y0'] for r in roi_list],
                          'sigma_x': [fitr[r]['sigma_x'] for r in roi_list],
                          'sigma_y': [fitr[r]['sigma_y'] for r in roi_list],
                          'theta': [fitr[r]['theta'] % (2*np.pi) for r in roi_list],
                          'offset': [fitr[r]['offset'] for r in roi_list],
                          'amplitude': [fitr[r]['amplitude'] for r in roi_list],

                          'r2': [fitr[r]['r2'] for r in roi_list]},
                              index=roi_list)

    if convert_coords:
        fitdf = convert_fit_to_coords(fitdf, fit_params, scale_sigma=False)
        fitdf['sigma_x'] = fitdf['sigma_x']*sigma_scale
        fitdf['sigma_y'] = fitdf['sigma_y']*sigma_scale

#            x0, y0, sigma_x, sigma_y = convert_fit_to_coords(fitdf, row_vals, col_vals)
#            fitdf['x0'] = x0
#            fitdf['y0'] = y0
#            fitdf['sigma_x'] = sigma_x * sigma_scale
#            fitdf['sigma_y'] = sigma_y * sigma_scale

    return fitdf

def apply_scaling_to_df(row, grid_points=None, new_values=None):
    #r2 = row['r2']
    #theta = row['theta']
    #offset = row['offset']
    x0, y0, sx, sy = get_scaled_sigmas(grid_points, new_values,
                                        row['x0'], row['y0'], 
                                        row['sigma_x'], row['sigma_y'], row['theta'],
                                        convert=True)
    return x0, y0, sx, sy #sx, sy, x0, y0


def get_spherical_coords(cart_pointsX=None, cart_pointsY=None, cm_to_degrees=True,
                    resolution=(1080, 1920),
                   xlim_degrees=(-59.7, 59.7), ylim_degrees=(-33.6, 33.6)):

    # Monitor size and position variables
    width_cm = 103; #%56.69;  % 103 width of screen, in cm
    height_cm = 58; #%34.29;  % 58 height of screen, in cm
    pxXmax = resolution[1] #1920; #%200; % number of pixels in an image that fills the whole screen, x
    pxYmax = resolution[0] #1080; #%150; % number of pixels in an image that fills the whole screen, y

    # Eye info
    cx = width_cm/2. # % eye x location, in cm
    cy = height_cm/2. # %11.42; % eye y location, in cm
    eye_dist = 30.; #% in cm

    # Distance to bottom of screen, along the horizontal eye line
    zdistBottom = np.sqrt((cy**2) + (eye_dist**2)) #; %24.49;     % in cm
    zdistTop    = np.sqrt((cy**2) + (eye_dist**2)) #; %14.18;     % in cm

    # Internal conversions
    top = height_cm-cy;
    bottom = -cy;
    right = cx;
    left = cx - width_cm;

    if cart_pointsX is None or cart_pointsY is None:
        [xi, yi] = np.meshgrid(np.arange(0, pxXmax), np.arange(0, pxYmax))
        print(xi.shape, yi.shape)

        cart_pointsX = left + (float(width_cm)/pxXmax)*xi;
        cart_pointsY = top - (float(height_cm)/pxYmax)*yi;
        cart_pointsZ = zdistTop + ((zdistBottom-zdistTop)/float(pxYmax))*yi
    else:
        cart_pointsZ = zdistTop + ((zdistBottom-zdistTop)/float(pxYmax))*cart_pointsY

    if cm_to_degrees:
        xmin_cm=cart_pointsX.min(); xmax_cm=cart_pointsX.max();
        ymin_cm=cart_pointsY.min(); ymax_cm=cart_pointsY.max();
        xmin_deg, xmax_deg = xlim_degrees
        ymin_deg, ymax_deg = ylim_degrees
        cart_pointsX = hutils.convert_range(cart_pointsX, oldmin=xmin_cm, oldmax=xmax_cm, 
                                       newmin=xmin_deg, newmax=xmax_deg)
        cart_pointsY = hutils.convert_range(cart_pointsY, oldmin=ymin_cm, oldmax=ymax_cm, 
                                       newmin=ymin_deg, newmax=ymax_deg)
        cart_pointsZ = hutils.convert_range(cart_pointsZ, oldmin=ymin_cm, oldmax=ymax_cm, 
                                       newmin=ymin_deg, newmax=ymax_deg)

    sphr_pointsTh, sphr_pointsPh, sphr_pointsR = cart2sph(cart_pointsZ, cart_pointsX, cart_pointsY)

    return cart_pointsX, cart_pointsY, sphr_pointsTh, sphr_pointsPh


def warp_spherical(image_values, cart_pointsX, cart_pointsY, sphr_pointsTh, sphr_pointsPh, 
                    normalize_range=True, in_radians=True, method='linear'):
    '''
    Take an image, warp from cartesian to spherical ("perceived" image)
    '''
    from scipy.interpolate import griddata

    xmaxRad = sphr_pointsTh.max()
    ymaxRad = sphr_pointsPh.max()

    # normalize max of Cartesian to max of Spherical
    fx = xmaxRad/cart_pointsX.max() if normalize_range else 1.
    fy = ymaxRad/cart_pointsY.max() if normalize_range else 1.
    x0 = cart_pointsX.copy()*fx
    y0 = cart_pointsY.copy()*fy
   
    if in_radians and not normalize_range:
        points = np.array( (np.rad2deg(sphr_pointsTh).flatten(), np.rad2deg(sphr_pointsPh).flatten()) ).T
    else:
        points = np.array( (sphr_pointsTh.flatten(), sphr_pointsPh.flatten()) ).T

    values_ = image_values.flatten()
    #values_y = cart_pointsY.flatten()

    warped_values = griddata( points, values_, (x0,y0) , method=method)
    
    return warped_values



def get_screen_lim_pixels(lin_coord_x, lin_coord_y, row_vals=None, col_vals=None):
    '''
    Get Bbox of stimulated pixels on screen for RF maps
    ''' 
    #pix_per_deg=16.050716 pix_per_deg = screeninfo['pix_per_deg']
    stim_size = float(np.unique(np.diff(row_vals)))

    right_lim = max(col_vals) + (stim_size/2.)
    left_lim = min(col_vals) - (stim_size/2.)
    top_lim = max(row_vals) + (stim_size/2.)
    bottom_lim = min(row_vals) - (stim_size/2.)

    # Get actual stimulated locs in pixels
    i_x, i_y = np.where( np.abs(lin_coord_x-right_lim) == np.abs(lin_coord_x-right_lim).min() )
    pix_right_edge = int(np.unique(i_y))

    i_x, i_y = np.where( np.abs(lin_coord_x-left_lim) == np.abs(lin_coord_x-left_lim).min() )
    pix_left_edge = int(np.unique(i_y))
    #print("AZ bounds (pixels): ", pix_right_edge, pix_left_edge)

    i_x, i_y = np.where( np.abs(lin_coord_y-top_lim) == np.abs(lin_coord_y-top_lim).min() )
    pix_top_edge = int(np.unique(i_x))

    i_x, i_y = np.where( np.abs(lin_coord_y-bottom_lim) == np.abs(lin_coord_y-bottom_lim).min() )
    pix_bottom_edge = int(np.unique(i_x))
    #print("EL bounds (pixels): ", pix_top_edge, pix_bottom_edge)

    # Check expected tile size
    #ncols = len(col_vals); nrows = len(row_vals);
    #expected_sz_x = (pix_right_edge-pix_left_edge+1) * (1./pix_per_deg) / ncols
    #expected_sz_y = (pix_bottom_edge-pix_top_edge+1) * (1./pix_per_deg) / nrows
    #print("tile sz-x, -y should be ~(%.2f, %.2f) deg" % (expected_sz_x, expected_sz_y))
    
    return (pix_bottom_edge, pix_left_edge, pix_top_edge,  pix_right_edge)


def coordinates_for_transformation(fit_params):
    ds_factor = fit_params['downsample_factor']
    col_vals = fit_params['col_vals']
    row_vals = fit_params['row_vals']
    nx = len(col_vals)
    ny = len(row_vals)

    # Downsample screen resolution
    resolution_ds = [int(i/ds_factor) for i in fit_params['screen']['resolution'][::-1]]
    print("Screen res (ds=%ix): [%i, %i]" % (ds_factor, resolution_ds[0], resolution_ds[1]))

    if fit_params['use_linear']:
        # Get linear coordinates in degrees (downsampled)
        lin_x, lin_y = get_lin_coords(resolution=resolution_ds, cm_to_deg=True) 
        # Get Spherical coordinate mapping
        cart_x, cart_y, sphr_x, sphr_y = get_spherical_coords(cart_pointsX=lin_x, 
                                                          cart_pointsY=lin_y,
                                                          cm_to_degrees=False) # in deg

    else:
        cart_x, cart_y, sphr_x, sphr_y = get_spherical_coords(cart_pointsX=None, 
                                                          cart_pointsY=None,
                                                          cm_to_degrees=True,
                                                          resolution=resolution_ds)

    screen_bounds_pix = get_screen_lim_pixels(lin_x, lin_y, 
                                            row_vals=row_vals, col_vals=col_vals)
    (pix_bottom_edge, pix_left_edge, pix_top_edge, pix_right_edge) = screen_bounds_pix
 
    # Trim and downsample coordinate space to match corrected map
    cart_x_ds  = cv2.resize(cart_x[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))
    cart_y_ds  = cv2.resize(cart_y[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))

    sphr_x_ds  = cv2.resize(sphr_x[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx,ny))
    sphr_y_ds  = cv2.resize(sphr_y[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))

    grid_x, grid_y = np.meshgrid(range(nx),range(ny)[::-1])
    grid_points = np.array( (grid_x.flatten(), grid_y.flatten()) ).T
    cart_values = np.array( (cart_x_ds.flatten(), cart_y_ds.flatten()) ).T
    sphr_values = np.array( (np.rad2deg(sphr_x_ds).flatten(), np.rad2deg(sphr_y_ds).flatten()) ).T

    return grid_points, cart_values, sphr_values


def convert_fit_to_coords(fitdf0, fit_params, scale_sigma=True, 
                        sigma_scale=2.35):
    '''
    RF map arrays to be converted to real-world fit params
    '''
    fitdf = fitdf0.copy()
    sigma_scale = sigma_scale if scale_sigma else 1.0
    #grid_points, cart_values, sphr_values = coordinates_for_transformation(fit_params)
    #print(fit_params.keys())
    newdf = convert_fit_to_coords_og(fitdf, fit_params['row_vals'], 
                                                fit_params['col_vals'])

#    if 'do_spherical_correction' not in fit_params.keys():
#        # TMP until can run all of fit_rf again
#        newdf = convert_fit_to_coords_og(fitdf, fit_params['row_vals'], 
#                                                fit_params['col_vals'])
#
#    elif fit_params['do_spherical_correction']:
#        print("-- converting sphr")
#        grid_points, cart_values, sphr_values = coordinates_for_transformation(fit_params)
#
#        # Upsampled, with spherical correction
#        converted = fitdf.apply(apply_scaling_to_df, 
#                                args=(grid_points, sphr_values), axis=1)
#        newdf = pd.DataFrame([[x0, y0, sx*sigma_scale, sy*sigma_scale] \
#                        for x0, y0, sx, sy in converted.values], index=converted.index, 
#                        columns=['x0', 'y0', 'sigma_x', 'sigma_y'])
#    else:
#        print("reg")
#        # This is just upsampling
#        #converted = fitdf.apply(apply_scaling_to_df, 
#        #                        args=(grid_points, cart_values), axis=1)
#        newdf = convert_fit_to_coords_og(fitdf, fit_params['row_vals'], 
#                                                fit_params['col_vals'])
#
    fitdf[['sigma_x', 'sigma_y', 'x0', 'y0']] = newdf[['sigma_x', 'sigma_y', 'x0', 'y0']]

    return fitdf


def convert_fit_to_coords_og(fitdf, row_vals, col_vals, rid=None):
    df_=None 
    if rid is not None:
        xx = hutils.convert_range(fitdf['x0'][rid], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0) 
        sigma_x = hutils.convert_range(abs(fitdf['sigma_x'][rid]), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0) 
        yy = hutils.convert_range(fitdf['y0'][rid], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0) 
        sigma_y = hutils.convert_range(abs(fitdf['sigma_y'][rid]), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    else:
        xx = hutils.convert_range(fitdf['x0'], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0) 
        sigma_x = hutils.convert_range(abs(fitdf['sigma_x']), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0) 
        yy = hutils.convert_range(fitdf['y0'], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0) 
        sigma_y = hutils.convert_range(abs(fitdf['sigma_y']), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
   
    df_ = pd.DataFrame({'x0': xx, 'y0': yy, 'sigma_x': sigma_x, 'sigma_y': sigma_y}) 
    return df_ #xx, yy, sigma_x, sigma_y



# ###############################################3
# Data loading
# ###############################################3
def get_fit_desc(response_type='dff', do_spherical_correction=False):
    if do_spherical_correction:
        fit_desc = 'fit-2dgaus_%s_sphr' % response_type
    else:
        fit_desc = 'fit-2dgaus_%s-no-cutoff' % response_type        

    return fit_desc

def create_rf_dir(datakey, run_name, traceid='traces001', response_type='dff', 
                do_spherical_correction=False, fit_thr=0.5, rootdir='/n/coxfs01/2p-data'):

    session, animalid, fovnum = hutils.split_datakey_str(datakey)

    # Get RF dir for current fit type
    fit_desc = get_fit_desc(response_type=response_type, 
                            do_spherical_correction=do_spherical_correction)
    fov_dir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovnum))[0]

    if 'combined' in run_name:
        traceid_dirs = [t for t in glob.glob(os.path.join(fov_dir, run_name, \
                                                'traces/%s*' % traceid))]
    else: 
        traceid_dirs = [t for t in glob.glob(os.path.join(fov_dir, \
                        'combined_%s_*' % run_name, 'traces/%s*' % traceid))]
    if len(traceid_dirs) > 1:
        print("[creating RF dir, %s] More than 1 trace ID found:" % run_name)
        for ti, traceid_dir in enumerate(traceid_dirs):
            print(ti, traceid_dir)
        sel = input("Select IDX of traceid to use: ")
        traceid_dir = traceid_dirs[int(sel)]
    else:
        traceid_dir = traceid_dirs[0]
    #traceid = os.path.split(traceid_dir)[-1]
         
    rfdir = os.path.join(traceid_dir, 'receptive_fields', fit_desc)
    return rfdir, fit_desc

def load_fit_results(datakey, experiment='rfs', 
                response_type='dff', do_spherical_correction=False,
                traceid='traces001', fit_desc=None,  
                rootdir='/n/coxfs01/2p-data'): 
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    fit_results = None
    fit_params = None
    try: 
        if fit_desc is None:
            assert response_type is not None, "No response_type or fit_desc provided"
            fit_desc = get_fit_desc(response_type=response_type, 
                            do_spherical_correction=do_spherical_correction)  
        exp_name = 'gratings' if int(session) < 20190511 else experiment
        run_name = exp_name.split('_')[1] if 'combined' in exp_name else exp_name
        rfdir = glob.glob(os.path.join(rootdir, animalid, session,
                        'FOV%i_*' % fovn, 'combined_%s_*' % run_name,
                        'traces/%s*' % traceid, 'receptive_fields', 
                        '%s' % fit_desc))[0]
    except AssertionError as e:
        traceback.print_exc()
       
    # Load results
    rf_results_fpath = os.path.join(rfdir, 'fit_results.pkl')
    with open(rf_results_fpath, 'rb') as f:
        fit_results = pkl.load(f, encoding='latin1')
   
    # Load params 
    rf_params_fpath = os.path.join(rfdir, 'fit_params.json')
    with open(rf_params_fpath, 'r') as f:
        fit_params = json.load(f)
        
    return fit_results, fit_params
 

def load_eval_results(datakey, experiment='rfs', rfdir=None,
                    traceid='traces001', fit_desc=None,
                    response_type='dff', do_spherical_correction=False,
                    rootdir='/n/coxfs01/2p-data'):

    '''
    Load EVALUTION for a given fit. Provide RFDIR to load
    eval_results from same dir. Either provide:
    - fit_desc OR response_type, do_spherical_evaluation
    - rfdir OR traceid, experiment (rfs, rfs10)
    '''
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    eval_results=None; eval_params=None;            
    try: 
        if rfdir is None:
            run_name = experiment.split('_')[1] if 'combined' in experiment else experiment
            if fit_desc is None:
                fit_desc = get_fit_desc(response_type=response_type,
                                        do_spherical_correction=do_spherical_correction)
                rfdir = glob.glob(os.path.join(rootdir, animalid, session, 
                        'FOV%i_*' % fovn, '*%s_*' % run_name, 'traces/%s*' % traceid, 
                        'receptive_fields', '%s*' % fit_desc))[0]
        evaldir = os.path.join(rfdir, 'evaluation')
        assert os.path.exists(evaldir), \
                        "No evaluation exists\n(%s)\n. Aborting" % evaldir
    except IndexError as e:
        traceback.print_exc()
        return None, None
    except AssertionError as e:
        traceback.print_exc()
        return None, None

    # Load results
    rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    assert os.path.exists(rf_eval_fpath), "No eval result: %s" % datakey 
    with open(rf_eval_fpath, 'rb') as f:
        eval_results = pkl.load(f, encoding='latin1')
   
    #  Load params 
    eval_params_fpath = os.path.join(rfdir, 'fit_params.json')
    with open(eval_params_fpath, 'r') as f:
        eval_params = json.load(f)
        
    return eval_results, eval_params


def load_matching_fit_results(animalid, session, fov, traceid='traces001',
                              experiment='rfs', response_type='dff',
                              nframes_post=0, do_spherical_correction=False,
                              sigma_scale=2.35, scale_sigma=True):
    fit_results=None
    fit_params=None
    try:
        fit_results, fit_params = fitrf.load_fit_results(animalid, session,
                                        fov, traceid=traceid,
                                        experiment=experiment,
                                        response_type=response_type,
                                        do_spherical_correction=do_spherical_correction)
        assert fit_params['nframes_post_onset'] == nframes_post, \
            "Incorrect nframes_post (found %i, requested %i)" \
            % (fit_params['nframes_post_onset'], nframes_post)
        assert fit_params['response_type'] == response_type, \
            "Incorrect response type (found %i, requested %i)" \
            %(fit_params['repsonse_type'], response_type)
        if sigma_scale!=fit_params['sigma_scale'] or scale_sigma!=fit_params['scale_sigma']:
            print("... updating scale_sigma: %s" % str(fit_params['sigma_scale']))
            scale_sigma=fit_params['scale_sigma']               
            print("... updating sigma_scale to %.2f (from %.2f)" \
                            % (fit_params['sigma_scale'], sigma_scale))
            sigma_scale=fit_params['sigma_scale']
            do_fits=True
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("[err]: unable to load fit results, re-fitting.")
    
    return fit_results, fit_params



def get_reliable_fits(pass_cis, pass_criterion='all', single=False):
    '''
    Only return cells with measured PARAM within 95% CI (based on bootstrap fits)
    pass_cis:  gives the CI range
    '''
    if single is True:
        keep_rids = [i for i in pass_cis.index.tolist() \
                        if pass_cis[pass_criterion][i]==True]
    else:       
        param_cols = [p for p in pass_cis.columns if p!='cell']
        if pass_criterion=='all':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() if all(tmp_ci.loc[i])]
        elif pass_criterion=='any':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() if any(tmp_ci.loc[i])]
        elif pass_criterion=='size':
            keep_rids = [i for i in pass_cis.index.tolist() 
                            if (pass_cis['sigma_x'][i]==True and pass_cis['sigma_y'][i]==True)]
        elif pass_criterion=='position':
            keep_rids = [i for i in pass_cis.index.tolist() 
                            if (pass_cis['x0'][i]==True and pass_cis['y0'][i]==True)]
        elif pass_criterion=='most':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() 
                            if sum([pv==True 
                            for pv in tmp_ci.loc[rid]])/float(len(param_cols))>0.5]
        else:   
            keep_rids = [i for i in pass_cis.index.tolist() if any(pass_cis.loc[i])]
       
    pass_df = pass_cis.loc[keep_rids]
 
    reliable_rois = sorted(pass_df.index.tolist())

    return reliable_rois

 
def cycle_and_load(rfmeta, assigned_cells, fit_desc=None, traceid='traces001', fit_thr=0.5, 
                      scale_sigma=True, sigma_scale=2.35, verbose=False, 
                      response_type='None', reliable_only=True,
                      rootdir='/n/coxfs01/2p-data'):
    '''
    Combines fit_results.pkl(fit from data) and evaluation_results.pkl (evaluated fits via bootstrap)
    and gets fit results only for those cells that are good/robust fits based on bootstrap analysis.
    '''
    rfdf = None
    df_list = []
    no_fits=[]
    no_eval=[]
    for (visual_area, datakey, experiment), g in \
                            rfmeta.groupby(['visual_area', 'datakey', 'experiment']):
        if experiment not in ['rfs', 'rfs10']:
            continue


        session, animalid, fovnum = hutils.split_datakey_str(datakey)
        fov = 'FOV%i_zoom2p0x' % fovnum
        curr_cells = assigned_cells[(assigned_cells.visual_area==visual_area)
                                  & (assigned_cells.datakey==datakey)]['cell'].unique() 
        curr_rfname = experiment if int(session)>=20190511 else 'gratings'
        #### Load fit results from measured
        try:
            fit_results, fit_params = load_fit_results(datakey,
                                            experiment=curr_rfname,
                                            traceid=traceid, 
                                             fit_desc=fit_desc)
            assert fit_results is not None 
        except Exception as e:
            no_fits.append((visual_area, datakey))
            continue

        if reliable_only:
            try:
                #### Load eval results 
                eval_results, eval_params = load_eval_results(datakey,
                                                experiment=curr_rfname, 
                                                traceid=traceid, 
                                                fit_desc=fit_desc)   
                fit_rois = sorted(eval_results['bootdf']['cell'].unique())
                assert eval_results is not None, \
                    '-- no good (%s, %s)), skipping' % (datakey, experiment)
            except Exception as e: 
                no_eval.append((visual_area, datakey))
                continue
        else:
            fit_rois = sorted(list(fit_results.keys()))

        if len(fit_rois)==0:
            continue

        scale_sigma = fit_params['scale_sigma']
        sigma_scale = fit_params['sigma_scale']
        rfit_df = rfits_to_df(fit_results, fit_params=fit_params,
                        scale_sigma=scale_sigma, 
                        sigma_scale=sigma_scale,
                        roi_list=fit_rois)

        #### Identify cells with measured params within 95% CI of bootstrap distN
        pass_rois = rfit_df[rfit_df['r2']>fit_thr].index.tolist()
        param_list = [param for param in rfit_df.columns if param != 'r2']
        reliable_rois=[]
        if reliable_only:
            # check if all params within 95% CI
            reliable_rois = get_reliable_fits(eval_results['pass_cis'],
                                                     pass_criterion='all')
        if verbose:
            print("[%s] %s: %i of %i fit rois pass for all params" \
                        % (visual_area, datakey, len(pass_rois), len(fit_rois)))
            print("...... : %i of %i fit rois passed as reliiable" \
                        % (len(reliable_rois), len(pass_rois)))

        #### Create dataframe with params only for good fit cells
        keep_rois = reliable_rois if reliable_only else pass_rois
        if curr_cells is not None:
            keep_rois = [r for r in keep_rois if r in curr_cells]

        passdf = rfit_df.loc[keep_rois].copy()
        # "un-scale" size, if flagged
        if not scale_sigma:
            sigma_x = passdf['sigma_x']/sigma_scale
            sigma_y = passdf['sigma_y'] / sigma_scale
            passdf['sigma_x'] = sigma_x
            passdf['sigma_y'] = sigma_y

        passdf['cell'] = keep_rois
        passdf['datakey'] = datakey
        passdf['animalid'] = animalid
        passdf['session'] = session
        passdf['fovnum'] = fovnum
        passdf['visual_area'] = visual_area
        passdf['experiment'] = experiment
        df_list.append(passdf)


    if len(df_list)>0:
        rfdf = pd.concat(df_list, axis=0).reset_index(drop=True)    
        rfdf = update_rf_metrics(rfdf, scale_sigma=scale_sigma)

    return rfdf


def update_rf_metrics(rfdf, scale_sigma=True):
    # Include average RF size (average of minor/major axes of fit ellipse)
    if scale_sigma:
        rfdf = rfdf.rename(columns={'sigma_x': 'fwhm_x', 'sigma_y': 'fwhm_y'})
        rfdf['std_x'] = rfdf['fwhm_x']/2.35
        rfdf['std_y'] = rfdf['fwhm_y']/2.35
    else:
        rfdf = rfdf.rename(columns={'sigma_x': 'std_x', 'sigma_y': 'std_y'})
        rfdf['fwhm_x'] = rfdf['std_x']*2.35
        rfdf['fwhm_y'] = rfdf['std_y']*2.35

    rfdf['fwhm_avg'] = rfdf[['fwhm_x', 'fwhm_y']].mean(axis=1)
    rfdf['std_avg'] = rfdf[['std_x', 'std_y']].mean(axis=1)
    rfdf['area'] = np.pi * (rfdf['std_x'] * rfdf['std_y'])

    # Add some additional common info
    #### Split fx, fy for theta comp
    rfdf['fx'] = abs(rfdf[['std_x', 'std_y']].max(axis=1) * np.cos(rfdf['theta']))
    rfdf['fy'] = abs(rfdf[['std_x', 'std_y']].max(axis=1) * np.sin(rfdf['theta']))
    rfdf['ratio_xy'] = rfdf['std_x']/rfdf['std_y']

    # Convert thetas to [-90, 90]
    thetas = [(t % np.pi) - np.pi if 
                    ((np.pi/2.)<t<(np.pi) or (((3./2)*np.pi)<t<2*np.pi)) \
                    else (t % np.pi) for t in rfdf['theta'].values]
    rfdf['theta_c'] = thetas

    # Anisotropy metrics
    #rfdf['anisotropy'] = [(sx-sy)/(sx+sy) for (sx, sy) in rfdf[['std_x', 'std_y']].values]
    # Find indices where std_x < std_y
    swap_ixs = rfdf[rfdf['std_x'] < rfdf['std_y']].index.tolist()

    # Get thetas in deg for plotting (using Ellipse() patch function)
    # Note: regardless of whether std_x or _y bigger, when plotting w/ width=Major, height=minor
    #       or width=std_x, height=std_y, should have correct theta orientation 
    # theta_Mm_deg = Major, minor as width/height, corresponding theta for Ellipse(), in deg.
    rfdf['theta_Mm_deg'] = np.rad2deg(rfdf['theta'].copy())
    rfdf.loc[swap_ixs, 'theta_Mm_deg'] = [ (theta + 90) % 360 if (90 <= theta < 360) \
                                          else (((theta) % 90) + 90) % 360
                                    for theta in np.rad2deg(rfdf['theta'][swap_ixs].values) ]        

    # Get true major and minor axes 
    rfdf['major_axis'] = [max([sx, sy]) for sx, sy in rfdf[['std_x', 'std_y']].values]
    rfdf['minor_axis'] = [min([sx, sy]) for sx, sy in rfdf[['std_x', 'std_y']].values]

    # Get anisotropy index from these (0=isotropic, >0=anisotropic)
    rfdf['anisotropy'] = [(sx-sy)/(sx+sy) for (sx, sy) in rfdf[['major_axis', 'minor_axis']].values]

    # Calculate true theta that shows orientation of RF relative to major/minor axes
    nu_thetas = [(t % np.pi) - np.pi if ((np.pi/2.)<t<(np.pi) or (((3./2)*np.pi)<t<2*np.pi)) \
                 else (t % np.pi) for t in np.deg2rad(rfdf['theta_Mm_deg'].values) ]
    rfdf['theta_Mm_c'] = nu_thetas


    # Get anisotropy index
    sins = abs(np.sin(rfdf['theta_Mm_c']))
    sins_c = hutils.convert_range(sins, oldmin=0, oldmax=1, newmin=-1, newmax=1)
    rfdf['aniso_index'] = sins_c * rfdf['anisotropy']
 
    return rfdf


def get_fit_dpaths(dsets, traceid='traces001', fit_desc=None,
                    excluded_sessions = ['JC110_20191004_fov1',
                                         'JC080_20190602_fov1',
                                         'JC113_20191108_fov1', 
                                         'JC113_20191108_fov2'],
                    rootdir='/n/coxfs01/2p-data'):
    '''
    rfdata: (dataframe)
        Metadata (subset of 'sdata') of all datasets to include in current analysis
        
    Gets paths to fit_results.pkl, which contains all (fit-able) results for each cell.
    Adds new column of paths to rfdata.
    '''
    assert fit_desc is not None, "No fit-desc specified!"
    
    rfmeta = dsets.copy()
    fit_these = []
    dpaths = {}
    unknown = []
    for (va, datakey), g in dsets.groupby(['visual_area','datakey']):
        session, animalid, fovnum = hutils.split_datakey_str(datakey)
        fov='FOV%i_zoom2p0x' % fovnum
        if datakey in excluded_sessions:
            rfmeta = rfmeta.drop(g.index)
            continue
        rfruns = g['experiment'].unique()
        for rfname in rfruns:
            curr_rfname = 'gratings' if int(session) < 20190511 else rfname
            fpath = glob.glob(os.path.join(rootdir, animalid, session, '*%s' % fov, 
                                        'combined_%s_*' % curr_rfname, 'traces', '%s*' % traceid, 
                                        'receptive_fields', fit_desc, 'fit_results.pkl'))
            if len(fpath) > 0:
                assert len(fpath)==1, "Too many paths: %s" % str(fpath)
                dpaths[(va, datakey, rfname)] = fpath[0] #['-'.join([animalid, session, fov, rfname])] = fpath[0]
            elif len(fpath) == 0:
                fit_these.append((animalid, session, fov, rfname))
            else:
                print("[%s] %s - warning: unknown file paths" % (datakey, rfname))
    print("N dpaths: %i, N unfit: %i" % (len(dpaths), len(fit_these)))
    print("N datasets included: %i, N sessions excluded: %i" % (rfmeta.shape[0], len(excluded_sessions)))
   
    rmeta= pd.concat([g for (va, dk, rfname), g in rfmeta.groupby(['visual_area', 'datakey', 'experiment'])\
                if (va, dk, rfname) in dpaths.keys()])
    rmeta['path'] = None
    for (va, dk, rfname), g in rmeta.groupby(['visual_area', 'datakey', 'experiment']):
        curr_fpath = dpaths[(va, dk, rfname)]
        rmeta.loc[g.index, 'path'] = curr_fpath
        
    rmeta = rmeta.drop_duplicates().reset_index(drop=True)
    
    return rmeta, fit_these


def aggregate_rfdata(rf_dsets, assigned_cells, traceid='traces001', 
                        fit_desc='fit-2dgaus_dff-no-cutoff', reliable_only=True, verbose=False):
    # Gets all results for provided datakeys (sdata, for rfs/rfs10)
    # Aggregates results for the datakeys
    # assigned_cells:  cells assigned by visual area

    # Only try to load rfdata if we can find fit + evaluation results
    rfmeta, no_fits = get_fit_dpaths(rf_dsets, traceid=traceid, fit_desc=fit_desc)
    rfdf = cycle_and_load(rfmeta, assigned_cells, reliable_only=reliable_only,
                            fit_desc=fit_desc, traceid=traceid, verbose=verbose)
    rfdf = rfdf.reset_index(drop=True)

    return rfdf

def add_rf_positions(rfdf, calculate_position=False, traceid='traces001'):
    '''
    Add ROI position info to RF dataframe (converted and pixel-based).
    Set calculate_position=True, to re-calculate.
    '''
    import roi_utils as rutils
    print("Adding RF position info...")
    pos_params = ['fov_xpos', 'fov_xpos_pix', 'fov_ypos', 'fov_ypos_pix', 'ml_pos','ap_pos']
    for p in pos_params:
        rfdf[p] = ''
    p_list=[]
    #for (animalid, session, fovnum, exp), g in rfdf.groupby(['animalid', 'session', 'fovnum', 'experiment']):
    for (va, dk, exp), g in rfdf.groupby(['visual_area', 'datakey', 'experiment']):
        session, animalid, fovnum = split_datakey_str(dk)

        fcoords = rutils.load_roi_coords(animalid, session, 'FOV%i_zoom2p0x' % fovnum,
                                  traceid=traceid, create_new=False)

        #for ei, e_df in g.groupby(['experiment']):
        cell_ids = g['cell'].unique()
        p_ = fcoords['roi_positions'].loc[cell_ids]
        for p in pos_params:
            rfdf.loc[g.index, p] = p_[p].values

    return rfdf


def combine_rfs_single(rfdf):
    '''Combine RF data so only 1 RF experiment per datakey'''
    final_rfdf=None
    rf_=[]
    for (visual_area, datakey), curr_rfdf in rfdf.groupby(['visual_area', 'datakey']):
        final_rf=None
        if visual_area in ['V1', 'Lm']:
            if 'rfs' in curr_rfdf['experiment'].values:
                final_rf = curr_rfdf[curr_rfdf.experiment=='rfs'].copy()
            else:
                final_rf = curr_rfdf[curr_rfdf.experiment=='rfs10'].copy() 
        else:
            if 'rfs10' in curr_rfdf['experiment'].values:
                final_rf = curr_rfdf[curr_rfdf.experiment=='rfs10'].copy()
            else:
                final_rf = curr_rfdf[curr_rfdf.experiment=='rfs'].copy()
        rf_.append(final_rf)

    final_rfdf = pd.concat(rf_).reset_index(drop=True)

    return final_rfdf


def average_rfs_select(rfdf):
    final_rfdf=None
    rf_=[]
    for (visual_area, datakey), curr_rfdf in rfdf.groupby(['visual_area', 'datakey']):
        final_rf=None
        if visual_area=='V1' and 'rfs' in curr_rfdf['experiment'].values:
            final_rf = curr_rfdf[curr_rfdf.experiment=='rfs'].copy()
        elif visual_area in ['Lm', 'Li']:
            # Which cells have receptive fields
            rois_ = curr_rfdf['cell'].unique()

            # Means by cell id (some dsets have rf-5 and rf10 measurements, average these)
            meanrf = curr_rfdf.groupby(['cell']).mean().reset_index()
            mean_thetas = curr_rfdf.groupby(['cell'])['theta']\
                                    .apply(spstats.circmean, low=0, high=2*np.pi).values
            meanrf['theta'] = mean_thetas
            meanrf['visual_area'] = visual_area
            meanrf['experiment'] = ['average_rfs' if len(g['experiment'].values)>1 \
                                    else str(g['experiment'].unique()[0]) \
                                    for c, g in curr_rfdf.groupby(['cell'])]
            # Add the meta/non-numeric info
            non_num = [c for c in curr_rfdf.columns if c \
                            not in meanrf.columns and c!='experiment']
            metainfo = pd.concat([g[non_num].iloc[0] for c, g in \
                            curr_rfdf.groupby(['cell'])], axis=1).T.reset_index(drop=True)
            final_rf = pd.concat([metainfo, meanrf], axis=1)            
            final_rf = update_rf_metrics(final_rf, scale_sigma=True)
        rf_.append(final_rf)

    final_rfdf = pd.concat(rf_).reset_index(drop=True)

    return final_rfdf


def average_rfs(rfdf):
    final_rfdf=None
    rf_=[]
    for (visual_area, datakey), curr_rfdf in rfdf.groupby(['visual_area', 'datakey']):
        # Means by cell id (some dsets have rf-5 and rf10 measurements, average these)
        meanrf = curr_rfdf.groupby(['cell']).mean().reset_index()
        mean_thetas = curr_rfdf.groupby(['cell'])['theta'].apply(spstats.circmean, low=0, high=2*np.pi).values
        meanrf['theta'] = mean_thetas
        meanrf['visual_area'] = visual_area # reassign area
        meanrf['experiment'] = ['average_rfs' if len(g['experiment'].values)>1 \
                                else str(g['experiment'].unique()) \
                                for c, g in curr_rfdf.groupby(['cell'])]
        #meanrf['experiment'] = ['average_rfs' for _ in np.arange(0, len(assigned_with_rfs))]

        # Add the meta/non-numeric info
        non_num = [c for c in curr_rfdf.columns if c not in meanrf.columns and c!='experiment']
        metainfo = pd.concat([g[non_num].iloc[0] for c, g in \
                            curr_rfdf.groupby(['cell'])], axis=1).T.reset_index(drop=True)
        final_rf = pd.concat([metainfo, meanrf], axis=1)
        final_rf = update_rf_metrics(final_rf, scale_sigma=True)
        rf_.append(final_rf)

    final_rfdf = pd.concat(rf_).reset_index(drop=True)

    return final_rfdf


# ----------------------------------------------------------------------------
# RFMAPS
# ----------------------------------------------------------------------------
#def get_trials_by_cond(labels):
#    # Get single value for each trial and sort by config:
#    trials_by_cond = dict()
#    for k, g in labels.groupby(['config']):
#        trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])
#
#    return trials_by_cond
#

def load_rfmap_array(rfdir, do_spherical_correction=True):  
    '''
    Rows=positions, Cols=cells -- unraveled RF map array 
    (averaged across trials per position)
    '''
    rfarray_dpath = os.path.join(rfdir, 'rfmap_array.pkl')    
    rfmaps_arr=None
    if os.path.exists(rfarray_dpath):
        #print("-- loading: %s" % rfarray_dpath)
        with open(rfarray_dpath, 'rb') as f:
            rfmaps_arr = pkl.load(f, encoding='latin1')
    return rfmaps_arr

def save_rfmap_array(rfmaps_arr, rfdir):  
    rfarray_dpath = os.path.join(rfdir, 'rfmap_array.pkl')    
    with open(rfarray_dpath, 'wb') as f:
        pkl.dump(rfmaps_arr, f, protocol=2) #pkl.HIGHEST_PROTOCOL)
    return


def reshape_array_for_nynx(rfmap_values, nx, ny):
    if isinstance(rfmap_values, (pd.Series, pd.DataFrame)):
        rfmap_orig = np.reshape(rfmap_values.values, (nx, ny)).T
    else:
        rfmap_orig = rfmap_values.reshape(nx, ny).T
    return rfmap_orig.ravel()


def group_trial_values_by_cond(trialdata):
    '''
    Get average across trials per cond -- return, Nconds X Nrois
    Removes config and trial columns (if exist). 
    '''
    rois = [r for r in trialdata.columns.tolist() if r not in ['config', 'trial']]
    meanrs = trialdata.groupby('config').mean().reset_index()
    trialdata = meanrs[rois]
 
    return trialdata 



def get_lin_coords(resolution=[1080, 1920], cm_to_deg=True, 
                   xlim_degrees=(-59.7, 59.7), ylim_degrees=(-33.6, 33.6)):
    """
    **From: https://github.com/zhuangjun1981/retinotopic_mapping (Monitor initialiser)

    Parameters
    ----------
    resolution : tuple of two positive integers
        value of the monitor resolution, (pixel number in height, pixel number in width)
    dis : float
         distance from eyeball to monitor (in cm)
    mon_width_cm : float
        width of monitor (in cm)
    mon_height_cm : float
        height of monitor (in cm)
    C2T_cm : float
        distance from gaze center to monitor top
    C2A_cm : float
        distance from gaze center to anterior edge of the monitor
    center_coordinates : tuple of two floats
        (altitude, azimuth), in degrees. the coordinates of the projecting point
        from the eye ball to the monitor. This allows to place the display monitor
        in any arbitrary position.
    visual_field : str from {'right','left'}, optional
        the eye that is facing the monitor, defaults to 'left'
    """
    mon_height_cm = 58.
    mon_width_cm = 103.
    # resolution = [1080, 1920]
    visual_field = 'left'
    
    C2T_cm = mon_height_cm/2. #np.sqrt(dis**2 + mon_height_cm**2)
    C2A_cm = mon_width_cm/2.
    
    # distance form projection point of the eye to bottom of the monitor
    C2B_cm = mon_height_cm - C2T_cm
    # distance form projection point of the eye to right of the monitor
    C2P_cm = -C2A_cm #mon_width_cm - C2A_cm

    map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]),
                                           range(resolution[0]))

    if visual_field == "left":
        #map_x = np.linspace(C2A_cm, -1.0 * C2P_cm, resolution[1])
        map_x = np.linspace(C2P_cm, C2A_cm, resolution[1])

    if visual_field == "right":
        map_x = np.linspace(-1 * C2A_cm, C2P_cm, resolution[1])

    map_y = np.linspace(C2T_cm, -1.0 * C2B_cm, resolution[0])
    old_map_x, old_map_y = np.meshgrid(map_x, map_y, sparse=False)

    lin_coord_x = old_map_x
    lin_coord_y = old_map_y    
    
    if cm_to_deg:
        xmin_cm = lin_coord_x.min(); xmax_cm = lin_coord_x.max();
        ymin_cm = lin_coord_y.min(); ymax_cm = lin_coord_y.max();
        
        xmin_deg, xmax_deg = xlim_degrees
        ymin_deg, ymax_deg = ylim_degrees
        
        lin_coord_x = hutils.convert_range(lin_coord_x, oldmin=xmin_cm, oldmax=xmax_cm, 
                                           newmin=xmin_deg, newmax=xmax_deg)
        lin_coord_y = hutils.convert_range(lin_coord_y, oldmin=ymin_cm, oldmax=ymax_cm, 
                                           newmin=ymin_deg, newmax=ymax_deg)
    return lin_coord_x, lin_coord_y


def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def get_spherical_coords(cart_pointsX=None, cart_pointsY=None, cm_to_degrees=True,
                    resolution=(1080, 1920),
                   xlim_degrees=(-59.7, 59.7), ylim_degrees=(-33.6, 33.6)):
    '''
    From spherical_correction analyses. Convert monitor linear coordinates to spherical
    '''
    # Monitor size and position variables
    width_cm = 103; #%56.69;  % 103 width of screen, in cm
    height_cm = 58; #%34.29;  % 58 height of screen, in cm
    pxXmax = resolution[1] #1920; #%200; % number of pixels in an image that fills the whole screen, x
    pxYmax = resolution[0] #1080; #%150; % number of pixels in an image that fills the whole screen, y

    # Eye info
    cx = width_cm/2. # % eye x location, in cm
    cy = height_cm/2. # %11.42; % eye y location, in cm
    eye_dist = 30.; #% in cm

    # Distance to bottom of screen, along the horizontal eye line
    zdistBottom = np.sqrt((cy**2) + (eye_dist**2)) #; %24.49;     % in cm
    zdistTop    = np.sqrt((cy**2) + (eye_dist**2)) #; %14.18;     % in cm

    # Internal conversions
    top = height_cm-cy;
    bottom = -cy;
    right = cx;
    left = cx - width_cm;

    if cart_pointsX is None or cart_pointsY is None:
        [xi, yi] = np.meshgrid(np.arange(0, pxXmax), np.arange(0, pxYmax))
        print(xi.shape, yi.shape)
        cart_pointsX = left + (float(width_cm)/pxXmax)*xi;
        cart_pointsY = top - (float(height_cm)/pxYmax)*yi;
        cart_pointsZ = zdistTop + ((zdistBottom-zdistTop)/float(pxYmax))*yi
    else:
        cart_pointsZ = zdistTop + ((zdistBottom-zdistTop)/float(pxYmax))*cart_pointsY

    if cm_to_degrees:
        xmin_cm=cart_pointsX.min(); xmax_cm=cart_pointsX.max();
        ymin_cm=cart_pointsY.min(); ymax_cm=cart_pointsY.max();
        xmin_deg, xmax_deg = xlim_degrees
        ymin_deg, ymax_deg = ylim_degrees
        cart_pointsX = hutils.convert_range(cart_pointsX, oldmin=xmin_cm, oldmax=xmax_cm, 
                                       newmin=xmin_deg, newmax=xmax_deg)
        cart_pointsY = hutils.convert_range(cart_pointsY, oldmin=ymin_cm, oldmax=ymax_cm, 
                                       newmin=ymin_deg, newmax=ymax_deg)
        cart_pointsZ = hutils.convert_range(cart_pointsZ, oldmin=ymin_cm, oldmax=ymax_cm, 
                                       newmin=ymin_deg, newmax=ymax_deg)

    sphr_pointsTh, sphr_pointsPh, sphr_pointsR = cart2sph(cart_pointsZ, cart_pointsX, cart_pointsY)
    #sphr_pointsTh, sphr_pointsPh, sphr_pointsR = cart2sph(cart_pointsX, cart_pointsY, cart_pointsZ)

    return cart_pointsX, cart_pointsY, sphr_pointsTh, sphr_pointsPh


def get_scaled_sigmas(grid_points, new_values, x0, y0, sx, sy, th, convert=True):
    '''
    From upsampled RF map array, calculate RF parameters
    '''
    x0_scaled, y0_scaled = interpolate.griddata(grid_points, new_values, (x0, y0))
    x0_scaled, y0_scaled = interpolate.griddata(grid_points, new_values, (x0, y0))

    
    # Get flanking points spanned by sx, sy
    try:
        sx_linel, sx_line2, sy_line1, sy_line2 = get_endpoints_from_sigma(
                                                        x0, y0, sx, sy, th, 
                                                        scale_sigma=False)
    except Exception as e:
        return None, None, None, None

    # Get distances
    if convert:
        # Convert coordinates of array to new coordinate system
        sx_x1_sc, sx_y1_sc = interpolate.griddata(grid_points, new_values, sx_linel) 
        sx_x2_sc, sx_y2_sc = interpolate.griddata(grid_points, new_values, sx_line2)
        sx_scaled = math.hypot(sx_x2_sc - sx_x1_sc, sx_y2_sc - sx_y1_sc)
    else:
        #sx_scaled = math.hypot(sx_x2 - sx_x1, sx_y2 - sx_y1)
        sx_scaled = math.hypot(sx_line2[0] - sx_linel[0], sx_line2[1] - sx_linel[1])

    if convert:
        sy_x1_sc, sy_y1_sc = interpolate.griddata(grid_points, new_values, sy_line1)
        sy_x2_sc, sy_y2_sc = interpolate.griddata(grid_points, new_values, sy_line2)
        sy_scaled = math.hypot(sy_x2_sc - sy_x1_sc, sy_y2_sc - sy_y1_sc)
    else:
        #sy_scaled = math.hypot(sy_x2 - sy_x1, sy_y2 - sy_y1)
        sy_scaled = math.hypot(sy_line2[0] - sy_line1[0], sy_line2[1] - sy_line1[1])
    
    return x0_scaled, y0_scaled, abs(sx_scaled), abs(sy_scaled)


def get_endpoints_from_sigma(x0, y0, sx, sy, th, scale_sigma=False, sigma_scale=2.35):
    '''
    Calculate major and minor axis lines spanning sigma_x and sigma_y
    '''    
    sx = sx*sigma_scale if scale_sigma else sx
    sy = sy*sigma_scale if scale_sigma else sy
    
    sx_x1, sx_y1 = (x0-(sx/2.)*np.cos(th), y0-(sx/2.)*np.sin(th)) # Get min half
    sx_x2, sx_y2 = (x0+(sx/2.)*np.cos(th), y0+(sx/2.)*np.sin(th)) # Get other half

    th_orth = th + (np.pi/2.)
    sy_x1, sy_y1 = (x0-(sy/2.)*np.cos(th_orth), y0-(sy/2.)*np.sin(th_orth))
    sy_x2, sy_y2 = (x0+(sy/2.)*np.cos(th_orth), y0+(sy/2.)*np.sin(th_orth))

    lA = (sy_x1, sy_y1), (sy_x2, sy_y2) # The line along y-axis
    lB = (sx_x1, sx_y1), (sx_x2, sx_y2) # The line along x-axis
    ang_deg = ang(lA, lB)
    assert ang_deg==90.0, "bad angle calculation (%.1f)..." % ang_deg

    return (sx_x1, sx_y1), (sx_x2, sx_y2), (sy_x1, sy_y1), (sy_x2, sy_y2)


def ang(lineA, lineB):
    '''
    Calculate angle between 2 lines, given their coords as:
    lA = (A_x1, A_y1), (A_x2, A_y2)
    lB = (B_x1, B_y1), (B_x2, B_y2)
    
    '''
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = np.dot(vA, vB)
    # Get magnitudes
    magA = np.dot(vA, vA)**0.5
    magB = np.dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360.

    if ang_deg-180>=0:
        # As in if statement
        return round(360 - ang_deg, 2)
    else: 
        return round(ang_deg, 2)



def resample_map(rfmap, lin_coord_x, lin_coord_y, row_vals=None, col_vals=None,
                 resolution=(1080,1920)):
    '''Get resampled RF map in screen coords
    '''
    screen_bounds_pix = get_screen_lim_pixels(lin_coord_x, lin_coord_y, 
                                            row_vals=row_vals, col_vals=col_vals)
    (pix_bottom_edge, pix_left_edge, pix_top_edge, pix_right_edge) = screen_bounds_pix

    # We don't stimulate every pixel, so get BBox of stimulated positions
    stim_height = pix_bottom_edge-pix_top_edge+1
    stim_width = pix_right_edge-pix_left_edge+1
    stim_resolution = [stim_height, stim_width]

    # Upsample rfmap to match resolution of stimulus-occupied space
    rfmap_r = cv2.resize(rfmap.astype(float), (stim_resolution[1], stim_resolution[0]), 
                          interpolation=cv2.INTER_NEAREST)

    rfmap_to_screen = np.ones((resolution[0], resolution[1]))*np.nan
    rfmap_to_screen[pix_top_edge:pix_bottom_edge+1, pix_left_edge:pix_right_edge+1] = rfmap_r

    return rfmap_to_screen

def trim_resampled_map(rfmap_r, screen_bounds_pix): 
    '''
    After resampling the RF map (in downsampled pixel resolution coords),
    trim off the reshaped (or also warped) map to match dims of screen portions that 
    were stimulated based on non-warped
    '''
    #screen_bounds_pix = get_screen_lim_pixels(lin_coord_x, lin_coord_y, 
    #                                        row_vals=row_vals, col_vals=col_vals)
    (pix_bottom_edge, pix_left_edge, pix_top_edge, pix_right_edge) = screen_bounds_pix

    rfmap_trim  = rfmap_r[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge]
    return rfmap_trim


def sphr_correct_maps(rfmaps_arr, fit_params, use_lin=True, 
                        ds_factor=3, multiproc=False):
    '''
    rfmaps_arr:  array (dataframe) of rfmap (vector) x nrois
    do spherical correction, return rfmap array

    '''
    # ds_factor = fit_params['downsample_factor']
    col_vals = fit_params['col_vals']
    row_vals = fit_params['row_vals']
    # Downsample screen resolution
    resolution_ds = [int(i/ds_factor) for i in \
                     fit_params['screen']['resolution'][::-1]]
    print("Screen res (ds=%ix): [%i, %i]" \
                    % (ds_factor, resolution_ds[0], resolution_ds[1]))
    # Get Spherical coordinate mapping
    if fit_params['use_linear']:
        # Get linear coordinates in degrees (downsampled)
        lin_x, lin_y = get_lin_coords(resolution=resolution_ds, cm_to_deg=True)
        cart_x, cart_y, sphr_th, sphr_ph = get_spherical_coords(
                                        cart_pointsX=lin_x, cart_pointsY=lin_y,
                                        cm_to_degrees=False)
    else:
        cart_x, cart_y, sphr_th, sphr_ph = get_spherical_coords(
                                        cart_pointsX=None, cart_pointsY=None,
                                        cm_to_degrees=True, resolution=resolution_ds) 

    args=(cart_x, cart_y, sphr_th, sphr_ph, resolution_ds, row_vals, col_vals,)
    rfmaps_arr0 = rfmaps_arr.apply(warp_spherical_fromarr, axis=0, args=args)

    return rfmaps_arr0.reset_index(drop=True)

def sphr_correct_maps_mp(avg_resp_by_cond, fit_params, use_lin=True, n_processes=2, 
                            test_subset=False):
    
    if test_subset:
        roi_list=[92, 249, 91, 162, 61, 202, 32, 339]
        df_ = avg_resp_by_cond[roi_list]
    else:
        df_ = avg_resp_by_cond.copy()
    print("Parallel", df_.shape)

    df = parallelize_dataframe(df_, sphr_correct_maps, \
                               fit_params, use_lin, n_processes=n_processes)

    return df

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_

def parallelize_dataframe(df, func, fit_params, use_lin=True, n_processes=4):
    #cart_x=None, cart_y=None, sphr_th=None, sphr_ph=None,
    #                      row_vals=None, col_vals=None, resolution=None, n_processes=4):
    results = []
    terminating = mp.Event()
    
    df_split = np.array_split(df, n_processes, axis=1) # chunk columns
    pool = mp.Pool(processes=n_processes, initializer=initializer, initargs=(terminating,))
    try:
        results = pool.map(partial(func, fit_params=fit_params, use_lin=use_lin), df_split)
        print("done!")
    except KeyboardInterrupt:
        pool.terminate()
        print("terminating")
    finally:
        pool.close()
        pool.join()
  
    print(results[0].shape)
    df = pd.concat(results, axis=1)
    print(df.shape)
    return df #results


def warp_spherical_fromarr(rfmap_values, cart_x=None, cart_y=None, 
                           sphr_th=None, sphr_ph=None, resolution=(1080, 1920),
                           row_vals=None, col_vals=None,normalize_range=True):  
    '''
    Given rfmap values as vector, reshape, warp, return as array.
    '''
    nx = len(col_vals)
    ny = len(row_vals)
    #rfmap = rfmap_values.values.reshape(ny, nx) #rfmap_values.reshape(nx, ny).T
    rfmap = rfmap_values.values.reshape(nx, ny).T #rfmap_values.reshape(nx, ny).T

    # Upsample to screen resolution (pixels)
    rfmap_orig = resample_map(rfmap, cart_x, cart_y, #lin_coord_x, lin_coord_y, 
                            row_vals=row_vals, col_vals=col_vals,
                            resolution=resolution) 
    # Warp upsampled: convert/interp to sphr_thr, sphr_ph (from cart_x, cart_y)
    rfmap_warp = warp_spherical(rfmap_orig, sphr_th, sphr_ph, cart_x, cart_y,
                                normalize_range=normalize_range, method='linear')
    # Crop 
    screen_bounds_pix = get_screen_lim_pixels(cart_x, cart_y, 
                                              row_vals=row_vals, col_vals=col_vals)
    rfmap_trim  = trim_resampled_map(rfmap_warp, screen_bounds_pix)

    # Resize back to known grid
    rfmap_resize = cv2.resize(rfmap_trim, (nx, ny))

    #  
    # flatten w/ order='F' so output format is same as rfmaps_array 
    # to make map: rfarray.reshape((nx, ny)).T (nx=ncols, ny=nrows)
    return rfmap_resize.flatten(order='F') 





# plotting
def get_centered_screen_points(screen_xlim, nc):
    col_pts = np.linspace(screen_xlim[0], screen_xlim[1], nc+1) # n points for NC columns
    pt_spacing = np.mean(np.diff(col_pts)) 
    # Add half point spacing on either side of boundary points to center the column points
    xlim_min = col_pts.min() - (pt_spacing/2.) 
    xlim_max = col_pts.max() + (pt_spacing/2.)
    return (xlim_min, xlim_max)


def plot_rfs_to_screen_pretty(fitdf, sdf, screen, sigma_scale=2.35, fit_roi_list=[], ax=None,
                             ellipse_lw=1, roi_colors=None):
    '''
    fitdf:  dataframe w/ converted fit params
    '''
    rows='ypos'
    cols='xpos'

    screen_left = -1*screen['azimuth_deg']/2.
    screen_right = screen['azimuth_deg']/2.
    screen_top = screen['altitude_deg']/2.
    screen_bottom = -1*screen['altitude_deg']/2.
    
    row_vals = sorted(sdf[rows].unique())
    col_vals = sorted(sdf[cols].unique())
    tile_sz = np.mean(np.diff(row_vals))
    screen_rect = mpl.patches.Rectangle(
                ( min(col_vals)-tile_sz/2., min(row_vals)-tile_sz/2.), 
                max(col_vals)-min(col_vals)+tile_sz,
                max(row_vals)-min(row_vals)+tile_sz, 
                facecolor='none', edgecolor='k', lw=0.5)
    ax.add_patch(screen_rect)
    if ax is None:
        fig, ax = pl.subplots(figsize=(12, 6))
        fig.patch.set_visible(False) #(False) #('off')
    if roi_colors is None:
        roi_colors=sns.color_palette('bone', n_colors=len(fit_roi_list)+5) 
    for rid, rcolor in zip(fit_roi_list, roi_colors):
        ell = mpl.patches.Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]),
                      abs(fitdf['sigma_x'][rid])*sigma_scale, 
                      abs(fitdf['sigma_y'][rid])*sigma_scale,
                      angle=np.rad2deg(fitdf['theta'][rid]))
        ell.set_alpha(1.0)
        ell.set_linewidth(ellipse_lw)
        ell.set_edgecolor(rcolor)
        ell.set_facecolor('none')
        ax.add_patch(ell)
    ax.set_ylim([screen_bottom, screen_top])
    ax.set_xlim([screen_left, screen_right])
    # print(screen_bottom, screen_top, screen_left, screen_right)
 
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

    #summary_str = "Avg sigma-x, -y: (%.2f, %.2f)\nAvg RF size: %.2f (min: %.2f, max: %.2f)" % (np.mean(majors), np.mean(minors), np.mean([np.mean(majors), np.mean(minors)]), avg_rfs.min(), avg_rfs.max())
    #pl.text(ax.get_xlim()[0]-12, ax.get_ylim()[0]-8, summary_str, ha='left', rotation=0, wrap=True)
    return ax


    
def plot_rf_map(rfmap, cmap='inferno', ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    im = ax.imshow(rfmap, cmap='inferno')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation='vertical')
   
    return ax

def plot_rf_ellipse(df_, ax=None, sigma_scale=2.35, scale_sigma=True):
      
    sigma_scale = sigma_scale if scale_sigma else 1.0
    if isinstance(df_, (pd.Series, pd.DataFrame)):
        params = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset', 'r2']
        amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f, r2 = df_[params] 
    else:
        amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = df_['popt']
        r2 = df_['r2']

    # Draw ellipse:  
    if ax is None:
        fig, ax = pl.subplots()
        ax.set_ylim([y0_f-sigy_f*2., y0_f+sigy_f*2.])
        ax.set_xlim([x0_f-sigx_f*2., x0_f+sigx_f*2.])
 
    ell = Ellipse((x0_f, y0_f), abs(sigx_f)*sigma_scale, abs(sigy_f)*sigma_scale, 
                    angle=np.rad2deg(theta_f), alpha=0.5, edgecolor='w') #theta_f)
    ax.add_patch(ell)
    ax.text(0, -1, 'r2=%.2f, theta=%.2f' % (r2, theta_f), color='k')

    return ax

def plot_fit_results(fitr, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    im2 = ax.imshow(fitr['pcov'])
    ax.set_yticks(np.arange(0, 7))
    ax.set_yticklabels(['amplitude', 'x0', 'y0', 'sigx', 'sigy', 'theta', 'offset'], rotation=0)
    
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    pl.colorbar(im2, cax=cax2, orientation='vertical')
   
    return ax
 

# -----------------------------------------------------------------------------
# Fitting functions
# -----------------------------------------------------------------------------


def twoD_Gaussian(X, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = X
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    # b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    b = (np.sin(2*theta))/(2*sigma_x**2) - (np.sin(2*theta))/(2*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
#                            + c*((y-yo)**2)))
    g = offset + amplitude*np.exp( -a*((x-xo)**2) - b*(x-xo)*(y-yo) - c*((y-yo)**2) )
    return g.ravel()


def twoD_gauss(X, b, x0, y0, sigma_x, sigma_y, theta, a):
    x, y = X
    res = a + b * np.exp( -( ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta)) / (np.sqrt(2)*sigma_x) )**2 - ( ( -(x-x0)*np.sin(theta) + (y-y0)*np.cos(theta) ) / (np.sqrt(2)*sigma_y) )**2 )
    
    return res.ravel()



class MultiplePeaks(Exception): pass
class NoPeaksFound(Exception): pass

from scipy.interpolate import splrep, sproot, splev, interp1d


def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = max(y)/2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)
    #print(roots)
    if len(roots) > 2:
        return None
#        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
#                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        return None
#        raise NoPeaksFound("No proper peaks were found in the data set; likely "
#                "the dataset is flat (e.g. all zeros).")
    else:
        return abs(roots[1] - roots[0])


def raw_fwhm(arr):
    
    interpf = interp1d(np.linspace(0, len(arr)-1, num=len(arr)), arr, kind='linear')
    xnew = np.linspace(0, len(arr)-1, num=len(arr)*3)
    ynew = interpf(xnew)
    
    hm = ((ynew.max() - ynew.min()) / 2 ) + ynew.min() # half-max
    pk = ynew.argmax() # find peak

    if pk == 0:
        r2 = pk + np.abs(ynew[pk:] - hm).argmin()
        return abs(xnew[r2]*2)
    else:
        r1 = np.abs(ynew[0:pk] - hm).argmin() # find index of local min on left
        r2 = pk + np.abs(ynew[pk:] - hm).argmin() # find index of local min on right
        
        return abs(xnew[r2]-xnew[r1]) # return full width
    



def fit_rfs(rfmaps_arr, fit_params, #row_vals=[], col_vals=[], fitparams=None,
            roi_list=None, #scale_sigma=True,
            #rf_results_fpath='/tmp/fit_results.pkl', 
            data_identifier='METADATA'):
            #response_thr=None):

    '''
    Main fitting function.    
    Saves 2 output files for fitting: 
        fit_results.pkl 
        fit_params.json
    '''
    print("@@@ doing rf fits @@@")
    scale_sigma = fit_params['scale_sigma']
    sigma_scale = fit_params['sigma_scale']
    row_vals = fit_params['row_vals']
    col_vals = fit_params['col_vals']

    rfdir = fit_params['rfdir'] #os.path.split(rf_results_fpath)[0]    
    rf_results_fpath = os.path.join(rfdir, 'fit_results.pkl')
    rf_params_fpath = os.path.join(rfdir, 'fit_params.json')

    # Save params
    with open(rf_params_fpath, 'w') as f:
        json.dump(fit_params, f, indent=4, sort_keys=True)
    
    # Create subdir for saving each roi's fit
    if not os.path.exists(os.path.join(rfdir, 'roi_fits')):
        os.makedirs(os.path.join(rfdir, 'roi_fits'))

    roi_list = rfmaps_arr.columns.tolist()
    bad_rois = [r for r in roi_list if rfmaps_arr.max()[r] > 1.0]
    print("... %i bad rois (skipping: %s)" % (len(bad_rois), str(bad_rois)))
    if len(bad_rois) > 0:
        badr_fpath = os.path.join(rfdir.split('/receptive_fields/')[0], 'funky.json')
        with open(badr_fpath, 'w') as f:
            json.dump(bad_rois, f)
     
    fit_results = {}
    for rid in roi_list:
        #print rid
        if rid in bad_rois:
            continue
        roi_fit_results, fig = plot_and_fit_roi_RF(rfmaps_arr[rid], 
                                                    row_vals, col_vals,
                                                    scale_sigma=scale_sigma, 
                                                    sigma_scale=sigma_scale) 
        fig.suptitle('roi %i' % int(rid+1))
        pplot.label_figure(fig, data_identifier)            
        figname = '%s_RF_roi%05d' % (fit_params['response_type'], int(rid+1))
        pl.savefig(os.path.join(rfdir, 'roi_fits', '%s.png' % figname))
        pl.close()    
        if len(roi_fit_results)>0: # != {}:
            fit_results[rid] = roi_fit_results
        #%
    xi = np.arange(0, len(col_vals))
    yi = np.arange(0, len(row_vals))
    xx, yy = np.meshgrid(xi, yi)
        
    with open(rf_results_fpath, 'wb') as f:
        pkl.dump(fit_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    return fit_results, fit_params


def get_rf_map(response_vector, ncols, nrows):
    #if do_spherical_correction:
    #coordmap_r = np.reshape(response_vector, (nrows, ncols))
    #else:
    coordmap_r = np.reshape(response_vector, (ncols, nrows)).T
    
    return coordmap_r

#
def plot_and_fit_roi_RF(response_vector, row_vals, col_vals, 
                        plot_roi_fit=True, cmap='inferno',
                        min_sigma=2.5, max_sigma=50, 
                        sigma_scale=2.35, scale_sigma=True):
    '''
    Fits RF for single ROI. 
    Note: This does not filter by R2, includes all fit-able.
 
    Returns a dict with fit info if doesn't error out.
    
    Sigma must be [2.5, 50]...
    '''
    sigma_scale = sigma_scale if scale_sigma else 1.0
    results = {}
    fig = None

    # Do fit 
    # ---------------------------------------------------------------------
    nrows = len(row_vals)
    ncols = len(col_vals)
    rfmap = get_rf_map(response_vector.values, ncols, nrows)
    #np.reshape(response_vector.values, (nrows, ncols)) 
    fitr, fit_y = do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))

    xres = np.mean(np.diff(sorted(row_vals)))
    yres = np.mean(np.diff(sorted(col_vals)))
    min_sigma = xres/2.0
    
    if fitr['success']:
        amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
        if any(s < min_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale])\
            or any(s > max_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale]):
            fitr['success'] = False

    if plot_roi_fit:    
        fig, axes = pl.subplots(1,2, figsize=(8, 4)) # pl.figure()
        ax = axes[0]
        ax2 = axes[1] 
        ax = plot_rf_map(rfmap, ax=ax, cmap=cmap)
        if fitr['success']:
            # Draw ellipse: 
            ax = plot_rf_ellipse(fitr, ax, scale_sigma=scale_sigma)                
            # Visualize fit results:
            ax2 = plot_fit_results(fitr, ax=ax2)         
            # Adjust subplot:
            bbox1 = ax.get_position()
            subplot_ht = bbox1.height
            bbox2 = ax2.get_position()
            ax2.set_position([bbox2.x0, bbox1.y0, subplot_ht, subplot_ht])
        else:
            ax.text(0, -1, 'no fit')
            ax2.axis('off') 
        pl.subplots_adjust(wspace=0.3, left=0.1, right=0.9)

    xi = np.arange(0, len(col_vals))
    yi = np.arange(0, len(row_vals))
    xx, yy = np.meshgrid(xi, yi)
    if fitr['success']:
        results = {'amplitude': amp_f,
                   'x0': x0_f, 'y0': y0_f,'sigma_x': sigx_f, 'sigma_y': sigy_f,
                   'theta': theta_f, 'offset': offset_f, 'r2': fitr['r2'],
                   'fit_y': fit_y,'fit_r': fitr,'data': rfmap} 
    
    return results, fig
    
def do_2d_fit(rfmap, nx=None, ny=None, verbose=False):

    #TODO:  Instead of finding critical pts w/ squared RF map, do:
    #    mean subtraction, followed by finding max delta from the  ____
    #nx=len(col_vals); ny=len(row_vals);
    # Set params for fit:
    xi = np.arange(0, nx)
    yi = np.arange(0, ny)
    popt=None; pcov=None; fitr=None; r2=None; success=None;
    xx, yy = np.meshgrid(xi, yi)
    initial_guess = None
    try:
        #amplitude = (rfmap**2).max()
        #y0, x0 = np.where(rfmap == rfmap.max())
        #y0, x0 = np.where(rfmap**2. == (rfmap**2.).max())
        #print "x0, y0: (%i, %i)" % (int(x0), int(y0))    

        rfmap_sub = np.abs(rfmap - np.nanmean(rfmap))
        y0, x0 = np.where(rfmap_sub == np.nanmax(rfmap_sub))
        amplitude = rfmap[y0, x0][0]
        #print "x0, y0: (%i, %i) | %.2f" % (int(x0), int(y0), amplitude)    
        try:
            #sigma_x = fwhm(xi, (rfmap**2).sum(axis=0))
            #sigma_x = fwhm(xi, abs(rfmap.sum(axis=0) - rfmap.sum(axis=0).mean()) )
            sigma_x = fwhm(xi, np.nansum(rfmap_sub, axis=0) )
            assert sigma_x is not None
        except AssertionError:
            #sigma_x = raw_fwhm(rfmap.sum(axis=0)) 
            sigma_x = raw_fwhm( np.nansum(rfmap_sub, axis=0) ) 
        try:
            sigma_y = fwhm(yi, np.nansum(rfmap_sub, axis=1))
            assert sigma_y is not None
        except AssertionError: #Exception as e:
            sigma_y = raw_fwhm(np.nansum(rfmap_sub, axis=1))
        #print "sig-X, sig-Y:", sigma_x, sigma_y
        theta = 0
        offset=0
        initial_guess = (amplitude, int(x0), int(y0), sigma_x, sigma_y, theta, offset)
        valid_ixs = ~np.isnan(rfmap)
        popt, pcov = opt.curve_fit(twoD_gauss, (xx[valid_ixs], yy[valid_ixs]), 
                                    rfmap[valid_ixs].ravel(), 
                                    p0=initial_guess, maxfev=2000)
        fitr = twoD_gauss((xx, yy), *popt)

        # Get residual sum of squares 
        residuals = rfmap.ravel() - fitr
        ss_res = np.nansum(residuals**2)
        ss_tot = np.nansum((rfmap.ravel() - np.nanmean(rfmap.ravel()))**2)
        r2 = 1 - (ss_res / ss_tot)
        #print(r2)
        if len(np.where(fitr > fitr.min())[0]) < 2 or pcov.max() == np.inf or r2 == 1: 
            success = False
        else:
            success = True
            # modulo theta
            #mod_theta = popt[5] % np.pi
            #popt[5] = mod_theta            
    except Exception as e:
        if verbose:
            traceback.print_exc() 
    
    return {'popt': popt, 'pcov': pcov, 'init': initial_guess, 'r2': r2, 'success': success}, fitr

#
#
def get_fit_params(datakey, run='combined_rfs_static', traceid='traces001', 
                   trace_type='corrected', response_type='dff', fit_thr=0.5, 
                   do_spherical_correction=False, ds_factor=3., use_lin=True,
                   post_stimulus_sec=0.5, sigma_scale=2.35, scale_sigma=True,
                   rootdir='/n/coxfs01/2p-data'):
    
    screen = hutils.get_screen_dims()
    session, animalid, fovnum = hutils.split_datakey_str(datakey)

    run_info, sdf = aggr.load_run_info(animalid, session, 'FOV%i_zoom2p0x' % fovnum, 
                                        run,traceid=traceid, rootdir=rootdir)
    
    run_name = run.split('_')[1] if 'combined' in run else run
    sdf= aggr.get_stimuli(datakey, run_name)
    row_vals = sorted(sdf['ypos'].unique())
    col_vals = sorted(sdf['xpos'].unique())
   
    rfdir, fit_desc = create_rf_dir(datakey, run, 
                                    traceid=traceid,
                                    response_type=response_type, 
                                    do_spherical_correction=do_spherical_correction, 
                                    fit_thr=fit_thr)
    fr = run_info['framerate'] 
    nframes_post_onset = int(round(post_stimulus_sec * fr))

    #print(fit_params) 
    fit_params = {
            'response_type': response_type,     
            'trace_type': trace_type,
            'frame_rate': fr,
            'nframes_per_trial': int(run_info['nframes_per_trial'][0]),
            'stim_on_frame': run_info['stim_on_frame'],
            'nframes_on': int(run_info['nframes_on'][0]),
            'post_stimulus_sec': post_stimulus_sec,
            'nframes_post_onset': nframes_post_onset,
            'row_spacing': np.mean(np.diff(row_vals)),
            'column_spacing': np.mean(np.diff(col_vals)),
            'fit_thr': fit_thr,
            'sigma_scale': float(sigma_scale),
            'scale_sigma': scale_sigma,
            'screen': screen,
            'row_vals': row_vals,
            'col_vals': col_vals,
            'rfdir': rfdir,
            'fit_desc': fit_desc,
            'do_spherical_correction': do_spherical_correction,
            'downsample_factor': ds_factor,
            'use_linear': use_lin
            } 
   
    with open(os.path.join(rfdir, 'fit_params.json'), 'w') as f:
        json.dump(fit_params, f, indent=4, sort_keys=True)
    
    return fit_params

def fit_2d_rfs(datakey, run, traceid, 
                        reload_data=False, create_new=False,
                        trace_type='corrected', response_type='dff', 
                        do_spherical_correction=False, use_lin=True, ds_factor=3, 
                        post_stimulus_sec=0.5, scaley=None,
                        make_pretty_plots=False, nrois_plot=10,
                        plot_response_type='dff', plot_format='svg',
                        ellipse_ec='w', ellipse_fc='none', ellipse_lw=2, 
                        plot_ellipse=True, scale_sigma=True, sigma_scale=2.35,
                        linecolor='darkslateblue', cmap='bone', legend_lw=2, 
                        fit_thr=0.5, rootdir='/n/coxfs01/2p-data', 
                        n_processes=1, test_subset=False, return_trialdata=False):

    #datakey = '%s_%s_fov%i' % (session, animalid, int(fov.split('_')[0][3:]))
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    fov = 'FOV%i_zoom2p0x' % fovn

    fit_results=None; fit_params=None; trialdata=None;
    rows = 'ypos'; cols = 'xpos';
    # Set output dirs
    # -----------------------------------------------------------------------------
    # rf_param_str = 'fit-2dgaus_%s-no-cutoff' % (response_type) 
    if int(session)<20190511:
        run = 'gratings'
    run_name = run.split('_')[1] if 'combined' in run else run
    rfdir, fit_desc = create_rf_dir(datakey, 
                                'combined_%s_static' % run_name, traceid=traceid,
                                response_type=response_type, 
                                do_spherical_correction=do_spherical_correction, 
                                fit_thr=fit_thr)
    # Get data source
    traceid_dir = rfdir.split('/receptive_fields/')[0]
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'np_subtracted.npz')
    data_id = '|'.join([datakey, run, traceid, fit_desc])
    if not os.path.exists(data_fpath):
        # Realign traces
        print("*****corrected offset unfound, NEED TO RUN:*****")
        print("%s | %s | %s | %s | %s" % (datakey, run, traceid))
        # aggregate_experiment_runs(animalid, session, fov, run_name, traceid=traceid)
        # print("*****corrected offsets!*****")
        return None
 
    # Create results outfile, or load existing:
    if create_new is False:
        try:
            print("... checking for existing fit results")
            fit_results, fit_params = load_fit_results(datakey, 
                                        experiment=run_name, traceid=traceid,
                                        response_type=response_type, 
                                        do_spherical_correction=do_spherical_correction)
            print("... loaded RF fit results")
            assert fit_results is not None and fit_params is not None, "EMPTY fit_results"
        except Exception as e:
            traceback.print_exc()
            create_new = True
    print("... do fits?", create_new) 
    rfmaps_arr=None
    if create_new: #do_fits:
        if os.path.exists(rfdir):
            edir = rfdir
            tmp_od, tmp_rt = os.path.split(edir)
            old_dir = os.path.join(tmp_od, '_%s' % tmp_rt)
            # Delete old OLD dir
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
            # Convert existingd dir to the "new" OLD dir
            os.rename(edir, old_dir)    
            print('renamed: %s' % old_dir)
        # Make new dir
        if not os.path.exists(rfdir):
            os.makedirs(rfdir)

        # Get screen dims and fit params
        fit_params = get_fit_params(datakey, run=run, traceid=traceid, 
                                    response_type=response_type, 
                                    do_spherical_correction=do_spherical_correction, 
                                    ds_factor=ds_factor, use_lin=use_lin,
                                    fit_thr=fit_thr,
                                    post_stimulus_sec=post_stimulus_sec, 
                                    sigma_scale=sigma_scale, scale_sigma=scale_sigma,
                                    trace_type=trace_type)
        # -------------------------------------------------------
        if create_new is False:
            rfmaps_arr = load_rfmap_array(fit_params['rfdir'], 
                                do_spherical_correction=do_spherical_correction)
        if rfmaps_arr is None:
            #print("Error loading array, extracting now")
            print("...getting avg by cond")
            # Get trialdata
            trialdata, labels = load_trialdata(fit_params)
            rfmaps_arr = group_trial_values_by_cond(trialdata)
            if do_spherical_correction:
                print("...doin spherical warps")
                if n_processes>1:
                    rfmaps_arr = sphr_correct_maps_mp(rfmaps_arr, fit_params, 
                                    use_lin=use_lin,
                                    n_processes=n_processes, test_subset=test_subset)
                else:
                    rfmaps_arr = sphr_correct_maps(rfmaps_arr, fit_params, 
                                    use_lin=use_lin,
                                    multiproc=False)
            print("...saved array")
            save_rfmap_array(rfmaps_arr, fit_params['rfdir'])
         
        # Do fits 
        print("...now, fitting")
        fit_results, fit_params = fit_rfs(rfmaps_arr, fit_params,
                                    data_identifier=data_id)             
    #
    fitdf = rfits_to_df(fit_results, fit_params=fit_params, scale_sigma=False, 
                        convert_coords=False )
 
    fit_roi_list = fitdf[fitdf['r2']>fit_params['fit_thr']]\
                .sort_values('r2', axis=0, ascending=False).index.tolist()
    print("... %i out of %i fit rois with r2 > %.2f" % 
                (len(fit_roi_list), fitdf.shape[0], fit_params['fit_thr']))

    try:

        fig = plot_top_rfs(fitdf, fit_params, fit_roi_list=fit_roi_list)
        pplot.label_figure(fig, data_id)
        figname = 'top%i_fit_thr_%.2f_%s_ellipse__p3' \
                % (len(fit_roi_list), fit_thr, fit_desc)
        pl.savefig(os.path.join(fit_params['rfdir'], '%s.png' % figname))
        print(figname)
        pl.close()
    except Exception as e:
        traceback.print_exc()
        print("Error plotting best RFs grid")

    try:
        fitdf = rfits_to_df(fit_results, fit_params=fit_params,
                        scale_sigma=False, convert_coords=True)
 
        #dk = '%s_%s_fov%i' % (session, animalid, int(fov.split('_')[0][3:]))
        sdf = aggr.get_stimuli(datakey, run_name)
        screen = hutils.get_screen_dims()
        print(fit_params['sigma_scale'])
        fig = plot_rfs_to_screen(fitdf, sdf, screen, \
                        sigma_scale=fit_params['sigma_scale'],
                        fit_roi_list=fit_roi_list)
        pplot.label_figure(fig, data_id)

        figname = 'overlaid_RFs_pretty__p3' 
        pl.savefig(os.path.join(rfdir, '%s.svg' % figname))
        print(figname)
        pl.close()
    except Exception as e:
        traceback.print_exc()
        print("Error printing RFs to screen, pretty")

    if return_trialdata:
        return fit_results, fit_params, trialdata
    else: 
        return fit_results, fit_params


def plot_rfs_to_screen(fitdf, sdf, screen,sigma_scale=2.35, fit_roi_list=None):
    '''
    plot overlay of cell RFs
    '''
    if fit_roi_list is None:
        fit_roi_list = fitdf.index.tolist()

    fig, ax = pl.subplots( figsize=(10, 5.7))
    #fig.patch.set_visible(False) #(False) #('off')

    ax = plot_rfs_to_screen_pretty(fitdf, sdf, screen, 
                               sigma_scale=sigma_scale,
                               fit_roi_list=fit_roi_list, ax=ax, 
                               roi_colors=['w']*len(fit_roi_list), ellipse_lw=0.2)

    ax.patch.set_color([0.7]*3)
    ax.patch.set_alpha(1)
    ax.set_aspect('equal')

    return fig
 


def plot_top_rfs(fitdf, fit_params,fit_roi_list=None): 
    # Convert to dataframe
    if fit_roi_list is None:
        fit_roi_list = fitdf.index.tolist()
    rfmaps_arr = load_rfmap_array(fit_params['rfdir'], 
                                do_spherical_correction=fit_params['do_spherical_correction'])

    # Plot all RF maps for fit cells (limit = 60 to plot)
    fig = plot_best_rfs(fit_roi_list, rfmaps_arr, fitdf, fit_params,
                        single_colorbar=True, plot_ellipse=True, nr=6, nc=10)
    return fig


from mpl_toolkits.axes_grid1 import AxesGrid

def plot_best_rfs(fit_roi_list, rfmaps_arr, fitdf, fit_params,
                    plot_ellipse=True, single_colorbar=True, nr=6, nc=10):
    #plot_ellipse = True
    #single_colorbar = True
    
    row_vals = fit_params['row_vals']
    col_vals = fit_params['col_vals']
    response_type = fit_params['response_type']
    sigma_scale = fit_params['sigma_scale'] if fit_params['scale_sigma'] else 1.0
     
    cbar_pad = 0.05 if not single_colorbar else 0.5 
    cmap = 'magma' if plot_ellipse else 'inferno' # inferno
    cbar_mode = 'single' if single_colorbar else  'each'
 
    vmin = round(max([rfmaps_arr.min().min(), 0]), 1)
    vmax = round(min([.5, rfmaps_arr.max().max()]), 1)
   
    nx = len(col_vals)
    ny = len(row_vals)
 
    fig = pl.figure(figsize=(nc*2,nr+2))
    grid = AxesGrid(fig, 111,
                nrows_ncols=(nr, nc),
                axes_pad=0.5,
                cbar_mode=cbar_mode,
                cbar_location='right',
                cbar_pad=cbar_pad, cbar_size="3%") 
    for aix, rid in enumerate(fit_roi_list[0:nr*nc]):
        ax = grid.axes_all[aix]
        ax.clear()
        coordmap = rfmaps_arr[rid].values.reshape(ny, nx) 
        
        im = ax.imshow(coordmap, cmap=cmap, vmin=vmin, vmax=vmax) #, vmin=vmin, vmax=vmax)
        ax.set_title('roi %i (r2=%.2f)' % (int(rid+1), fitdf['r2'][rid]), fontsize=8) 
        if plot_ellipse:    
            ell = Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]), 
                          abs(fitdf['sigma_x'][rid])*sigma_scale, abs(fitdf['sigma_y'][rid])*sigma_scale, 
                          angle=np.rad2deg(fitdf['theta'][rid]))
            ell.set_alpha(0.5)
            ell.set_edgecolor('w')
            ell.set_facecolor('none')
            ax.add_patch(ell)            
        if not single_colorbar:
            cbar = ax.cax.colorbar(im)
            cbar = grid.cbar_axes[aix].colorbar(im)
            cbar_yticks = [vmin, vmax] #[coordmap.min(), coordmap.max()]
            cbar.cbar_axis.axes.set_yticks(cbar_yticks)
            cbar.cbar_axis.axes.set_yticklabels([ cy for cy in cbar_yticks], fontsize=8) 
        ax.set_ylim([0, len(row_vals)]) # This inverts y-axis so values go from positive to negative
        ax.set_xlim([0, len(col_vals)])
        #ax.invert_yaxis() 

    if single_colorbar:
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_title(response_type) 
    #%
    for a in np.arange(0, nr*nc):
        grid.axes_all[a].set_axis_off()  
    if not single_colorbar and len(fit_roi_list) < (nr*nc):
        for nix in np.arange(len(fit_roi_list), nr*nc):
            grid.cbar_axes[nix].remove()    
    pl.subplots_adjust(left=0.05, right=0.95, wspace=0.3, hspace=0.3)
    
    return fig


# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------
def load_trialdata(fit_params, return_traces=False):
    rfdir = fit_params['rfdir']
 
    traceid_dir = rfdir.split('/receptive_fields/')[0]
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'np_subtracted.npz')
    if not os.path.exists(data_fpath):
        # Realign traces
        print("*****corrected offset unfound, NEED TO RUN:*****")
        print("%s | %s | %s | %s | %s" % (datakey, run, traceid))
        # aggregate_experiment_runs(animalid, session, fov, run_name, traceid=traceid)
        # print("*****corrected offsets!*****")
        return None

    # Load processed traces 
    raw_traces, labels, sdf, run_info = aggr.load_dataset(data_fpath, 
                                        trace_type=fit_params['trace_type'],
                                        add_offset=True, make_equal=False)
    # Z-score or dff the traces:
    processed_traces, trialdata = aggr.process_traces(raw_traces, labels, 
                                response_type=fit_params['response_type'],
                                nframes_post_onset=fit_params['nframes_post_onset'])       
    if 'trial' not in trialdata.columns:
        trialdata['trial'] = trialdata.index.tolist() 

    if return_traces:
        return trialdata, labels, processed_traces
    else:
        return trialdata, labels

def do_evaluation(datakey, fit_results, fit_params, trialdata,
                n_bootstrap_iters=1000, n_resamples=None, ci=0.95,
                pass_criterion='all', model='ridge', 
                plot_boot_distns=True, deviant_color='dodgerblue', plot_all_cis=False,
                n_processes=1,
                create_new=False, rootdir='/n/coxfs01/2p-data'):
       
    # Set directories
    rfdir = fit_params['rfdir'] 
    fit_desc = fit_params['fit_desc'] 
    evaldir = os.path.join(rfdir, 'evaluation')
    if not os.path.exists(evaldir):
        os.makedirs(evaldir)
    if n_resamples is None:
        n_resamples = int(trialdata.groupby(['config']).count().min().unique())

    #%% Do bootstrap analysis    
    print("-evaluating (%s)-" % str(create_new))
    if not create_new:
        try:
            print("... loading eval results")
            eval_results, eval_params = load_eval_results(datakey, 
                                                     rfdir=rfdir,
                                                     fit_desc=fit_desc) 
            assert 'data' in eval_results.keys(), \
                    "... old datafile, redoing boot analysis"
            assert 'pass_cis' in eval_results.keys(), \
                    "... no criteria passed, redoing"
            print("N eval:", len(eval_results['pass_cis'].index.tolist()))
        except Exception as e:
            # traceback.print_exc()
            create_new=True


    if create_new: 
        if trialdata is None:
            trialdata, labels = load_trialdata(fit_params)

        roi_list = sorted(list(fit_results.keys()))
        roidf_list = [trialdata[[roi, 'config', 'trial']] for roi in roi_list]
        # Update params to include evaluation info 
        evaluation = {'n_bootstrap_iters': n_bootstrap_iters, 
                      'n_resamples': n_resamples,
                      'ci': ci}   
        fit_params.update({'evaluation': evaluation})
        # Do evaluation 
        print("... doing rf evaluation")
        eval_results = evaluate_rfs(roidf_list, fit_params, 
                                    n_processes=n_processes) 
        save_eval_results(eval_results, fit_params)

    if eval_results is None: 
        return None


    ##------------------------------------------------
    fitdf = rfits_to_df(fit_results, fit_params=fit_params, 
                            scale_sigma=fit_params['scale_sigma'], 
                            sigma_scale=fit_params['sigma_scale'])
    fitdf = fitdf[fitdf['r2']>fit_params['fit_thr']]
    fit_rois = fitdf.index.tolist()
    data_id = '|'.join([datakey, fit_desc])

    # Identify cells w fit params within CIs
    pass_cis = check_reliable_fits(fitdf, eval_results['cis']) 
    # Update eval results
    eval_results.update({'pass_cis': pass_cis})
    # Identify reliable fits (params fall within CIs)
    reliable_rois = get_reliable_fits(eval_results['pass_cis'], 
                                      pass_criterion=pass_criterion)
    eval_results.update({'reliable_rois': reliable_rois})
    # Save
    save_eval_results(eval_results, fit_params)
    print("%i out of %i cells w. R2>0.5 are reliable (95%% CI)" 
            % (len(reliable_rois), len(fit_rois)))
    # Plotting
    roidir = os.path.join(evaldir, 
                'rois_%i-iters_%i-resample' % (n_bootstrap_iters, n_resamples))
    if not os.path.exists(roidir):
        os.makedirs(roidir) 
    if os.path.exists(os.path.join(evaldir, 'rois')):
        shutil.rmtree(os.path.join(evaldir, 'rois'))
    # Plot distribution of params w/ 95% CI
    if plot_boot_distns:
        print("... plotting boot distn") #.\n(to: %s" % outdir)
        plot_eval_summary(fitdf, fit_results, eval_results, 
                          sigma_scale=fit_params['sigma_scale'], 
                          scale_sigma=fit_params['scale_sigma'],
                          outdir=roidir, plot_format='svg', 
                          data_id=data_id)
    
#    # Figure out if there are any deviants
     # TODO:  DO THIS< but with projected fov coords
     # Compare CIs of each cell to the regression model CIs.

#    session, animalid, fovn = hutils.split_datakey_str(datakey)
#    traceid = fit_params['rfdir'].split('/traces/')[1].split('_')[0] 
#    fovcoords = roiutils.load_roi_coords(animalid, session, 'FOV%i_zoom2p0x' % fovn,
#                                      traceid=traceid, create_new=False)
#    posdf = pd.concat([fitdf,
#                   fovcoords['roi_positions'].loc[fitdf.index]], axis=1)
#    posdf = posdf.rename(columns={'x0': 'xpos_rf', 'y0': 'ypos_rf',
#                                  'ml_pos': 'xpos_fov', 'ap_pos': 'ypos_fov'})
#
#    marker_size=30; fill_marker=True; marker='o';
#    reg_results = regr_rf_fov(posdf, eval_results, fit_params, 
#                                     data_id=data_id, 
#                                     pass_criterion=pass_criterion, model=model,
#                                     marker=marker, marker_size=marker_size, 
#                                     fill_marker=fill_marker, 
#                                     deviant_color=deviant_color)
    
    return eval_results

from functools import partial

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_


def pool_evaluation(rdf_list, params, n_processes=1):   
    '''
    Wrapper to do bootstrap analysis with Pool
    '''
    #try:
    results = []# None
    terminating = mp.Event()
        
    pool = mp.Pool(initializer=initializer, initargs=(terminating, ), 
                    processes=n_processes)
    try:
        results = pool.map_async(partial(bootstrap_rf_params, fit_params=params), 
                                        rdf_list).get(99999999)
        #pool.close()
    except KeyboardInterrupt:
        print("**interupt")
        pool.terminate()
        print("***Terminating!")
    finally:
        pool.close()
        pool.join()
       
    return results


def evaluate_rfs(roidf_list, fit_params, n_processes=1):
    '''
    Evaluate receptive field fits for cells with R2 > fit_thr.

    Returns:
        eval_results = {'data': bootdata, 'params': bootparams, 'cis': bootcis}
        
        bootdata : dataframe containing results of param fits for bootstrap iterations
        bootparams: params used to do bootstrapping
        cis: confidence intervals of fit params

        If no fits, returns {}

    '''
    # Create output dir for bootstrap results

    # Get params        
    eval_results = {}
    scale_sigma = fit_params['scale_sigma']
    sigma_scale = fit_params['sigma_scale'] if scale_sigma else 1.0
    response_type = fit_params['response_type']
    do_spherical_correction = fit_params['do_spherical_correction']
        
    #print("... doing bootstrap analysis for param fits.")
    start_t = time.time()
    bootdf_list = pool_evaluation(roidf_list, fit_params, 
                                                n_processes=n_processes)
    end_t = time.time() - start_t
    print("Multiple processes: {0:.2f}sec".format(end_t))
    print("--- %i results" % len(bootdf_list))

    if len(bootdf_list)==0:
        return eval_results #None

    # Create dataframe of bootstrapped data
    bootdf = pd.concat(bootdf_list)    
    bootdf = convert_fit_to_coords(bootdf, fit_params)
 
#    if do_spherical_correction is False: 
#        bootdata = convert_fit_to_coorsd(bootdata, fit_params)
#        xx, yy, sigx, sigy = convert_fit_to_coords(bootdata, fit_params)
#        bootdata['x0'] = xx
#        bootdata['y0'] = yy
#        bootdata['sigma_x'] = sigx
#        bootdata['sigma_y'] = sigy

    bootdf['sigma_x'] = bootdf['sigma_x'] * sigma_scale
    bootdf['sigma_y'] = bootdf['sigma_y'] * sigma_scale
    theta_vs = bootdf['theta'].values.copy()
    bootdf['theta'] = theta_vs % (2*np.pi)

    # Calculate confidence intervals
    bootdf = bootdf.dropna()
    bootcis = get_cis_for_params(bootdf, ci=fit_params['evaluation']['ci'])

    # Plot bootstrapped distn of x0 and y0 parameters for each roi (w/ CIs)
    counts = bootdf.groupby(['cell']).count()['x0']
    n_bootstrap_iters = fit_params['evaluation']['n_bootstrap_iters']
    unreliable = counts[counts < n_bootstrap_iters*0.5].index.tolist()
    print("%i cells seem to have <50%% iters with fits" % len(unreliable))
    
    eval_results = {'bootdf': bootdf, 
                    'params': fit_params, 
                    'cis': bootcis, 
                    'unreliable': unreliable}
    # Save
    save_eval_results(eval_results, fit_params)

    #%% Identify reliable fits 
  
    return eval_results

def save_eval_results(eval_results, fit_params):
    rfdir = fit_params['rfdir']
    evaldir = os.path.join(rfdir, 'evaluation')
    rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    with open(rf_eval_fpath, 'wb') as f:
        pkl.dump(eval_results, f, protocol=2)

    rf_params_fpath = os.path.join(rfdir, 'fit_param.json')
    with open(rf_params_fpath, 'w') as f:
        json.dump(fit_params, f, indent=4, sort_keys=True)

    return
 

#def group_configs(group, response_type):
#    '''
#    Takes each trial's reponse for specified config, and puts into dataframe
#    '''
#    config = group['config'].unique()[0]
#    group.index = np.arange(0, group.shape[0])
#
#    return pd.DataFrame(data={'%s' % config: group[response_type]})
 

def bootstrap_rf_params(roi_df, fit_params={},
                        sigma_scale=2.35 ):     

    param_order =['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset', 'r2']

    do_spherical_correction=fit_params['do_spherical_correction']
    row_vals = fit_params['row_vals']
    col_vals = fit_params['col_vals']
    n_resamples = fit_params['evaluation']['n_resamples']
    n_bootstrap_iters = fit_params['evaluation']['n_bootstrap_iters']
    
    xres=1 if do_spherical_correction else float(np.unique(np.diff(row_vals)))
    yres=1 if do_spherical_correction else float(np.unique(np.diff(col_vals)))
    sigma_scale=1 if do_spherical_correction else sigma_scale
    sigma_scale=1 if do_spherical_correction else sigma_scale
    min_sigma=2.5; max_sigma=50;

    paramsdf = None
    try:
        if not terminating.is_set():
            time.sleep(1)
            
        #if do_spherical_correction:
        #    grid_points, cart_vals, sphr_vals = coordinates_for_transformation(fit_params)

        # Get all trials for each config (indicese_= trial reps, columns = conditions)
        roi = int(np.unique([r for r in roi_df.columns if r not in ['config', 'trial']])) 
        responses_df = pd.concat([pd.Series(g[roi], name=c)\
                        .reset_index(drop=True)\
                        for c, g in roi_df.groupby(['config'])], axis=1).dropna(axis=0)        
        # Bootstrap distN of responses (rand w replacement):
#        grouplist = [group_configs(group, response_type) \
#                        for config, group in rdf.groupby(['config'])]
#        responses_df = pd.concat(grouplist, axis=1) 

        # Get mean response across re-sampled trials for each condition 
        # (i.e., each position). Do this n-bootstrap-iters times
        # cols = boot iters, rows = configs
        bootresp_ = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) 
                                for ni in range(n_bootstrap_iters)], axis=1)

        # Reshape array so that it matches for fitrf changs 
        # (should match .reshape(ny, nx))
        nx=len(col_vals)
        ny=len(row_vals)
        if do_spherical_correction:
            lin_x, lin_y = get_lin_coords(resolution=resolution_ds, cm_to_deg=True)
            cart_x, cart_y, sphr_th, sphr_ph = get_spherical_coords(
            cart_pointsX=lin_x, cart_pointsY=lin_y, cm_to_degrees=False)
            bootresp_tmp = bootresp_.copy()
            bootresp_resp = bootresp_tmp.apply(warp_spherical_fromarr, 
                        cart_x=cart_x, cart_y=cart_y,   
                        sphr_th=sphr_th, sphr_ph=sphr_ph, resolution=resolution_ds,
                        row_vals=row_vals, col_vals=col_vals,normalize_range=True)


        bootresp = bootresp_.apply(reshape_array_for_nynx, args=(nx, ny))

        # Fit receptive field for each set of bootstrapped samples 
        bparams = []; #x0=[]; y0=[];
        for ii in bootresp.columns:
            response_vector = bootresp[ii].values
            # nx=len(col_vals), ny=len(row_vals)
            rfmap = get_rf_map(response_vector, nx, ny) 
            fitr, fit_y = do_2d_fit(rfmap, nx=nx, ny=ny) 
            if fitr['success']:
                amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
#                if do_spherical_correction:
#                    # Correct for spher correction, if nec
#                    x0_f, y0_f, sigx_f, sigy_f = get_scaled_sigmas(
#                                                        grid_points, sphr_vals,
#                                                        x0_f, y0_f,
#                                                        sigx_f, sigy_f, theta_f,
#                                                        convert=True)
#                    fitr['popt'] = (amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f) 
                if any(s<min_sigma for s \
                        in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale])\
                        or any(s > max_sigma for s \
                        in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale]):
                    fitr['success'] = False
                
            # If the fit for current bootstrap sample is good, 
            # add it to dataframe of bootstrapped rf params
            if fitr['success']:
                #amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
                curr_fit_results = list(fitr['popt'])
                curr_fit_results.append(fitr['r2'])
                bparams.append(tuple(curr_fit_results)) #(fitr['popt'])
        if len(bparams)==0:
            return None
        paramsdf = pd.DataFrame(data=np.array(bparams), 
                                columns=param_order)
        paramsdf['cell'] = roi    
    except KeyboardInterrupt:
        print("----exiting----")
        terminating.set()
        print("---set terminating---")

    return paramsdf


def get_cis_for_params(bdata, ci=0.95):
    roi_list = [roi for roi, bdf in bdata.groupby(['cell'])]
    param_names = [p for p in bdata.columns if p != 'cell']
    CI = {}
    for p in param_names:
        CI[p] = dict((roi, hutils.get_empirical_ci(bdf[p].values, ci=ci)) \
                        for roi, bdf in bdata.groupby(['cell']))
    
    cis = {}
    for p in param_names:
        cvals = np.array([hutils.get_empirical_ci(bdf[p].values, ci=ci) \
                        for roi, bdf in bdata.groupby(['cell'])])
        cis['%s_lower' % p] = cvals[:, 0]
        cis['%s_upper' % p] = cvals[:, 1]
    cis = pd.DataFrame(cis, index=[roi_list])
    
    return cis

def check_reliable_fits(meas_df, boot_cis): 
    # Test which params lie within 95% CI
    params = [p for p in meas_df.columns.tolist() if p!='r2']
    pass_cis = pd.concat([pd.DataFrame(
            [boot_cis['%s_lower' % p][ri]<=meas_df[p][ri]<=boot_cis['%s_upper' % p][ri] \
            for p in params], columns=[ri], index=params) \
            for ri in meas_df.index.tolist()], axis=1).T
       
    return pass_cis

# Plotting - EVAL
def plot_eval_summary(fitdf, fit_results, eval_results,
                        sigma_scale=2.35, scale_sigma=True, 
                        outdir='/tmp/rf_fit_evaluation', plot_format='svg',
                        data_id='DATA ID'):
    '''
    For all fit ROIs, plot summary of results (fit + evaluation).
    Expect that meas_df has R2>fit_thr, since those are the ones that get bootstrap evaluation 
    '''
    bootdfs = eval_results['bootdf']
    roi_list = fitdf.index.tolist() #sorted(bootdata['cell'].unique())
     
    for ri, rid in enumerate(sorted(roi_list)):
        if ri % 20 == 0:
            print("... plotting eval summary (%i of %i)" % (int(ri+1), len(roi_list))) 
        rfmap = fit_results[rid]['data']
        bootdf_roi = bootdfs[bootdfs['cell']==rid]
        fig = plot_roi_evaluation(rid, fitdf.loc[rid], rfmap, bootdf_roi, 
                                  scale_sigma=scale_sigma, sigma_scale=sigma_scale)
        fig.suptitle('rid %i**' % rid)
        pplot.label_figure(fig, data_id)
        pl.savefig(os.path.join(outdir, 'roi%05d.%s' % (int(rid+1), plot_format)))
        pl.close()
    return

def plot_bootstrapped_distribution(boot_values, true_x, ci=0.95, ax=None, param_name=''):
    lower_x0, upper_x0 = hutils.get_empirical_ci(boot_values, ci=ci)

    if ax is None:
        fig, ax = pl.subplots()
    ax.hist(boot_values, color='k', alpha=0.5)
    ax.axvline(x=lower_x0, color='k', linestyle=':')
    ax.axvline(x=upper_x0, color='k', linestyle=':')
    ax.axvline(x=true_x, color='r', linestyle='-')
    ax.set_title('%s (n=%i)' % (param_name, len(boot_values)))
   
    return ax

def plot_roi_evaluation(rid, fitdf_roi, rfmap, bootdf_roi, ci=0.95, 
                             scale_sigma=True, sigma_scale=2.35):
    
    fig, axn = pl.subplots(2,3, figsize=(10,6))
    ax = axn.flat[0]
    ax = plot_rf_map(rfmap, cmap='inferno', ax=ax)
    ax = plot_rf_ellipse(fitdf_roi, ax=ax, scale_sigma=scale_sigma)
    params = ['sigma_x', 'sigma_y', 'theta', 'x0', 'y0']
    ai=0
    for param in params:
        ai += 1
        try:
            ax = axn.flat[ai]
            ax = plot_bootstrapped_distribution(bootdf_roi[param].values, 
                                                fitdf_roi[param], 
                                                ci=ci, ax=ax, param_name=param)
            pl.subplots_adjust(wspace=0.7, hspace=0.5, top=0.8)
            fig.suptitle('rid %i' % rid)
        except Exception as e:
            print("!! eval error (plot_boot_distn): rid %i, param %s" \
                % (rid, param))
            traceback.print_exc()
            
    return fig


# Check scatter
# ----------------------------------------------------------------------     
#%% FITTING FUNCTIONS
# ----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import scipy.stats as spstats
import sklearn.metrics as skmetrics #import mean_squared_error

def regr_rf_fov(posdf, fit_params, eval_results, 
                model='ridge', pass_criterion='all', data_id='ID', 
                deviant_color='magenta', marker='o', 
                marker_size=20, fill_marker=True):
    print("~regressing rf on fov~")
    reliable_rois = eval_results['reliable_rois']

    evaldir = os.path.join(fit_params['rfdir'], 'evaluation')
    fig = plot_linear_regr_by_condition( posdf.loc[reliable_rois],model=model)
    pl.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)
    pplot.label_figure(fig, data_id)
    pl.savefig(os.path.join(evaldir, 'RFpos_v_CTXpos_split_axes.svg'))   
    pl.close()
   
    #%% Compare regression fit to bootstrapped params 
    reg_results = compare_regr_to_boot_params(eval_results, posdf, 
                                        outdir=evaldir, data_id=data_id, 
                                        pass_criterion=pass_criterion, model=model, 
                                        deviant_color=deviant_color, marker=marker,
                                        marker_size=marker_size, 
                                        fill_marker=fill_marker)

    #%% Identify "deviants" based on spatial coordinates
    print('%i reliable of %i fit (thr>.5) | regr R2=%.2f' \
                % (len(reg_results['reliable_rois']), 
                   len(meas_df), reg_results['azimuth']['regr_R2']))

    return reg_results #deviants

def plot_linear_regr_by_condition(posdf, model='ridge'):
    
    fig, axes = pl.subplots(2, 3, figsize=(10, 6))
    for ri, cond in enumerate(['azimuth', 'elevation']):
        # Do fit
        axname = 'xpos' if cond=='azimuth' else 'ypos' 
        yv = posdf['%s_rf' % axname].values
        xv = posdf['%s_fov' % axname].values    
        try:
            fitv, regr = fit_linear_regr(xv, yv, return_regr=True, model=model)
        except Exception as e:
            traceback.print_exc()
            print("Error fitting cond %s" % cond)
            continue
        # Evaluate fit
        rmse = np.sqrt(skmetrics.mean_squared_error(yv, fitv))
        r2 = float(skmetrics.r2_score(yv, fitv))
        print("[%s] Mean squared error: %.2f | Variance score: %.2f" % (cond, rmse, r2))

        # Plot 
        ax=axes[ri, 0]
        ax.set_title(cond, fontsize=12, loc='left')
        ax.scatter(xv, yv, c='k', alpha=0.5)
        ax.set_ylabel('RF position (rel. deg.)')
        ax.set_xlabel('FOV position (um)')
        #ax.set_xlim([0, ylim])
        #sns.despine(offset=1, trim=True, ax=ax)
        ax.plot(xv, fitv, 'r')
        ax.set_xlim([0, 1200])
        #ax.set_ylim()    
        r, p = spstats.pearsonr(posdf['%s_fov' % axname], posdf['%s_rf' % axname]) 
        corr_str = 'pearson=%.2f (p=%.2f)' % (r, p)
        ax.plot(ax.get_xlim()[0], ax.get_ylim()[0], alpha=0, label=corr_str)
        ax.legend(loc='upper right', fontsize=8)
    
        ax = axes[ri, 1]
        residuals = fitv - yv
        ax.hist(residuals, histtype='step', color='k')
        #sns.despine(offset=1, trim=True, ax=ax)
        ax.set_xlabel('residuals')
        ax.set_ylabel('counts')
        maxval = max (abs(residuals))
        ax.set_xlim([-maxval, maxval])
         
        ax = axes[ri, 2]
        r2_vals = posdf['r2'].values
        ax.scatter(r2_vals, abs(residuals), c='k', alpha=0.5)
        ax.set_xlabel('r2')
        ax.set_ylabel('abs(residuals)')       
        if model=='ridge':
            regr = Ridge()
        elif model=='Lasso':
            regr = Lasso()
        else:
            model = 'ols'
            regr = LinearRegression()
        # Add some metrics
        regr.fit(r2_vals.reshape(-1, 1), residuals.reshape(-1, 1)) #, yv)
        r2_dist_corr = regr.predict(r2_vals.reshape(-1, 1))
        ax.plot(r2_vals, r2_dist_corr, 'r', label=model)
        #sns.despine(offset=1, trim=True, ax=ax)
        r, p = spstats.pearsonr(r2_vals, np.abs(residuals))
        corr_str2 = 'pearson=%.2f (p=%.2f)' % (r, p)
        ax.plot(ax.get_xlim()[0], ax.get_xlim()[-1], alpha=0, label=corr_str2)
        ax.legend(loc='upper right', fontsize=8)
    
    pl.subplots_adjust(hspace=0.5, wspace=0.5)    
    return fig



 
def plot_linear_regr(xv, yv, ax=None,  model='ridge', 
                     marker='o', marker_size=30, alpha=1.0, marker_color='k',
                     linestyle='_', linecolor='r'):
    try:
        fitv, regr = fit_linear_regr(xv, yv, return_regr=True, model=model)
    except Exception as e:
        traceback.print_exc()
        print("... no lin fit")
        return None

    if ax is none:
        fig, ax = pl.subplots()
        
    rmse = np.sqrt(skmetrics.mean_squared_error(yv, fitv))
    r2 = float(skmetrics.r2_score(yv, fitv))
    print("[%s] Mean squared error: %.2f" % (cond, rmse))
    print('[%s] Variance score: %.2f' % (cond, r2))
    
    ax.scatter(xv, yv, c=marker_color, marker=marker, s=marker_size, alpha=alpha)
    ax.plot(xv, fitv, linestyle, color=linecolor, label=model)
    ax.set_xlim([0, 1200])
    #ax.set_ylim()    
    eq_str = 'y=%.2fx + %.2f' % (regr.coef_[0], regr.intercept_[0])
    ax.set_title(eq_str, loc='left', fontsize=12)
 
    r, p = spstats.pearsonr(xv, yv) #.abs())
    corr_str = 'pearson=%.2f (p=%.2f)' % (r, p)
    ax.plot(ax.get_xlim()[0], ax.get_ylim()[0], alpha=0, label=corr_str)
    ax.legend(loc='upper right', fontsize=8)

    return regr

def fit_linear_regr(xvals, yvals, return_regr=False, model='ridge'):
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
    if any(np.isnan(xvals)) or any(np.isnan(yvals)):
        print("NAN")
        #print(np.where(np.isnan(xvals)))
        #print(np.where(np.isnan(yvals)))
    regr.fit(xvals, yvals)
    fitv = regr.predict(xvals)
    if return_regr:
        return fitv.reshape(-1), regr
    else:
        return fitv.reshape(-1)


def compare_regr_to_boot_params(eval_results, posdf, xlim=None, ylim=None, 
                                pass_criterion='all', model='ridge', 
                                deviant_color='dodgerblue', marker='o',
                                marker_size=20, fill_marker=True,
                                outdir='/tmp', data_id='DATAID',
                                plot_all_cis=False):

    '''
    deviants:  
        Cells w/ good RF fits (boostrapped, measured lies within some CI), but
               even CI lies outside of estimated regression CI
    bad_fits:  
        Cells w/ measured RF locations that do not fall within 
                the CI from bootstrapping
    
    To get all "pass" rois, include all returned ROIs with fits that are NOT in bad_fits.
    '''
    bootdata = eval_results['data']
    bootcis = eval_results['cis']
    fit_rois = [int(k) for k, g in bootdata.groupby(['cell'])]    
    pass_rois = eval_results['pass_cis'].index.tolist()
    pass_cis = eval_results['pass_cis'].copy()
    reliable_rois = get_reliable_fits(eval_results['pass_cis'], 
                                        pass_criterion=pass_criterion)
    
    #% # Plot bootstrapped param CIs + regression CI
    xaxis_lim = max([xlim, ylim])
    reg_results = {}

    for cond in ['azimuth', 'elevation']:
        fig, regr, regci, reliable_c, deviants, bad_fits = do_regr_on_fov_cis(
                                                        bootdata, bootcis,
                                                        posdf, cond=cond,
                                                        model=model, 
                                                        roi_list=[], #reliable_rois,
                                                        deviant_color=deviant_color,
                                                        fill_marker=fill_marker,
                                                        marker=marker, 
                                                        marker_size=marker_size,
                                                        xaxis_lim=xlim) #xaxis_lim)
        # Get some stats from linear regr
        rmse = np.sqrt(skmetrics.mean_squared_error(regr['yv'], regr['fitv']))
        r2 = skmetrics.r2_score(regr['yv'], regr['fitv'])
        pearson_r, pearson_p = spstats.pearsonr(regr['xv'], regr['yv'])


        pass_rois = [i for i in fit_rois if i not in bad_fits]
        reg_results[cond] = {'cis': [tuple(ci) for ci in regci], 
                            'deviants': deviants, 
                            'bad_fits': bad_fits, 
                            'pass_rois': pass_rois,
                            'reliable_rois': reliable_rois,
                            'regr_coef': float(regr['regr'].coef_[0]), #r_coef,
                            'regr_int': float(regr['regr'].intercept_[0]),
                            'regr_R2': r2, 'regr_RMSE': rmse, 
                            'regr_pearson_p': pearson_p, 'regr_pearson_r': pearson_r}
 
        pplot.label_figure(fig, data_id)
        pl.savefig(os.path.join(outdir, 'VF2RF_regr_deviants_%s%s.svg' \
                                                    % (cond, filter_str)))
        pl.close()

    reg_results['reliable_rois'] = reliable_rois
    reg_results['pass_criterion'] = pass_criterion
   
    with open(os.path.join(outdir, 'regr_results_deviants_bycond.json'), 'w') as f:
        json.dump(reg_results, f, indent=4)    
    print("--- saved roi info after evaluation.")
  
    return reg_results
   

