import os

import glob
import json
import traceback

# import pickle as pkl
import statsmodels as sm
import dill as pkl
import numpy as np
import pylab as pl
import seaborn as sns
import scipy.stats as spstats
import pandas as pd
import importlib

import scipy as sp
import itertools
import matplotlib as mpl
mpl.use('agg')

from matplotlib.lines import Line2D
import statsmodels as sm
#import statsmodels.api as sm # put this in Nb

from .stats import do_mannwhitney

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)       

import re
import cv2
from matplotlib.patches import PathPatch


# ###############################################################
# Plotting:
# ###############################################################
def print_means(plotdf, groupby=['visual_area', 'arousal'], params=None, metric='mean'):
    if params is None:
        params = [k for k in plotdf.columns if k not in groupby]
       
    if metric=='mean': 
        m_ = plotdf.groupby(groupby)[params].mean().reset_index()
    else:
        m_ = plotdf.groupby(groupby)[params].median().reset_index()
       
    s_ = plotdf.groupby(groupby)[params].std().reset_index()
    for p in params:
        m_['%s_std' % p] = s_[p].values
    print("[%s]:" % metric)
    print(m_)

def set_threecolor_palette(c1='magenta', c2='orange', c3='dodgerblue', cmap=None, soft=False,
                            visual_areas = ['V1', 'Lm', 'Li']):
    if soft:
        c1='turquoise';c2='cornflowerblue';c3='orchid';

    # colors = ['k', 'royalblue', 'darkorange'] #sns.color_palette(palette='colorblind') #, n_colors=3)
    # area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}
    if cmap is not None:
        c1, c2, c3 = sns.color_palette(palette=cmap, n_colors=len(visual_areas))#'colorblind') #, n_colors=3) 
    area_colors = dict((k, v) for k, v in zip(visual_areas, [c1, c2, c3]))

    return visual_areas, area_colors


def set_plot_params(lw_axes=0.25, labelsize=6, color='k', dpi=100):
    import pylab as pl
    #### Plot params
    pl.rcParams['font.size'] = labelsize
    #pl.rcParams['text.usetex'] = True
    
    pl.rcParams["axes.labelsize"] = labelsize
    pl.rcParams["axes.linewidth"] = lw_axes
    pl.rcParams["xtick.labelsize"] = labelsize
    pl.rcParams["ytick.labelsize"] = labelsize
    pl.rcParams['xtick.major.width'] = lw_axes
    pl.rcParams['xtick.minor.width'] = lw_axes
    pl.rcParams['ytick.major.width'] = lw_axes
    pl.rcParams['ytick.minor.width'] = lw_axes
    pl.rcParams['legend.fontsize'] = labelsize
    
    pl.rcParams['figure.figsize'] = (5, 4)
    pl.rcParams['figure.dpi'] = dpi
    pl.rcParams['savefig.dpi'] = dpi
    pl.rcParams['svg.fonttype'] = 'none' #: path
        
    
    for param in ['xtick.color', 'ytick.color', 'axes.labelcolor', 'axes.edgecolor']:
        pl.rcParams[param] = color

    return 

# Colorbar stuff
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as lsc

def cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using pl.get_cmap(name).
    """
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
#     # Cast the steps into lists:
#     step_dict = {key: map(lambda x: x[0], cdict[key]) for key in cdict}
#     print(step_dict)
#     # Now get the unique steps (first column of the arrays):
#     step_list = np.unique(sum(step_dict.values(), []))
    step_dict = dict((k, list(v)) for k, v in cdict.items())
    step_list = np.unique(sum(step_dict.values(), []))

    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
#     y0 = np.array(map(function, y0))
#     y1 = np.array(map(function, y1))
    y0 = function(y0)
    y1 = function(y1)
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    return lsc(name, step_dict, N=N, gamma=gamma)

def darken(x, ):
   return x * 0.85



 
def adjust_polar_axes(ax, theta_loc='E'):

    ax.set_theta_zero_location(theta_loc)
    #ax.set_theta_direction(1)
    ax.set_xticklabels(['0$^\circ$', '', '90$^\circ$', '', '', '', '-90$^\circ$', ''])
    ax.set_rlabel_position(135) #315)
    ax.set_yticks(np.linspace(0, 0.8, 3))
    ax.set_yticklabels(np.linspace(0, 0.8, 3))
    #ax.set_ylabel(metric, fontsize=12)
    # Grid lines and such
    ax.spines['polar'].set_visible(False)
    

def equal_corr_axes(fg):

    for ax in fg.axes:
        ax.set_aspect('equal')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        #ax.legend_.remove()
        ax.plot([-1, 1], [-1, 1], color='r', ls='-', lw=0.5)


# Adjust plotting of default seaborn 
def change_width(ax, new_value):
    '''Adjust spacing bw grouped bar plots in seaborn (fraction)'''
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def adjust_box_widths(ax, fac):
    # iterating through axes artists:
    for c in ax.get_children():

        # searching for PathPatches
        if isinstance(c, PathPatch):
            # getting current width of box:
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5*(xmin+xmax)
            xhalf = 0.5*(xmax - xmin)

            # setting new width of box
            xmin_new = xmid-fac*xhalf
            xmax_new = xmid+fac*xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

            # setting new width of median line
            for l in ax.lines:
                if np.all(l.get_xdata() == [xmin, xmax]):
                    l.set_xdata([xmin_new, xmax_new])

    return


#def adjust_box_widths(g, fac):
#    """
#    Adjust the withs of a seaborn-generated boxplot.
#    """
#
#    # iterating through Axes instances
#    for ax in g.axes:
#
#        # iterating through axes artists:
#        for c in ax.get_children():
#
#            # searching for PathPatches
#            if isinstance(c, PathPatch):
#                # getting current width of box:
#                p = c.get_path()
#                verts = p.vertices
#                verts_sub = verts[:-1]
#                xmin = np.min(verts_sub[:, 0])
#                xmax = np.max(verts_sub[:, 0])
#                xmid = 0.5*(xmin+xmax)
#                xhalf = 0.5*(xmax - xmin)
#
#                # setting new width of box
#                xmin_new = xmid-fac*xhalf
#                xmax_new = xmid+fac*xhalf
#                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
#                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
#
#                # setting new width of median line
#                for l in ax.lines:
#                    if np.all(l.get_xdata() == [xmin, xmax]):
#                        l.set_xdata([xmin_new, xmax_new])
#
#    return
#
    
def crop_legend_labels(ax, n_hues, start_ix=0,  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
                        title='', ncol=1, markerscale=1):
    # Get the handles and labels.
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    # When creating the legend, only use the first two elements
     
    leg = ax.legend(leg_handles[start_ix:start_ix+n_hues], leg_labels[start_ix:start_ix+n_hues], title=title,
            bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, loc=loc, ncol=ncol, 
            markerscale=markerscale)
    return leg #, leg_handles

def get_empirical_ci(stat, ci=0.95):
    p = ((1.0-ci)/2.0) * 100
    lower = np.percentile(stat, p) #max(0.0, np.percentile(stat, p))
    p = (ci+((1.0-ci)/2.0)) * 100
    upper = np.percentile(stat, p) # min(1.0, np.percentile(x0, p))
    #print('%.1f confidence interval %.2f and %.2f' % (alpha*100, lower, upper))
    return lower, upper

def set_split_xlabels(ax, offset=0.25, a_label='rfs', b_label='rfs10', rotation=0, ha='center', ncols=3):
    locs = []
    labs = []
    for li in np.arange(0, ncols):
        locs.extend([li-offset, li+offset])
        labs.extend([a_label, b_label])
    ax.set_xticks(locs)
    ax.set_xticklabels(labs, rotation=rotation, ha=ha)
    ax.tick_params(axis='x', size=0)
    sns.despine(bottom=True, offset=4)
    return ax

def plot_paired(plotdf, aix=0, curr_metric='avg_size', ax=None,
                c1='rfs', c2='rfs10', compare_var='experiment',
                marker='o', offset=0.25, color='k', label=None, lw=0.5, alpha=1, 
                return_vals=False, return_stats=True, round_to=3, ttest=True):

    import analyze2p.stats as st
    if ax is None:
        fig, ax = pl.subplots()
    pdict, a_vals, b_vals = st.paired_ttest_from_df(plotdf, 
                                metric=curr_metric, c1=c1, c2=c2,
                                compare_var=compare_var, 
                                round_to=round_to, 
                                return_vals=True, ttest=ttest)
    by_exp = [(a, e) for a, e in zip(a_vals, b_vals)]
    for pi, p in enumerate(by_exp):
        ax.plot([aix-offset, aix+offset], p, marker=marker, color=color,
                alpha=alpha, lw=lw,  zorder=0, markerfacecolor=None,
                markeredgecolor=color, label=label)
    if return_vals:
        return ax, a_vals, b_vals

    elif return_stats:
        #tstat, pval = spstats.ttest_rel(a_vals, b_vals)
        #print("(t-stat:%.2f, p=%.2f)" % (tstat, pval))
        return ax, pdict
    else: 
        return ax

def pairwise_compare_single_metric(comdf, curr_metric='avg_size', 
                        c1='rfs', c2='rfs10', compare_var='experiment',
                        c1_label=None, c2_label=None,
                        ax=None, marker='o', visual_areas=['V1', 'Lm', 'Li'],
                        lw=1, alpha=1., 
                        xlabel_offset=-1, area_colors=None, bar_ci=95, bar_lw=0.5,
                        return_stats=False, round_to=3, ttest=True):

    import analyze2p.stats as st
    assert 'datakey' in comdf.columns, "Need a sorter, 'datakey' not found."
    if area_colors is None:
        visual_areas = ['V1', 'Lm', 'Li']
        colors = ['magenta', 'orange', 'dodgerblue'] 
        #sns.color_palette(palette='colorblind') #, n_colors=3)
        area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}
    offset = 0.25 
    if ax is None:
        fig, ax = pl.subplots(figsize=(5,4), dpi=150)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    # Plot paired values
    aix=0
    r_=[]
    for ai, visual_area in enumerate(visual_areas):
        plotdf = comdf[comdf['visual_area']==visual_area]
        # Do ttest or wilcoxon 
        pdict, a_vals, b_vals = st.paired_ttest_from_df(plotdf, 
                                    metric=curr_metric, c1=c1, c2=c2,
                                    compare_var=compare_var, 
                                    round_to=round_to, 
                                    return_vals=True, ttest=ttest)
        # plot
        by_exp = [(a, e) for a, e in zip(a_vals, b_vals)]
        color=area_colors[visual_area]
        for pi, p in enumerate(by_exp):
            ax.plot([aix-offset, aix+offset], p, marker=marker, color=color,
                    alpha=alpha, lw=lw,  zorder=0, markerfacecolor=None,
                    markeredgecolor=color, label=visual_area) #label)
        pdict.update({'visual_area': visual_area})
        res = pd.DataFrame(pdict, index=[ai])
        r_.append(res)
        aix = aix+1
    statdf = pd.concat(r_, axis=0)

    # Plot average
    sns.barplot("visual_area", curr_metric, data=comdf, 
                hue=compare_var, hue_order=[c1, c2], #zorder=0,
                ax=ax, order=visual_areas, ci=bar_ci,
                errcolor="k", edgecolor=('k', 'k', 'k'), 
                facecolor=(1,1,1,0), linewidth=bar_lw)
    ax.legend_.remove()
    for x in ax.get_xticks():
        ax.text(x, xlabel_offset, visual_areas[x])
    c1_label = c1 if c1_label is None else c1_label
    c2_label = c2 if c2_label is None else c2_label
    set_split_xlabels(ax, a_label=c1_label, b_label=c2_label)
    if return_stats:
        return ax, statdf
    else: 
        return ax

def annotate_sig_on_paired_plot(ax, plotd, pstats, metric, lw=1.,
                                stat='p_val', thr=0.05, offset=2, h=2,
                                visual_areas=['V1', 'Lm', 'Li']):
    sig_areas = list(pstats[pstats[stat]<thr]['visual_area'])
    for v in sig_areas:
        vix = visual_areas.index(v)
        xticks = ax.get_xticks()
        x1, x2 = xticks[vix*2], xticks[(vix*2)+1]
        y, h, col = plotd[metric].max() + offset, h, 'k'
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c=col)
        ax.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

    return

# STATS
# Plotting
def label_figure(fig, data_identifier):
    fig.text(0, 1,data_identifier, ha='left', va='top', fontsize=8)

    
def plot_mannwhitney(mdf, metric='I_rs', multi_comp_test='holm',
                        ax=None, y_loc=None, offset=0.1, lw=0.25, fontsize=6):
    if ax is None:
        fig, ax = pl.subplots()

    print("********* [%s] Mann-Whitney U test(mc=%s) **********" % (metric, multi_comp_test))
    statresults = do_mannwhitney(mdf, metric=metric, multi_comp_test=multi_comp_test)
    #print(statresults)

    # stats significance
    ax = annotate_stats_areas(statresults, ax, y_loc=y_loc, offset=offset, 
                                lw=lw, fontsize=fontsize)
    print("****************************")

    return statresults, ax


# Stats
def annotate_stats_areas(statresults, ax, lw=1, color='k',
                        y_loc=None, offset=0.1, fontsize=6,
                         visual_areas=['V1', 'Lm', 'Li']):

    if y_loc is None:
        y_loc = round(ax.get_ylim()[-1], 1)*1.2
        offset = y_loc*offset #0.1

    for ci in statresults[statresults['reject']].index.tolist():
    #np.arange(0, statresults[statresults['reject']].shape[0]):
        v1, v2, pv, uv = statresults.iloc[ci][['d1', 'd2', 'p_val', 'U_val']].values
        x1 = visual_areas.index(v1)
        x2 = visual_areas.index(v2)
        y1 = y_loc+(ci*offset)
        y2 = y1
        ax.plot([x1,x1, x2, x2], [y1, y2, y2, y1], linewidth=lw, color=color)
        ctrx = x1 + (x2-x1)/2.
        star_str = '**' if pv<0.01 else '*'
        ax.text(ctrx, y1+(offset/8.), star_str, fontsize=fontsize)

    return ax



# Drawing funcs
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from matplotlib.patches import Ellipse, Rectangle, Polygon

def get_icon(name, icon_type='ori'):
    src = '/n/coxfs01/julianarhee/legends/%s/%s%03d.png' % (icon_type, icon_type, name)
    #print(src)
    im = pl.imread(src)
    return im

def offset_image(coord, ax, name=None, pad=0, xybox=(0, 0), xycoords=('data'), yloc=None, zoom=1.0):
    img = get_icon(name)
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax
    
    if yloc is None:
        yloc=ax.get_ylim()[-1]
    ab = AnnotationBbox(im, (coord, yloc), xybox=xybox, frameon=False,
                        xycoords=xycoords,
                        boxcoords="offset points", pad=pad, annotation_clip=False)
    ax.add_artist(ab)
    
    return yloc

def replace_ori_labels(ori_names, bin_centers=None, ax=None, xybox=(0, 0), yloc=None,
                       zoom=0.25, pad=0, polar=False):
    if polar:
        xycoords=("data")
        for i, c in enumerate(ori_names):
            yloc = offset_image(np.deg2rad(c), ax, name=c, xybox=xybox, xycoords=xycoords, 
                                yloc=yloc,  pad=pad, zoom=zoom)
            ax.set_xticklabels([])
        #if yloc is not None:
        #ax.set_ylim([0, yloc])
        ax.spines['polar'].set_visible(False)
    else:
        xycoords=("data", "axes fraction")
        if bin_centers is None:
            bin_centers = ori_names
        for i, (binloc, binval) in enumerate(zip(bin_centers, ori_names)):
            yloc = offset_image(binloc, ax, name=binval, xybox=xybox, xycoords=xycoords,
                                yloc=yloc, pad=pad, zoom=zoom)
            ax.set_xticklabels([])
    
    return ax

# Plotting
from matplotlib.offsetbox import (DrawingArea, OffsetImage,AnnotationBbox)

def replace_rf_labels(rf_bins, ax=None, width=20, height=10, yloc=None, polar=True,
                      box_alignment=(0.5, 1),xybox=(0,0), alpha=0.5,
                      fill=False, color='k', lw=1):
    
    for rf_ang in rf_bins:
        # Annotate the 1st position with a circle patch
        da = DrawingArea(height, height, height, height)
        p = Ellipse((0, 0), width=width, height=height, angle=rf_ang, alpha=alpha, color=color,
                                    lw=lw, fill=fill, clip_on=False)
        da.add_artist(p)

        xloc = np.deg2rad(rf_ang) if polar else rf_ang
        
        ab = AnnotationBbox(da, (xloc, yloc),
                            xybox=xybox, 
                            xycoords=("data"),
                            box_alignment=box_alignment, #(.5, 0.5),
                            boxcoords="offset points",
                            bboxprops={"edgecolor" : "none", "facecolor": "none"},
                            annotation_clip=False)

        ax.add_artist(ab)
    
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    return ax

def add_subplot_axes(ax,rect,axisbg='w', axis_alpha=1):
    '''
    https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    '''
    fig = pl.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=axisbg) #axisbg=axisbg)
    subax.patch.set_alpha(axis_alpha)
    
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    
    return subax


def colorbar(mappable, label=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    if label is not None:
        cax.set_title(label)
    return cbar


def turn_off_axis_ticks(ax, despine=True):
    ax.tick_params(which='both', axis='both', size=0)
    if despine:
        sns.despine(ax=ax, left=True, right=True, bottom=True, top=True) #('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def custom_legend_markers(colors=['m', 'c'], labels=['label1', 'label2'], marker='o'):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    leg_elements=[]
    for col, label in zip(colors, labels):
        leg_elements.append(Line2D([0], [0], marker=marker, color=col, label=label))

    return leg_elements

def darken_cmap(colormap='spectral', alpha=0.9, 
        cmapdir = '/n/coxfs01/julianarhee/colormaps'):

    try:
        cmap = mpl.cm.get_cmap(colormap)
    except Exception as e:
        print(e)
        cdata = np.loadtxt(os.path.join(cmapdir, colormap) + ".txt")
        cmap = mpl.colors.LinearSegmentedColormap.from_list(colormap, cdata[::-1])
    # modify colormap
    colors = []
    for ind in range(cmap.N):
        c = []
        for x in cmap(ind)[:3]: c.append(x*alpha)
        colors.append(tuple(c))
    dark_cmap = mpl.colors.ListedColormap(colors, name='dark_%s' % colormap)
    mpl.cm.register_cmap("dark_%s" % colormap, dark_cmap)
    new_cmap_name='dark_%s' % colormap
    return new_cmap_name


# Axis labeling
# https://stackoverflow.com/questions/58854335/how-to-label-y-ticklabels-as-group-category-in-seaborn-clustermap
from itertools import groupby    
def add_line(ax, xpos, ypos, offset=0.2, lw=1):
    line = pl.Line2D([ypos, ypos+ offset], [xpos, xpos], color='black', 
                     transform=ax.transAxes, linewidth=lw)
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in groupby(labels)]

def label_group_bar_table(ax, df, offset=0.2, lw=1):
    xpos = -offset
    scale = 1./df.index.size
    for level in range(df.index.nlevels):
        pos = df.index.size
        for label, rpos in label_len(df.index,level):
            add_line(ax, pos*scale, xpos, offset=offset, lw=lw)
            pos -= rpos
            lypos = (pos + .5 * rpos)*scale
            ax.text(xpos+(offset*0.5), lypos, label, ha='center', transform=ax.transAxes) 
        add_line(ax, pos*scale , xpos)
        xpos -= offset


# Visualizing fov
def adjust_image_contrast(img, clip_limit=2.0, tile_size=10):#(10,10)):
    img[img<-50] = 0
    normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 8-bit
    img8 = cv2.convertScaleAbs(normed)

    # Equalize hist:
    tg = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tg)
    eq = clahe.apply(img8)

    return eq

