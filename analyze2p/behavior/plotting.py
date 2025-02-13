#!/usr/bin/env python2

import os
import glob
import json
#import pymworks
import re
#import datautils
import copy
import math
import time
import optparse
import sys
import pprint

import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
#import cPickle as pkl
import scipy.stats as spstats

import utils as util
import assign_phase as ph
import process_datafiles as procd

pp = pprint.PrettyPrinter(indent=4)

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------------

# Plotting:
# -----------------------------------------------------------------------------
def label_figure(fig, data_identifier):
    fig.text(0, 1,data_identifier, ha='left', va='top', fontsize=8)

def set_plot_params(lw_axes=0.25, labelsize=6, color='k', dpi=100):
    import pylab as pl
    #### Plot params
    #pl.rcParams['font.size'] = 6
    #pl.rcParams['text.usetex'] = True
   
    pl.rcParams["font.family"] = "Arial"
 
    pl.rcParams["axes.labelsize"] = labelsize + 1
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


def get_fig_id(animalids, cohort_list, phase_list):
    cohort_str = []
    for cohort in cohort_list:
        anums = [int(re.search(r'(\d+)', a).group()) for a in animalids \
                 if re.search(r'(\D+)', a).group()==cohort]
        if len(anums)>0:
            cohort_str.append('%s%i-%i' % (cohort, min(anums), max(anums)))
        else:
            print("Skipping <%s>, none found" % cohort)
    figid = 'phase%s_cohorts_%s\n%s' % ('-'.join([str(p) for p in phase_list]), '-'.join(cohort_list), ' | '.join(cohort_str))

    return figid


def update_fonts(labelsize=24): #big=True):

    # Font params
#     font_params = {'legend.fontsize': 20, #'medium',
#              'axes.labelsize': 24, #'x-large',
#              'axes.titlesize': 24, #'x-large',
#              'xtick.labelsize': 20, #'x-large',
#              'ytick.labelsize': 20} #'x-large'}

    font_params = {'legend.fontsize': 12, #'medium',
             'axes.labelsize': labelsize, #'x-large',
             'axes.titlesize': labelsize, #'x-large',
             'xtick.labelsize': labelsize-4, #'x-large',
             'ytick.labelsize': labelsize-4} #'x-large'}

    # font_params = {'legend.fontsize': 'medium',
    #          'axes.labelsize': 'medium',
    #          'axes.titlesize': 'medium',
    #          'xtick.labelsize': 'medium',
    #          'ytick.labelsize': 'medium'}
    pl.rcParams.update(font_params)


def get_pnas_cmap():
    # combine two color maps for 0-50% and 50-100% as in the PNAS paper
    colors1 = pl.cm.bone(np.linspace(0.95,0.0,128))
    colors2 = pl.cm.hot(np.linspace(0.,0.95,128))
    colors = np.vstack((colors1, colors2))
    pnas_cmap = mcolors.LinearSegmentedColormap.from_list('pnas_map', colors)
    
    return pnas_cmap

def outline_boxplot(ax):
    # iterate over boxes to make them outlines only
    for i,box in enumerate(ax.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')
        # iterate over whiskers and median lines
        for j in range(6*i,6*(i+1)):
             ax.lines[j].set_color('black')
    return


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def legend_from_dict(legend_dict, marker='o', markersize=5):

    leg_handles = [Line2D([0], [0], marker='o', color=c, label=l,
                        markerfacecolor=c, markersize=markersize) for l, c in legend_dict.items()]
    return leg_handles

def draw_no_feedback(ax, curr_no_fb, defaults, seaborn=True, lw=2):

    default_size = defaults['size']
    default_depth_rotation = defaults['depth_rotation']
    default_planar_rotation = defaults['planar_rotation']

    expected_sizes = defaults['expected_sizes']
    expected_drots = defaults['expected_depth_rotations']

    offset = 0 if seaborn else 0.5

    # Draw default box
    default_size_ix = list(expected_sizes).index(default_size)
    default_drot_ix = list(expected_drots).index(default_depth_rotation)
    ax.add_patch(Rectangle((default_drot_ix-offset, default_size_ix-offset), 1, 1, 
                           fill=False, edgecolor='forestgreen', lw=lw, label='default'))
    
    if len(curr_no_fb) > 0:
        # Draw no feedback box
        nofb_size_min = min([f[0] for f in curr_no_fb])
        nofb_size_max = max([f[0] for f in curr_no_fb])
        nofb_drot_min = min([f[1] for f in curr_no_fb])
        nofb_drot_max = max([f[1] for f in curr_no_fb])
        #print("size - no feedback: [%i, %i]" % (nofb_size_min, nofb_size_max))
        #print("dept rot - no feedback: [%i, %i]" % (nofb_drot_min, nofb_drot_max))
        fb_sz_ixs = (list(expected_sizes).index(nofb_size_min), list(expected_sizes).index(nofb_size_max))
        fb_drot_ixs = (list(expected_drots).index(nofb_drot_min), list(expected_drots).index(nofb_drot_max))

        n_fb_drot = fb_drot_ixs[1]-fb_drot_ixs[0]+1
        n_fb_sz = fb_sz_ixs[1]-fb_sz_ixs[0]+1

        ax.add_patch(Rectangle((fb_drot_ixs[0]-offset, fb_sz_ixs[0]-offset), n_fb_drot, n_fb_sz, 
                               fill=False, edgecolor='cornflowerblue', lw=lw, linestyle=':', label='no feedback'))

    return ax

def format_size_depth_ticks(ax, xvals=[], yvals=[], 
                            xmax=None, ymax=None,
                            minimal=True, seaborn=True):
    
    offset = 0.5 if seaborn else 0.
    
    if minimal:
        xmax = max(xvals) if xmax is None else xmax
        xtick_labels = [int(x) for x in xvals if x%xmax==0 ]
        xticks = [i+offset for i, x in enumerate(xvals) if x%xmax==0 ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

        ymax = max(yvals) if ymax is None else ymax
        ytick_labels = [int(x) for x in yvals if x%ymax==0 ]
        yticks = [i+offset for i, x in enumerate(yvals) if x%ymax==0 ]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation=0)
    else:
        ax.set_xticks([i+offset for i in np.arange(0, len(xvals))])
        ax.set_xticklabels(xvals)

        ax.set_yticks([i+offset for i in np.arange(0, len(yvals))])
        ax.set_yticklabels(yvals)
    
    return ax

def set_split_xlabels(ax, n_bars=3, offset=0.25, a_label='rfs', b_label='rfs10', rotation=0, ha='center'):
    xticks = []
    xticklabels=[]
    for b in np.arange(n_bars):
        xticks.extend([b-offset, b+offset])
        xticklabels.extend([a_label, b_label])

    #ax.set_xticks([0-offset, 0+offset, 1-offset, 1+offset, 2-offset, 2+offset])
    #ax.set_xticklabels([a_label, b_label, a_label, b_label, a_label, b_label], rotation=rotation, ha=ha
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('')
    ax.tick_params(axis='x', size=0)
    sns.despine(bottom=True, offset=4)
    return ax

def pairwise_compare_metric(comdf, curr_metric='accuracy', sorter='animalid',
                            c1=1, c2=2, compare_var='objectid',
                            column_var=None, column_colors=None,
                            ax=None, marker='o', col_colors=None, dpi=150):
    assert sorter in comdf.columns, "Need a sorter, <%s> not found." % sorter

    if column_var is None: 
        column_var='placeholder'
        comdf['placeholder'] = 'placeholder'
        column_vals = ['placeholder']

    column_vals = comdf[column_var].unique()

    if column_colors is None:
        colors = sns.color_palette(palette='colorblind', n_colors=len(column_vals))
        column_colors = dict((k, v) for k, v in zip(column_vals, colors))

    offset = 0.25
    
    if ax is None:
        fig, ax = pl.subplots(figsize=(5,4), dpi=dpi)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    # Plot paired values
    aix=0
    for ai, (column_val, plotdf) in enumerate(comdf.groupby([column_var])):
        a_df = plotdf[plotdf[compare_var]==c1].sort_values(by=sorter)
        b_df = plotdf[plotdf[compare_var]==c2].sort_values(by=sorter)
        a_=a_df[sorter].unique()
        b_ = b_df[sorter].unique()
        both_ = np.intersect1d(a_, b_)
        
        a_vals = a_df[a_df[sorter].isin(both_)][curr_metric].values
        b_vals = b_df[b_df[sorter].isin(both_)][curr_metric].values
        #a_vals = plotdf[plotdf[compare_var]==c1].sort_values(by=sorter)[curr_metric].values
        #b_vals = plotdf[plotdf[compare_var]==c2].sort_values(by=sorter)[curr_metric].values

        by_exp = [(a, e) for a, e in zip(a_vals, b_vals)]
        for pi, p in enumerate(by_exp):
            ax.plot([aix-offset, aix+offset], p, marker=marker, 
                    color=column_colors[column_val], 
                    alpha=1, lw=0.5,  zorder=0, markerfacecolor=None, 
                    markeredgecolor=column_colors[column_val])
        tstat, pval = spstats.ttest_rel(a_vals, b_vals)
        print("%s: (t-stat:%.2f, p=%.2f)" % (column_val, tstat, pval))
        aix = aix+1

    # Plot average
    sns.barplot(column_var, curr_metric, data=comdf, 
                hue=compare_var, hue_order=[c1, c2], #zorder=0,
                ax=ax, order=column_vals,
                errcolor="k", edgecolor='k', 
                facecolor='none', linewidth=2.5)
    ax.legend_.remove()

    set_split_xlabels(ax, n_bars=len(column_vals), a_label=c1, b_label=c2)
    
    return ax


def annotateBars(row, ax, fontsize=12, fmt='%.2f', fontcolor='k', xytext=(0, 10)): 
    for p in ax.patches:
        ax.annotate(fmt % p.get_height(), (p.get_x() + p.get_width() / 2., 0.), #p.get_height()),
                    ha='center', va='center', fontsize=fontsize, color=fontcolor, 
                    rotation=0, xytext=xytext, #(0, 10),
             textcoords='offset points')
 
