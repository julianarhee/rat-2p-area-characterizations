#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  Nov 12 12:04:28 2020

@author: julianarhee
"""
import matplotlib as mpl
mpl.use('agg')
import sys
import optparse
import os
import json
import glob
import copy
import copy
import itertools
import re
import datetime
import pprint 
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import statsmodels as sm
import dill as pkl

from scipy import stats as spstats

#from pipeline.python.classifications import experiment_classes as util
#from pipeline.python.classifications import aggregate_data_stats as aggr
#from pipeline.python import utils as putils
#from pipeline.python.eyetracker import dlc_utils as dlcutils
from analyze2p.arousal import dlc_utils as dlcutils
import analyze2p.aggregate_datasets as aggr
import analyze2p.utils.helpers as hutils



def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', 
              default='/n/coxfs01/2p-data',\
              help='data root dir [default: /n/coxfs01/2pdata]')
    parser.add_option('-Y', '--eye', action='store', dest='eyetracker_dir', 
            default='/n/coxfs01/2p-data/eyetracker_tmp',
            help='eyetracker raw dir [default: /n/coxfs01/2p-data/eyetracker_tmp]')

    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', 
            default='blobs', help="experiment type [default: blobs]")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', 
            default='traces001', help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--response-type', action='store', 
            dest='response_type', default='dff', 
            help="response type [default: dff]")

    parser.add_option('-i', '--animalid', action='store', dest='animalid', 
            default=None, help="animalid")
    parser.add_option('-A', '--fov', action='store', dest='fov', 
            default='FOV1_zoom2p0x', help="fov (default: FOV1_zoom2p0x)")
    parser.add_option('-S', '--session', action='store', dest='session', 
            default='', help="session (YYYYMMDD)")


    parser.add_option('-p', '--iti-pre', action='store', dest='iti_pre', 
            default=1.0, help="pre-stimulus iti (dfeault: 1.0 sec)")
    parser.add_option('-P', '--iti-post', action='store', dest='iti_post', 
            default=1.0, help="post-stimulus iti (dfeault: 1.0 sec)")

    parser.add_option('-s', '--snap', action='store', dest='dlc_snapshot', 
            default=391800, help="Snapshot num, if using dlc (default: 391800)")

    choices_c = ('trial', 'stimulus')
    default_c = 'stimulus'
    parser.add_option('-a', '--align', action='store', dest='alignment_type', 
            default=default_c, type='choice', choices=choices_c,
            help="Alignment choices: %s. (default: %s" % (choices_c, default_c))

    choices_e = ('pre', 'stimulus','plushalf', 'all')
    default_e = 'pre'
    parser.add_option('-e', '--epoch', action='store', dest='pupil_epoch', 
            default=default_e, type='choice', choices=choices_e,
            help="Average epoch over: %s. (default: %s" % (choices_e, default_e))

    choices_p = ('pupil_area', 'snout', 'pupil_fraction')
    default_p = 'pupil_area'
    parser.add_option('-f', '--feature', action='store', dest='feature_name', 
            default=default_p, type='choice', choices=choices_p,
            help="Feature to extract, choices: %s. (default: %s" \
            % (choices_p, default_p))
 
    # data filtering 
    parser.add_option('-V', action='store_true', dest='verbose', 
            default=False, help="verbose printage")

    parser.add_option('--new', action='store_true', dest='create_new', 
            default=False, help="re-save aggregate dataframes")
    parser.add_option('--realign', action='store_true', dest='realign', 
            default=False, help="Re-align trials (trigger indices). True, if redo_fov is flagged.")
    parser.add_option('--recombine', action='store_true', dest='recombine', 
            default=False, help="Recombine pupil data. True if redo_fov flagged.")


    (options, args) = parser.parse_args(options)

    return options

def parse_all_missing(experiment, create_new=False,
                    realign=False, recombine=False,
                    iti_pre=1., iti_post=1., #stim_dur=1., 
                    alignment_type='trial', pupil_epoch='pre', 
                    feature_list=['pupil'],
                    pupil_framerate=20., snapshot=391800, 
                    traceid='traces001', fov_type='zoom2p0x', state='awake',
                    rootdir='/n/coxfs01/2p-data', verbose=False,
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):

    '''
    Cycle through all datasets for specified experiment type, get pupildata.
   
    redo_fov: bool
        Set TRUE to re-save extracted/aligned pupil traces for each FOV.

    '''

    exclude_for_now = ['20190315_JC070_fov1', '20190517_JC083_fov1']
    missing=[]

    # Get all datasets
    sdata = aggr.get_aggregate_info(traceid=traceid, 
                                    fov_type=fov_type, state=state)
    edata = sdata[(sdata['experiment']==experiment)
                & ~(sdata['datakey'].isin(exclude_for_now))].copy()

    # Load pupil traces
    if realign or recombine:
        missing_dsets = edata['datakey'].unique()
        create_new=True
    
    aggr_dfs, aggr_params, missing_dsets = dlcutils.aggregate_dataframes(
                                        experiment,
                                        alignment_type=alignment_type, 
                                        trial_epoch=pupil_epoch,
                                        iti_pre=iti_pre, iti_post=iti_post, 
                                        in_rate=pupil_framerate, 
                                        out_rate=pupil_framerate,
                                        snapshot=snapshot, 
                                        create_new=create_new,
                                        realign=realign, recombine=recombine, 
                                        return_missing=True)

    dsets_todo = edata[edata['datakey'].isin(missing_dsets)]
    print("Missing pupil traces for %i dsets. Parsing pose data for these now." % len(missing_dsets))

    for (animalid, session, fov, datakey), g \
            in dsets_todo.groupby(['animalid', 'session', 'fov', 'datakey']):
        print("... parsing pose data: %s" % datakey)

        if datakey in exclude_for_now:
            print("Need to retransfer (%s), skipping for now" % datakey)
            continue
        # Parse extracted pose traces and save to disk
        ptraces, params = dlcutils.parse_pose_data(datakey, experiment, 
                            traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post, 
                            feature_list=feature_list,
                            alignment_type=alignment_type, 
                            snapshot=snapshot, verbose=verbose,
                            rootdir=rootdir,
                            eyetracker_dir=eyetracker_dir,
                            realign=realign, recombine=recombine)

        print("    %s, %s: finished parsing steps (snapshot=%i)" \
            % (datakey, experiment, snapshot))
        if ptraces is None:
            print("    [ERROR]: No DLC found, %s" % datakey)
            missing.append(datakey)
            continue
        else:
            print('    DLC files missing: %s' % str(params['dlc_missing_files']))

    return missing #None


def main(options):
    opts = extract_options(options)

    animalid=opts.animalid
    session=opts.session
    fov=opts.fov
    experiment=opts.experiment
    traceid=opts.traceid

    iti_pre=float(opts.iti_pre)
    iti_post=float(opts.iti_post)
    feature_name = opts.feature_name
    feature_list=[]
    if 'pupil' in feature_name:
        feature_list.append('pupil')

    alignment_type = opts.alignment_type
    pupil_epoch = opts.pupil_epoch

    realign = opts.realign
    recombine = opts.recombine
    create_new = opts.create_new
    if realign or recombine:
        create_new=True

    verbose = opts.verbose

    snapshot=int(opts.dlc_snapshot)
    rootdir = opts.rootdir
    eyetracker_dir = opts.eyetracker_dir
    print("TRACEID: %s" % traceid)

    if animalid is None:
        missing = parse_all_missing(experiment,
                            traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post,
                            feature_list=feature_list,
                            alignment_type=alignment_type, 
                            pupil_epoch=pupil_epoch,
                            snapshot=snapshot, verbose=verbose,
                            rootdir=rootdir, eyetracker_dir=eyetracker_dir,
                            create_new=create_new, realign=realign, recombine=recombine)
        print("Done!")
        print("Missing DLC or eyetracker data for %s dsets: " % (len(missing)))
        for i in missing:
            print(i)

    else:
        ptraces, params = dlcutils.parse_pose_data(datakey, experiment, 
                            traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post, 
                            feature_list=feature_list,
                            alignment_type=alignment_type, 
                            snapshot=snapshot, verbose=verbose,
                            rootdir=rootdir,
                            eyetracker_dir=eyetracker_dir,
                            realign=realign, recombine=recombine)
 
        print("******[%s|%s|%s] Finished parsing eyetracker data (snapshot=%i)" % (animalid, session, experiment, snapshot))

    return


if __name__=='__main__':
    main(sys.argv[1:])


