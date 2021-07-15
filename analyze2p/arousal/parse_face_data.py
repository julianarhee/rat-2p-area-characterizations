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


def parse_pose_data(datakey, experiment, 
                    traceid='traces001', 
                    iti_pre=1., iti_post=1., feature_name='pupil_area', 
                    alignment_type='trial', realign=False, recombine=False, 
                    snapshot=391800, verbose=False,
                    rootdir='/n/coxfs01/2p-data',
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    '''
    Load and parse eyetracker data, align to MW trials. 
    Calculates metrics, then saves to combined_run dir in 'facetracker' subdir.

    iti_pre, iti_post: float    
        How much of the trial to use for pre/post, in SEC.

    return_errors: bool
        Set TRUE to return dict of missing/bad file info.

    feature_name: str
        What feature to extract. If "pupil_fraction" this is calculate internally.
    
    alignment_type: str
        'trial' means use pre/post ITI + stimulus duration (read from MW file).
        'stimulus' means just use stimulus duration (ignores pre/post ITI).

    Outputs:
        
    ptraces: pd.DatFrame
        Saved to: .../combined_blobs_static/facetracker/FNAME.pkl

    params: dict
        Saved to: .../combined_blobs_static/facetracker/FNAME_params.json

    '''
    print("Parsing pose data.")
    #### Load pupil data
    feature_list = []
    if 'pupil' in feature_name:
        feature_list.append(feature_name)
    trialmeta, pupildata, params = dlcutils.get_pose_data(datakey, experiment,
                                        snapshot=snapshot,
                                        feature_list=feature_list,
                                        alignment_type=alignment_type,
                                        iti_pre=iti_pre, iti_post=iti_post, 
                                        verbose=verbose, 
                                        realign=realign, recombine=recombine,
                                        eyetracker_dir=eyetracker_dir)

    #### Parse pupil data into traces
    ptraces=None; missing_trials=None;
    if (trialmeta is not None) and (pupildata is not None):
        # Get labels
        labels = aggr.load_frame_labels(datakey, experiment, 
                                    traceid=traceid, rootdir=rootdir)
        # Split traces into trials
        ptraces, missing_trials = dlcutils.traces_to_trials(
                                    trialmeta, pupildata, labels,
                                    return_missing=True)

        params['missing_trials'] = missing_trials

    # Set output dir
    session, animalid, fovnum = hutils.split_datakey_str(datakey)                 
    rundir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovnum,
                            'combined_%s_static' % experiment))[0]
    dst_dir = os.path.join(rundir, 'facetracker')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    print("    saving output to: %s" % dst_dir) 
    # Setup output files. Create parse id.
    parse_id = dlcutils.create_parsed_traces_id(experiment=experiment, 
                                       alignment_type=alignment_type,
                                       snapshot=snapshot) 
    params_fpath = os.path.join(dst_dir, '%s_params.json' % parse_id)
    results_fpath = os.path.join(dst_dir, '%s_traces.pkl' % parse_id)

    if ptraces is not None:
        # Save params
        with open(params_fpath, 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)
        # Save aligned traces
        with open(results_fpath, 'wb') as f:
            pkl.dump(ptraces, f, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("    !!! no data !!!!!")

    return ptraces, params


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

    choices_e = ('pre', 'stimulus', 'all')
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

    parser.add_option('--new', action='store_true', dest='redo_fov', 
            default=False, help="re-extract pupil traces and dataframe")
    parser.add_option('--meta', action='store_true', dest='realign', 
            default=False, help="Re-align trials (get new triger indices). True, if redo_fov is flagged.")
    parser.add_option('--combine', action='store_true', dest='recombine', 
            default=False, help="Recombine pupil data. True if redo_fov flagged.")


    (options, args) = parser.parse_args(options)

    return options

def parse_all_missing(experiment, redo_fov=False,
                    iti_pre=1., iti_post=1., #stim_dur=1., 
                    feature_name='pupil_area', alignment_type='trial', 
                    pupil_epoch='pre', 
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
    edata = sdata[sdata['experiment']==experiment]

    # Load pupil traces
    if redo_fov:
        missing_dsets = edata['datakey'].unique()
        realign=True
        recombine=True
    else: 
        pupildata, missing_dsets = dlcutils.get_aggregate_dataframes(
                                        experiment=experiment, 
                                        feature_name=feature_name, 
                                        alignment_type=alignment_type, 
                                        trial_epoch=pupil_epoch,
                                        iti_pre=iti_pre, iti_post=iti_post, 
                                        in_rate=pupil_framerate, 
                                        out_rate=pupil_framerate,
                                        snapshot=snapshot, 
                                        create_new=True, redo_fov=redo_fov,
                                        return_missing=True)

    dsets_todo = edata[edata['datakey'].isin(missing_dsets)]
    print("Missing %i pupil data sets. Parsing pose data now." % len(missing_dsets))

    for (animalid, session, fov, datakey), g \
            in dsets_todo.groupby(['animalid', 'session', 'fov', 'datakey']):

        if datakey in exclude_for_now:
            print("Need to retransfer (%s), skipping for now" % datakey)
            continue
        # Parse extracted pose traces and save to disk
        ptraces, params = parse_pose_data(datakey, experiment, 
                            traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post, 
                            feature_name=feature_name, 
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
    alignment_type = opts.alignment_type
    pupil_epoch = opts.pupil_epoch

    realign = opts.realign
    recombine = opts.recombine
    redo_fov = opts.redo_fov
    if redo_fov:
        realign=True
        recombine=True

    verbose = opts.verbose

    snapshot=int(opts.dlc_snapshot)
    rootdir = opts.rootdir
    eyetracker_dir = opts.eyetracker_dir
    print("TRACEID: %s" % traceid)

    if animalid is None:
        missing = parse_all_missing(experiment, redo_fov=redo_fov,
                            traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post,
                            feature_name=feature_name, 
                            alignment_type=alignment_type, 
                            pupil_epoch=pupil_epoch,
                            snapshot=snapshot, verbose=verbose,
                            rootdir=rootdir, eyetracker_dir=eyetracker_dir)
        print("Done!")
        print("Missing DLC or eyetracker data for %s dsets: " % (len(missing)))
        for i in missing:
            print(i)

    else:
        ptraces, params = parse_pose_data(datakey, experiment, 
                            traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post, 
                            feature_name=feature_name, 
                            alignment_type=alignment_type, 
                            snapshot=snapshot,
                            rootdir=rootdir,
                            eyetracker_dir=eyetracker_dir,
                            realign=realign, recombine=recombine)
        print("******[%s|%s|%s] Finished parsing eyetracker data (snapshot=%i)" % (animalid, session, experiment, snapshot))

    return


if __name__=='__main__':
    main(sys.argv[1:])


