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

def load_frame_labels(animalid, session, fov, experiment, traceid='traces001',
                        rootdir='/n/coxfs01/2p-data'):
    
    ds =glob.glob(os.path.join(rootdir, animalid, session, fov, 'combined_%s_static' % experiment, 'traces', '%s*' % traceid))
    print(ds)
    if len(ds)==0:
        print("Animalid:%s, Session:%s, fov:%s, Exp: %s|%s" % (animalid, session, fov, experiment, traceid))

    labels_dfile = glob.glob(os.path.join(rootdir, animalid, session, fov, 
                        'combined_%s_static' % experiment, 'traces', 
                        '%s*' % traceid, 'data_arrays', 'labels.npz'))[0]
    l = np.load(labels_dfile, allow_pickle=True)
    labels = pd.DataFrame(data=l['labels_data'], columns=l['labels_columns'])
    #print(labels.head())
    labels = hutils.convert_columns_byte_to_str(labels)

    return labels

def parse_traces_for_experiment(animalid, session, fov, experiment, 
                    traceid='traces001',
                    iti_pre=1., iti_post=1., feature_name='pupil_area', 
                    alignment_type='trial', 
                    return_errors=False,
                    snapshot=391800,
                    rootdir='/n/coxfs01/2p-data',
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):

    #### Get labels
    labels = load_frame_labels(animalid, session, fov, experiment, 
                                traceid=traceid, rootdir=rootdir)

    #### Get sources
    dlc_results_dir, dlc_video_dir = dlcutils.get_dlc_sources()


    #### Load pupil data
    fovnum = int(fov.split('_')[0][3:])
    facemeta, pupildata, missing_dlc, bad_files = dlcutils.load_pose_data(
                                        animalid, session, 
                                        fovnum, experiment, 
                                        dlc_results_dir, 
                                        snapshot=snapshot,
                                        feature_list=[feature_name], 
                                        alignment_type=alignment_type,
                                        pre_ITI_ms=iti_pre*1000., 
                                        post_ITI_ms=iti_post*1000., 
                                        return_bad_files=True,
                                        eyetracker_dir=eyetracker_dir)

    #### Parse pupil data into traces
    print("in parse:")
    pupiltraces=None; missing_trials=None;
    if (facemeta is not None) and (pupildata is not None):
        #print(facemeta.head())
        pupiltraces, missing_trials = dlcutils.get_pose_traces(
                                    facemeta, pupildata, 
                                    labels, feature=feature_name,
                                    return_missing=True)

    missing_data = {'no_dlc_files': missing_dlc,
                    'empty_dlc_files': bad_files,
                    'missing_trials': missing_trials}

    if return_errors:
        return pupiltraces, missing_data
    else:
        return pupiltraces

def parse_pose_data(animalid, session, fov, experiment, 
                    traceid='traces001', 
                    iti_pre=1., iti_post=1., feature_name='pupil_area', 
                    alignment_type='trial', 
                    snapshot=391800,
                    rootdir='/n/coxfs01/2p-data',
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):

    ptraces=None; params=None;
    # Set output dir
    rundir = glob.glob(os.path.join(rootdir, animalid, session, fov, 
                            'combined_%s_static' % experiment))[0]

    dst_dir = os.path.join(rundir, 'facetracker')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    print("Saving output to: %s" % dst_dir)
    
    # Create parse id
    parse_id = dlcutils.create_parsed_traces_id(experiment=experiment, 
                               alignment_type=alignment_type,
                               feature_name=feature_name,
                               snapshot=snapshot) 

    params_fpath = os.path.join(dst_dir, '%s_params.json' % parse_id)
    results_fpath = os.path.join(dst_dir, '%s.pkl' % parse_id)

    # parse
    ptraces, missing_data =  parse_traces_for_experiment(
                                    animalid, session, fov, experiment, 
                                    traceid=traceid,
                                    iti_pre=iti_pre, iti_post=iti_post, 
                                    feature_name=feature_name, 
                                    alignment_type=alignment_type, 
                                    return_errors=True,
                                    snapshot=snapshot,
                                    rootdir=rootdir, 
                                    eyetracker_dir=eyetracker_dir) 

    # Save params
    if ptraces is not None:
        params = {'experiment': experiment, 'traceid': traceid,
                  'iti_pre': iti_pre, 
                  'iti_post': iti_post,
                  'alignment_type': alignment_type,
                  'missing_dlc_files': missing_data['no_dlc_files'],
                  'empty_dlc_files': missing_data['empty_dlc_files'], #files_,
                  'missing_trials': missing_data['missing_trials'], #trials_,
                  'n_missing_trials': len(missing_data['missing_trials']),
                  'feature_name': feature_name, 
                  'raw_src': eyetracker_dir}
        with open(params_fpath, 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)

        # Save results
        with open(results_fpath, 'wb') as f:
            pkl.dump(ptraces, f, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("!!! no data !!!!!")


    return ptraces, params


def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', 
              default='/n/coxfs01/2p-data',\
              help='data root dir [default: /n/coxfs01/2pdata]')
    parser.add_option('-Y', '--eye', action='store', dest='eyetracker_dir', 
            default='/n/coxfs01/2p-data/eyetracker_tmp',
            help='eyetracker src files dir [default: /n/coxfs01/2p-data/eyetracker_tmp]')


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
            help="Feature to extract, choices: %s. (default: %s" % (choices_p, default_p))


       
    # data filtering 
    parser.add_option('-V', action='store_true', dest='verbose', 
            default=False, help="verbose printage")
    parser.add_option('--new', action='store_true', dest='create_new', 
            default=False, help="re-do decode")

    (options, args) = parser.parse_args(options)

    return options

def parse_all_missing(experiment, 
                    iti_pre=1., iti_post=1., stim_dur=1., 
                    feature_name='pupil_area', alignment_type='trial', pupil_epoch='pre', 
                    pupil_framerate=20., snapshot=391800, 
                    traceid='traces001', fov_type='zoom2p0x', state='awake',
                    rootdir='/n/coxfs01/2p-data', 
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    exclude_for_now = ['20190315_JC070_fov1']
    missing=[]

    # Get all datasets
    sdata = aggr.get_aggregate_info(traceid=traceid, 
                                    fov_type=fov_type, state=state)
    edata = sdata[sdata['experiment']==experiment]

    # Load pupil traces (for plotting)
    #pupiltraces, missing_dsets = dlcutils.get_aggregate_pupil_traces(experiment, feature_name=feature_name,
    #                                              alignment_type=alignment_type,
    #                                              traceid=traceid, create_new=True, return_missing=True)
    pupildata, missing_dsets = dlcutils.get_aggregate_pupildfs(
                                        experiment=experiment, 
                                        feature_name=feature_name, 
                                        alignment_type=alignment_type, 
                                        trial_epoch=pupil_epoch,
                                        iti_pre=iti_pre, iti_post=iti_post, 
                                        stim_dur=stim_dur,
                                        in_rate=pupil_framerate, 
                                        out_rate=pupil_framerate,
                                        snapshot=snapshot, create_new=True, 
                                        return_missing=True)

    dsets_todo = edata[edata['datakey'].isin(missing_dsets)]
    print("Missing %i pupil data sets. Parsing pose data now." % len(missing_dsets))

    for (animalid, session, fov, datakey), g in dsets_todo.groupby(['animalid', 'session', 'fov', 'datakey']):
        if datakey in exclude_for_now:
            print("Need to retransfer (%s), skipping for now" % datakey)
            continue

        ptraces, params = parse_pose_data(animalid, session, fov, experiment, 
                            traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post, 
                            feature_name=feature_name, 
                            alignment_type=alignment_type, 
                            snapshot=snapshot,
                            rootdir=rootdir,
                            eyetracker_dir=eyetracker_dir)

        print("******[%s|%s] Finished parsing eyetracker data (snapshot=%i)" % (datakey, experiment, snapshot))
        if ptraces is None:
            print("[ERROR]: No DLC found, %s" % datakey)
            missing.append(datakey)
            continue
        else:
            print('MISSING: %s' % str(params['missing_dlc_files']))

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

    snapshot=int(opts.dlc_snapshot)
    rootdir = opts.rootdir
    eyetracker_dir = opts.eyetracker_dir
    print("TRACEID: %s" % traceid)

    if animalid is None:
        missing = parse_all_missing(experiment, traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post,
                            feature_name=feature_name, 
                            alignment_type=alignment_type, 
                            pupil_epoch=pupil_epoch,
                            snapshot=snapshot,
                            rootdir=rootdir, eyetracker_dir=eyetracker_dir)
        print("Done!")
        print("Missing DLC or eyetracker data for %s dsets: " % (len(missing)))
        for i in missing:
            print(i)
    else:
        ptraces, params = parse_pose_data(animalid, session, fov, experiment, 
                            traceid=traceid, 
                            iti_pre=iti_pre, iti_post=iti_post, 
                            feature_name=feature_name, 
                            alignment_type=alignment_type, 
                            snapshot=snapshot,
                            rootdir=rootdir,
                            eyetracker_dir=eyetracker_dir)
        print("******[%s|%s|%s] Finished parsing eyetracker data (snapshot=%i)" % (animalid, session, experiment, snapshot))

    return


if __name__=='__main__':
    main(sys.argv[1:])

