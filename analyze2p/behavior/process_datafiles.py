#!/usr/bin/env python2

import matplotlib as mpl
#mpl.use('agg')
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

import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
#import cPickle as pkl
#from cPickle import PicklingError
try:
    import cPickle as pkl
except:
    import pickle as pkl
#from cPickle import PicklingError


from scipy import stats
#import parse_behavior as pb
import utils as util

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_session(animalid, session, metadata, n_processes=1, 
                plot_each_session=False, paradigm='threeport', create_new=False,
                response_types=['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore'],
                outcome_types=['success', 'ignore', 'failure'],
                rootdir='/n/coxfs01/behavior-data'):

    print("***********************************************")
    print("ANIMAL:  %s | SESSION: %i" % (animalid, session))
    print("***********************************************")
    
    # --- Get current animal session info:
    session_meta = metadata[(metadata['animalid']==animalid) & (metadata['session']==session)].copy()

    # --- process.
    start_t = time.time()
    S = util.get_session_data(session_meta, create_new=create_new)
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))

    print(S.flags)

    return S



def get_sessions_for_animal(animalid, metadata, n_processes=1, plot_each_session=False,
                               paradigm='threeport', create_new=False,
                               response_types=['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore'],
                               outcome_types=['success', 'ignore', 'failure'],
                               ignore_flags=None,
                               rootdir='/n/coxfs01/behavior-data'):

    # Set experiment parsing vars and params:
#    response_types = ['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore']
#    outcome_types = outcome_types = ['success', 'ignore', 'failure']
#    ignore_flags = None
#
    print("***********************************************")
    print("ANIMAL:  %s" % animalid)
    print("***********************************************")
    
    # --- Get current animal session info:
    session_meta = metadata[metadata['animalid']==animalid].copy()
    A, new_sessions = util.load_animal_data(animalid, paradigm, session_meta, rootdir=rootdir)

    # --- process.
    start_t = time.time()
    session_meta = metadata[(metadata['animalid']==animalid)] # & (metadata.session==session)]
    if create_new:
        print("!!! (Re)processing all sessions.")
        new_sessions = session_meta['session'].unique()

    processed_sessions = util.get_sessions_mp(new_sessions, session_meta,
                                             n_processes=n_processes, plot_each_session=plot_each_session,
                                             ignore_flags=ignore_flags,
                                             response_types=response_types,
                                             outcome_types=outcome_types, create_new=create_new)
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))

    # --- Update animal sessions dict:
    for datestr, S in processed_sessions.items():
        A.sessions.update({datestr: S})

    # --- Save to disk:
    try:
        with open(A.path, 'wb') as f:
            pkl.dump(A, f, protocol=pkl.HIGHEST_PROTOCOL)
    except Exception as e: #PicklingError:
        print("Unable to pkl: New sessions are not the same class as old sessions.")
        print("Reprocessing %i old sessions..." % len(processed_sessions))
        for session in old_sessions:
            curr_sessionmeta = session_meta[session_meta.session==session] #session_info[datestr]
            S = util.get_session_data(curr_sessionmeta)
            A.sessions[session] = S
        with open(A.path, 'wb') as f:
            pkl.dump(A, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("[%s] ~~~ processing complete! ~~~" % A.animalid)

    return A



def get_animal_df(animalid, paradigm, metadata, create_new=False, rootdir='/n/coxfs01/behavior-data'):
   
    new_sessions=[]
    no_trials = []
    df = None

    # Check for dataframe
    outdir = os.path.join(rootdir, paradigm, 'processed', 'data')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    d_outfile = os.path.join(outdir, 'df_%s.pkl' % animalid)
    
    reload_df = False
    if os.path.exists(d_outfile) and create_new is False:
        print("... loading existing df")
        with open(d_outfile, 'rb') as f:
            df = pkl.load(f, encoding='latin1')
    else:
        reload_df = True

    if reload_df:
        print("... making dataframe for %s" % animalid)
        A, new_sessions = util.format_animal_data(animalid, paradigm, metadata, rootdir=rootdir)
        df = util.animal_data_to_dataframe(A)
        #df, new_sessions, no_trials = sessiondata_to_df(animalid, paradigm, metadata, rootdir=rootdir)
        if df is None:
            return None, None
        else:
            with open(d_outfile, 'wb') as f:
                pkl.dump(df, f, protocol=pkl.HIGHEST_PROTOCOL)

    return df, new_sessions


def sessiondata_to_df(animalid, paradigm, metadata, rootdir='/n/coxfs01/behavior-data'):
   
    A = util.load_session_data(animalid, paradigm, metadata, rootdir=rootdir)

    assert len(A.sessions)>0, "No sessions found"

    df = None
    sessionlist = []
    no_trials = []
    for si, (session, sessionobj) in enumerate(A.sessions.items()):
        if si % 20 == 0:
            print("... adding %i of %i sessions." % (int(si+1), len(A.sessions)))

        if sessionobj is None or sessionobj.trials is None or len(sessionobj.trials)==0:
            no_trials.append(session)
            continue

        trialdf = util.trialdict_to_dataframe(sessionobj.trials, session=session)
        if trialdf is not None:
            sessionlist.append(trialdf)
            
    if len(sessionlist) > 0:
        df = pd.concat(sessionlist, axis=0)
    print('%i sessions have no trials' % len(no_trials))

    return df


# --------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------
def combine_cohorts_to_dataframe(metadata, paradigm='threeport', cohorts=[], excluded_animals=[], rootdir='/n/coxfs01/behavior-data'):
    
    if len(cohorts) == 0:
        cohorts = sorted(metadata['cohort'].unique(), key=natural_keys)
    print("combining data from %i cohorts:" % len(cohorts), cohorts)
    
    dflist = []
    for (cohort, animalid), animal_meta in metadata.groupby(['cohort', 'animalid']):
        if animalid in excluded_animals:
            print("... skipping %i" % animalid)
            continue
            
        if cohort not in cohorts:
            continue
            
        a_df, _ = get_animal_df(animalid, paradigm, metadata, create_new=False, rootdir=rootdir)
        
        if a_df is None:
            print("... no DF found: %s" % animalid)
            continue
        a_df = a_df.reset_index(drop=True)

        #included_sessions = check_against_manual_sorting(animalid, phase)
        #currdf = a_df[a_df['session'].isin(included_sessions)].copy()
    
        #### Update some sorting values
        a_df['animalid'] = [animalid for _ in np.arange(0, len(a_df))]
        a_df['cohort'] = [cohort for _ in np.arange(0, len(a_df))]
        a_df['sessionid'] = ['%s%s' % (sess, sfx) for sess, sfx in zip(a_df['session'], a_df['suffix'])]
        
        dflist.append(a_df)
        
    df = pd.concat(dflist, axis=0).reset_index(drop=True)

    ignore_vars = ['Flag', 'action', 'alpha_multiplier', 'size_x', 'size_y']
    ignore_cols = [f for f in df.columns if any([desc in f for desc in ignore_vars])]
    keep_cols = [f for f in df.columns if f not in ignore_cols]
    df = df[keep_cols]

    return df




# --------------------------------------------------------------------

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/behavior-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/behavior-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default=None, 
                      help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default=None, 
                      help='session (YYYYMMDD)')

    parser.add_option('-c', '--cohort', action='store', dest='cohort', default=None, 
                      help='cohort, e.g, AL)')

    parser.add_option('-p', '--paradigm', action='store', dest='paradigm', default='threeport', 
            help='paradigm used (default:  threeport')

    parser.add_option('-n', '--nproc', action='store', dest='n_processes', default=1, 
            help='N processes to use (default:  1)')


    parser.add_option('--plot-session', action='store_true', dest='plot_each_session', default=False, 
            help='flag to plot performance for each sessions')
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
            help='flag to reprocess sessions anew')

    parser.add_option('--meta', action='store_true', dest='create_meta', default=False, 
            help='flag to recreate metadata (if adding new datafiles)')
 
    (options, args) = parser.parse_args(options)

    return options




def main(options):
    opts = extract_options(options)
    animalid = opts.animalid
    session = int(opts.session) 
    n_processes = int(opts.n_processes)
    create_new = opts.create_new
    create_meta = opts.create_meta

    paradigm = opts.paradigm
    plot_each_session = opts.plot_each_session
    rootdir = opts.rootdir

    #### Load metadata
    metadata = util.get_metadata(paradigm, create_meta=create_meta)

    S = get_session(animalid, session, metadata, n_processes=n_processes,
                        plot_each_session=plot_each_session, paradigm=paradigm, 
                        create_new=create_new)


    print('~ done ~')


if __name__ == '__main__':
    main(sys.argv[1:])


