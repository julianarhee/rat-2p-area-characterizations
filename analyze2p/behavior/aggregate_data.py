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

import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
#import cPickle as pkl
#from cPickle import PicklingError
import scipy.stats as spstats

import utils as util
import assign_phase as ph
import process_datafiles as procd


import pprint

pp = pprint.PrettyPrinter(indent=4)


def combine_cohorts_to_dataframe(metadata, cohort_list=[], excluded_animals=[],
                                paradigm='threeport', rootdir='/n/coxfs01/behavior-data'):
    
    if cohort_list is None or len(cohort_list) == 0:
        cohort_list = sorted(metadata['cohort'].unique(), key=util.natural_keys)
    print("combining data from %i cohorts:" % len(cohort_list), cohort_list)
    
    dflist = []
    for (cohort, animalid), animal_meta in metadata[metadata['cohort'].isin(cohort_list)].groupby(['cohort', 'animalid']):
        if animalid in excluded_animals:
            print("... skipping %s" % animalid)
            continue

        a_df, _ = procd.get_animal_df(animalid, paradigm, metadata, create_new=False, rootdir=rootdir)
        
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

def get_cohort_data_by_phase(cohortdf, phase_list=[], cohort_list=None, 
                        rootdir='/n/coxfs01/behavior-data'):
    
    assert len(phase_list) > 0, "No phases specified."
    if cohort_list is None:
        cohort_list = cohortdf['cohort'].unique()
        print("No cohorts specified, extracting all:", cohort_list)
        
    df_ = []
    for cohort in cohort_list: #['AL']:

        #### Load phase info for cohort
        phaseinfo = ph.get_phase_data(cohort, create_new=False, rootdir=rootdir)

        #### Get phase infor for current phase
        curr_phaseinfo =  phaseinfo[phaseinfo['phase'].isin(phase_list)]
        #print('Cohort %s: found phases' % cohort, phaseinfo['phase'].unique())

        #### Get datafiles for current phase
        for curr_phase, pgroup in curr_phaseinfo.groupby(['phase']):
            datafiles_in_phase = [s for s, g in pgroup.groupby(['animalid', 'session', 'suffix'])]

            #### Combine data for phase
            dlist = [g for s, g in cohortdf.groupby(['animalid', 'session', 'suffix']) if s in datafiles_in_phase]

            if len(dlist) > 0:
                tmpdf = pd.concat(dlist, axis=0).reset_index(drop=True)
                tmpdf['cohort'] = [cohort for _ in np.arange(0, len(tmpdf))]
                tmpdf['objectid'] = [str(i) for i in tmpdf['object']]
                tmpdf['phase'] = [curr_phase for _ in np.arange(0, len(tmpdf))]
                df_.append(tmpdf)

    df = pd.concat(df_, axis=0).reset_index(drop=True)
    
    return df


def get_portmapping(tmpdf, verbose=False):
    # object1_port3 = df[ (df['object']==1) & (df['outcome']=='success') 
    #                    & (df['response']=='Announce_AcquirePort3')  ]['animalid'].unique()
    # object1_port1 = df[ (df['object']==1) & (df['outcome']=='success') 
    #                    & (df['response']=='Announce_AcquirePort1')  ]['animalid'].unique()

    # object2_port3 = df[ (df['object']==2) & (df['outcome']=='success') 
    #                    & (df['response']=='Announce_AcquirePort3')  ]['animalid'].unique()
    # assert all(object1_port1 == object2_port3), "Mismatch for Object1-Port1 / Object2-Port3 mapping"

    # object2_port1 = df[ (df['object']==2) & (df['outcome']=='success') 
    #                    & (df['response']=='Announce_AcquirePort1')  ]['animalid'].unique()
    # assert all(object1_port3 == object2_port1), "Mismatch for Object1-Port3 / Object2-Port1 mapping"
    df = tmpdf[tmpdf['object']!='morph'].copy()
    df['objectid'] = df['objectid'].astype(int)

    portmapping = {'Object1_Port1': [], 'Object1_Port3': []}
    mixed_ = []
    # for animalid, dgroup in df[(df.objectid==1) & (df.outcome=='success')].groupby(['animalid']):
    for animalid, dgroup in df[(df.outcome=='success')].groupby(['animalid']):
        found_ports_1 = dgroup[(dgroup.objectid==1)]['response'].value_counts().keys()
        found_ports_2 = dgroup[(dgroup.objectid==2)]['response'].value_counts().keys()
        # found_ports = dgroup['response'].value_counts().keys()
        if len(found_ports_1)==1 and len(found_ports_2)==1:
            portname_1 = found_ports_1[0].split('Announce_Acquire')[-1]
            portmapping['Object1_%s' % portname_1].append(animalid)
            #portname_2 = found_ports_2[0].split('Announce_Acquire')[-1]
            #portmapping['Object2_%s' % portname_2].append(animalid)
        else:
            mixed_.append(animalid)

    #### For sessionids w/ multiple port-mappings, likely an error and should be filtered out
    bad_portmapping = dict((animalid, []) for animalid in mixed_)
    for animalid in mixed_:
        object1_port1 = df[(df.animalid==animalid) & (df.outcome=='success') 
                            & (df.objectid==1) & (df.response=='Announce_AcquirePort1')]['sessionid'].unique()
        object1_port3 = df[(df.animalid==animalid) & (df.outcome=='success')
                            & (df.objectid==1) & (df.response=='Announce_AcquirePort3')]['sessionid'].unique()
        object2_port1 = df[(df.animalid==animalid) & (df.outcome=='success') 
                            & (df.objectid==2) & (df.response=='Announce_AcquirePort1')]['sessionid'].unique()
        object2_port3 = df[(df.animalid==animalid) & (df.outcome=='success') 
                            & (df.objectid==2) & (df.response=='Announce_AcquirePort3')]['sessionid'].unique()


        if (len(object1_port1) > len(object1_port3)) and (len(object2_port3) > len(object2_port1)):
            # This is a Port1 mapper, with mistaken Port3 mapping
            # Object1 = Port1, Object2 = Port3
            portmapping['Object1_Port1'].append(animalid)
            bad_portmapping[animalid] = list(object1_port3)
            bad_portmapping[animalid].extend(object2_port1)
        elif (len(object1_port3) > len(object1_port1)) and (len(object2_port1) > len(object2_port3)):
            # Object1 = Port3, Object2 = Port1
            portmapping['Object1_Port3'].append(animalid)
            bad_portmapping[animalid] = list(object1_port1)
            bad_portmapping[animalid].extend(object2_port3)
        else:
            print(animalid, "[WARNING] %s -- funky mapping..." % animalid)
    if verbose:
        for animalid, badsessions in bad_portmapping.items():
            print("%s: %i sessions with bad mapping" % (animalid, len(badsessions)))

    #### Remove sessionids with incorrect mapping
    bad_port_ixs = [df[(df.animalid==animalid) & (df.sessionid.isin(sessionids))].index 
                    for animalid, sessionids in bad_portmapping.items()]
    for ix in bad_port_ixs:
        df.drop(ix, inplace=True)

    #### Now check animals w/ unique mapping
    for pmap, animalids in portmapping.items():
        other_pmap = 'Object2_Port3' if pmap=='Object1_Port1' else 'Object2_Port1'
        found_ports = df[(df.animalid.isin(animalids)) & (df.objectid==2) 
                         & (df.outcome=='success') & (df.response!='ignore')]['response'].unique()
        assert len(found_ports)==1 and 'Object2_%s' % found_ports[0].split('Announce_Acquire')[-1]==other_pmap, "[%s] %s" % (animalid, str(found_ports))

    #assert len(np.intersect1d(portmapping.values()[0], portmapping.values()[1]))==0, "[ERROR]: bad portmapping, got non-unique lists for each map. DEBUG."

    assert len(np.intersect1d(portmapping['Object1_Port1'], portmapping['Object1_Port3']))==0, \
            "[ERROR]: bad portmapping, got non-unique lists for each map. DEBUG."


    return portmapping


def assign_box_info(df, metadata, paradigm='threeport', rootdir='/n/coxfs01/behavior-data'):
    animal_ids = df['animalid'].unique()
    bboxes, towers = util.get_box_info(metadata[metadata.animalid.isin(animal_ids)], paradigm=paradigm)

    #### Assign info to unique animals first
    unique_setupids = [k for k, v in bboxes.items() if len(v) == 1 and k in animal_ids]
    print("%i of %i animalids with unique setup box nos." % (len(unique_setupids), len(animal_ids)))

    df['boxnum'] = [-1 for _ in  np.arange(0, len(df))]
    df['boxpos'] = [-1 for _ in  np.arange(0, len(df))]
    df['tower'] = [-1 for _ in  np.arange(0, len(df))]

    for animalid in unique_setupids:
        boxnum = int(re.search(r'(\d+)', bboxes[animalid][0]).group())
        towernum = [k for k, v in towers.items() if boxnum in v][0]
        df.loc[df['animalid']==animalid, 'boxnum'] = boxnum
        if boxnum != 0:
            df.loc[df['animalid']==animalid, 'tower'] = towernum
            df.loc[df['animalid']==animalid, 'boxpos'] = towers[towernum].index(boxnum)

    #### Check for animals run in more than 1 box
    multi_box = check_multi_box(bboxes, paradigm=paradigm, rootdir=rootdir)
    multi_box_animals = multi_box.keys()
    print("Found %i animals that were run in > 1 box" % len(multi_box_animals))
    
    #### Assign box info to all multi-box animals by session
    for (animalid, session), g in df[df['animalid'].isin(multi_box_animals)].groupby(['animalid', 'session']):
        curr_box = [int(bx) for bx, s_list in multi_box[animalid].items() if session in s_list][0]
        df.loc[g.index, 'boxnum'] = curr_box

        curr_tower = [tx for tx, bx in towers.items() if curr_box in bx][0]
        df.loc[g.index, 'tower'] = curr_tower
        df.loc[g.index, 'boxpos'] = towers[curr_tower].index(curr_box)

    return df, bboxes, towers


def check_multi_box(bboxes, paradigm='threeport', rootdir='/n/coxfs01/behavior-data'):
    multi_box_animals = [k for k, v in bboxes.items() if len(v) > 1]
    multi_box = dict((k, {}) for k in multi_box_animals)

    identify_multibox = False
    multi_box_fpath = os.path.join(rootdir, paradigm, 'processed', 'meta', 'multi_box_animals.json')

    if len(multi_box_animals) > 0:
        if os.path.exists(multi_box_fpath):
            print("...loading existing multi-box info:\n-->%s" % multi_box_fpath)
            with open(multi_box_fpath, 'r') as f:
                multi_box = json.load(f)
        else:
            identify_multibox = True
        # print(identify_multibox) 

    if identify_multibox:
        print("... identifying which animals ran in >1 box")
        for (animalid, session), smeta in metadata[metadata['animalid'].isin(multi_box_animals)].groupby(['animalid', 'session']):
            S = util.Session(smeta)
            _, _, metainfo = S.get_trials(create_new=False, verbose=False)
            if metainfo==-1:
                continue
            if isinstance(metainfo['server'], list):
                assert len(list(set(metainfo['server'])))==1, "ERROR. Setups switched mid day: %s" % str(session)
                boxname = list(set(metainfo['server']))[0]
            else:
                boxname = metainfo['server']

            boxnum = int(re.search(r'(\d+)', boxname).group())
            if boxnum not in multi_box[animalid].keys():
                multi_box[animalid][boxnum] = []
            multi_box[animalid][boxnum].append(session)

        with open(multi_box_fpath, 'wb') as f:
            json.dump(multi_box, f, indent=4, sort_keys=True)


    return multi_box



# Aggregating + adding info to dataframes
# ============================================================
def add_portmap(df_):
    portmapping = get_portmapping(df_)

    df_['portmap'] = None #[0 for _ in np.arange(0, len(df_))]
    df_.loc[df_['animalid'].isin(portmapping['Object1_Port1']), 'portmap'] = 1
    df_.loc[df_['animalid'].isin(portmapping['Object1_Port3']), 'portmap'] = 0

    return portmapping, df_
def add_train_day(df_):
    '''
    For each phase, get Nth day in that phase
    '''
    df_['train_day'] = -1
    for (animalid, curr_phase), g in df_.groupby(['animalid', 'phase']):
        sorted_sessions = list(sorted(g['sessionid'].unique(), key=util.natural_keys))
        sorted_session_ixs = [sorted_sessions.index(s) for s in g['sessionid']]
        df_.loc[g.index, 'train_day'] = sorted_session_ixs
    return df_


def get_counts_from_session_df(df_, group_all_conds=False):
    '''
    From aggregated session df, count all the trials per session for summary stats (e.g,. accuracy)
    '''
    #group_cols = ['cohort','animalid', 'session', 'phase']
    #group_cols = ['cohort', 'animalid', 'session', 'phase', 'objectid', 'train_day', 'portmap']
    group_cols = ['cohort', 'animalid', 'session', 'phase', 'objectid', 'portmap']

    if group_all_conds:
        group_cols.extend(['size', 'depth_rotation'])

    counts = df_.groupby(group_cols)['outcome']\
                          .value_counts().unstack().reset_index()
    counts=counts.fillna(0)

    if 'ignore' in counts:
        counts['n_total'] = counts['failure']+counts['success'] + counts['ignore']
    else:
        counts['n_total'] = counts['failure']+counts['success']
    #counts['n_total'] = counts['failure'] + counts['success']
    counts['accuracy'] = counts['success'] / counts['n_total']
    counts['accuracy'] = counts['accuracy'].astype(float)
    
    return counts


def pass_criterion(standard_df, criterion=0.7, defaults_only=False,
                            fullcross=True, group_all_conds=False, min_train_phase=1,
                            frac_max=0.75):
    max_train_phase = 4 if fullcross else 1

    if defaults_only:
        phase1_df = standard_df[ (standard_df['object']!='morph') 
                           & (standard_df['phase']<=max_train_phase) 
                           & (standard_df['size'].isin([30, 40])) 
                           & (standard_df['depth_rotation']==0)]#\
    else:        
        # Look at the end of phase trials
        phase1_df0 = pd.concat([g[g['train_day']>=g['train_day'].max()*frac_max] \
                            if ph==1 else g \
                            for ph, g in standard_df.groupby(['phase'])])
        phase1_df = phase1_df0[(phase1_df0['phase']<=max_train_phase)
                        & (phase1_df0['phase']>=min_train_phase)].copy()

    phase1_counts = get_counts_from_session_df(phase1_df, 
                        group_all_conds=group_all_conds)

    animalids = phase1_counts['animalid'].unique()
    mean_acc = phase1_counts.groupby(['animalid']).mean().reset_index()

    pass_animals = mean_acc[mean_acc['accuracy']>=criterion]['animalid'].unique()
    
    missing = [k for k in animalids if k not in pass_animals]

    print("%i of %i animals pass min crit. %.2f accuracy on training. Fail animals %i:\n  " \
          % (len(pass_animals), len(animalids), criterion, len(missing)), missing)

    return pass_animals, missing



def animals_pass_criterion(standard_df, criterion=0.7, defaults_only=False,
                            fullcross=True, group_all_conds=False, min_train_phase=1,
                            frac_max=0.75):
    max_train_phase = 4 if fullcross else 1

    if defaults_only:
        phase1_df = standard_df[ (standard_df['object']!='morph') 
                           & (standard_df['phase']<=max_train_phase) 
                           & (standard_df['size'].isin([30, 40])) 
                           & (standard_df['depth_rotation']==0)]#\
    else:        
        # Look at the end of phase trials
        phase1_df0 = pd.concat([g[g['train_day']>=g['train_day'].max()*frac_max] \
                            if ph==1 else g \
                            for ph, g in standard_df.groupby(['phase'])])
        phase1_df = phase1_df0[(phase1_df0['phase']<=max_train_phase)
                        & (phase1_df0['phase']>=min_train_phase)].copy()

    phase1_counts = get_counts_from_session_df(phase1_df, 
                        group_all_conds=group_all_conds)

    animalids = phase1_counts['animalid'].unique()
    mean_acc = phase1_counts.groupby(['animalid']).mean().reset_index()

    pass_animals = mean_acc[mean_acc['accuracy']>=criterion]['animalid'].unique()
    
    missing = [k for k in animalids if k not in pass_animals]

    print("%i of %i animals pass min crit. %.2f accuracy on training. Fail animals %i:\n  " \
          % (len(pass_animals), len(animalids), criterion, len(missing)), missing)

    return pass_animals, missing

