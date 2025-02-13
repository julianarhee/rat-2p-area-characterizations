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
try:
    import cPickle as pkl
except:
    import pickle as pkl
#from cPickle import PicklingError

import scipy.stats as spstats
import utils as util
#import process_datafiles as processd

import pprint

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def print_phase_lookup():
    phase_lookup = {0: 'always_reward',
                    1: 'default',
                    2: 'size',
                    3: 'depth_rotation',
                    4: 'cross',
                    5: 'size_and_depth_rotation',
                    6: 'depth_and_planar_rotation',
                    7: 'morph',
                    8: 'newstimuli',
                    9: 'fine_grained_size',
                    10: 'fine_grained_depth_rotation',
                    11: 'fine_grained_size_and_depth_rotation',
                    12: 'transparency',
                    13: 'clutter',
                    14: 'light_position',
                    15: 'x_rotation',
                    16: 'position',
                    17: 'punishcycle_long',
                    18: 'punishcycle_short',
                    19: 'no_min_RT',
                    -1: 'other'}
    pp.pprint(phase_lookup)

    return phase_lookup
 
def get_default_params(cohort, phase=None):
    # array(['', 'b', 'big', 'stimB', 'stimBlowerbound', 'stimClowerbound',
    #        'stimBupperbound', 'stimCupperbound', 'small', 'stimC', 'c',
    #        'stimCalwaysReward', 'alwaysReward', 'size', 'adeptrotl',
    #        'bdeptrotr', 'deptrotl', 'adeptrotr', 'asize', 'bdeptrotl',
    #        'background', 'deptrotr', 'balwaysReward', 'initial', 'deprotl',
    #        'backgroundalwaysReward', 'bsize', 'morphs', 'aalwaysReward',
    #        'empty', 'aasize', 'a', 'probeNewObjects', 'occluded', 'r', 'd',
    #        'transparency', 'nominrt'], dtype=object)

    default_depth_rotation = 0.
    default_planar_rotation = 0.

    if cohort in ['AK', 'AL', 'AM', 'AN', 'AO']:
        expected_sizes = np.linspace(15., 40., 11)
        if phase in [4, 5]:
            expected_drots = np.linspace(-60., 60., 9)
            default_size = 30.
        else:
            expected_drots = np.linspace(-60., 60., 25)
            default_size = 40.
        check_alwaysreward = True # Only these cohorts seem to follow AlwaysReward flag 

    elif cohort in ['AG', 'AJ']:
        expected_sizes = np.linspace(15., 40., 6)
        expected_drots = np.linspace(-60., 60., 9)
        default_size = 30
        check_alwaysreward = False

    expected_size_interval = np.diff(expected_sizes).mean()
    expected_drot_interval = np.diff(expected_drots).mean()

    defaults = {'size': default_size,
                'depth_rotation': default_depth_rotation,
                'planar_rotation':  default_planar_rotation,
                'expected_sizes': expected_sizes,
                'expected_depth_rotations': expected_drots,
                'standard_depth_rotations': np.linspace(-60., 60., 9),
                'fine_depth_rotations': np.linspace(-60., 60., 25),
                'check_alwaysreward': check_alwaysreward}
    
    return defaults

       

def get_phase_from_datafile(animalid, ameta, create_new=False):

    new_stim_descs = ['stimB', 'stimC', 'NewObjects']

    phase = -1
    (animalid, cohort, dfn, session, sfx), = ameta.values

    defaults = get_default_params(cohort)
    default_size = defaults['size']
    default_depth_rotation = defaults['depth_rotation']
    default_planar_rotation = defaults['planar_rotation']

    expected_sizes = defaults['expected_sizes']
    expected_drots = defaults['expected_depth_rotations']
    expected_size_interval = np.diff(expected_sizes).mean()
    expected_drot_interval = np.diff(expected_drots).mean()
    
    check_alwaysreward = defaults['check_alwaysreward']

    curr_trials=None
    curr_flags = None 
    curr_trials, curr_flags, metainfo = util.parse_mw_file(dfn, create_new=create_new)

    #print(session, len(curr_trials))
    if metainfo==-1 or curr_trials is None or len(curr_trials)==0:
        #exclude_ixs.extend(mgroup.index.tolist())
        return None #continue

    if 'N_PunishmentCycles' not in curr_flags.keys():
        # funky dfile: AO10_180623aasize_discard.. 
        return None

    sizes = sorted(np.unique([t['size'] for t in curr_trials]))
    drots = sorted(np.unique([t['depth_rotation'] for t in curr_trials]))
    prots = sorted(np.unique([t['rotation'] for t in curr_trials]))

    tested_size_interval = np.median(np.diff(sizes))
    tested_drot_interval = np.median(np.diff(drots))

    protocol = metainfo['protocol']
    experiment_folder = metainfo['experiment']

    alpha_values = []
    if any(['alpha_multiplier' in s.keys() for s in curr_trials]):
        alpha_values = np.unique([s['alpha_multiplier'] for s in curr_trials])
    
    light_positions = []
    if any(['light_position' in s.keys() for s in curr_trials]):
        light_positions = list(set([s['light_position'] for s in curr_trials]))

    x_rotations = []
    if any(['x_rotations' in s.keys() for s in curr_trials]):
        x_rotations = list(set([s['x_rotations'] for s in curr_trials]))

    positions = list(set([(s['pos_x'], s['pos_y']) for s in curr_trials]))

#     if session == 20180625:
#         break
#    print(curr_flags)  

    if check_alwaysreward and 1 in curr_flags['FlagAlwaysReward']:
        phase = 0
       
    elif max(curr_flags['N_PunishmentCycles']) != 5:
        if min(curr_flags['N_PunishmentCycles']) < 5:
            phase = 18 #'punishcycle_short'
        elif max(curr_flags['N_PunishmentCycles']) > 5:
            phase = 17

    elif 350000 not in curr_flags['TooFast_time']:
        phase = 19

    elif 'morph' in protocol or any(['morph' in s['name'] for s in curr_trials]):
        phase = 7

    elif 'newstim' in metainfo['experiment'] or any([descr in sfx for descr in new_stim_descs]):
        phase = 8

    elif len(alpha_values) > 1 or any([a != 1 for a in alpha_values]):
        phase = 12

    elif any(['background' in desc for desc in [sfx, metainfo['experiment']]]):
        phase = 13

    elif 'occluded' in sfx:
        phase = 13
        
    elif 'rotation in x axis' in metainfo['protocol'] or len(x_rotations)>1:
        phase = 15
        
    elif len(light_positions)> 1 and all([p is not None for p in light_positions]):
        phase = 14
        
    elif len(positions) > 1:
        phase = 16
        

    # ====== PHASE 1 ====================================
    elif len(sizes) == 1 and len(drots) == 1 and len(prots)==1 \
            and protocol in ['Initiate VIsual Pres Protocol', 'Initiate Visual Pres Protocol', '']:
        if (sizes[0]==default_size and drots[0]==default_depth_rotation and prots[0]==default_planar_rotation):
            phase = 1

    # ====== PHASE 2 ====================================
    elif len(sizes) > 1 and (len(drots)==1 and len(prots)==1) \
            and protocol in ['Staircase through shape parameters', '']:    
        
        if ( (expected_sizes==sizes) is False):
            if (tested_size_interval < expected_size_interval):
                phase = 9
            elif (tested_size_interval==expected_size_interval):
                if (len(sizes) > len(expected_sizes)-1):
                    # Probably just screwed up indexing
                    phase = 2
                elif all([d in expected_sizes for d in sizes]) : 
                    # only testing a subset of the expected values
                    phase = 2
        elif all(expected_sizes==sizes):
            phase = 2

    # ====== PHASE 3 ====================================
    elif protocol in ['Staircase through shape parameters', ''] \
            and (1 in curr_flags['FlagStaircaseDeptRotRight'] or 1 in curr_flags['FlagStaircaseDeptRotLeft']):
        if ( (expected_drots==drots) is False):
            if (len(drots) > 1 and (len(sizes)==1 and len(prots)==1)): 
                if (tested_drot_interval < expected_drot_interval):
                    # Fine-grained spacing
                    phase = 10
                elif (tested_drot_interval==expected_drot_interval) and all([d in expected_drots for d in drots]): 
                    # only testing a subset of the expected values
                    phase = 3
            elif len(sizes)==1 and len(drots)==1:
                # No other rotations tested yet
                phase = 3
        elif all(expected_drots==drots):
            phase = 3

    # ====== PHASE 4/5 ====================================
    elif (len(drots) > 1 and len(sizes) > 1 and len(prots) == 1) and protocol in ['Test all transformations', '']:
        off_cross_transforms = list(set([(t['size'], t['depth_rotation']) for t in curr_trials \
                                         if t['size']!=default_size \
                                         and t['depth_rotation']!=default_depth_rotation]))

        if ( (expected_drots==drots) is False) or ( (expected_sizes==sizes) is False):
            if (tested_size_interval < expected_size_interval) or (tested_drot_interval < expected_drot_interval):
                # Fine-grained size/rotation transformations
                phase = 11
            elif all([d in expected_drots for d in drots]) and all([s in expected_sizes for s in sizes]):
                phase = 4 if (len(off_cross_transforms)==0) else 5
        else:
            # STANDARD
            if all(drots==expected_drots) and all(sizes==expected_sizes):
                phase = 4 if (len(off_cross_transforms)==0) else 5

    # ====== PHASE 6 (in-plane) ====================================
    elif len(prots) > 1 and len(sizes)==1:
        if protocol == 'Test all transformations':
            phase = 6
            
    else:
        print("NO CLUE.")
        print("[%s, %s%s] unknown protocol: %s" % (animalid, session, sfx, protocol) )

        print("Size", sizes)
        print("D-rots", drots)
        print("P-rots", prots)

    print(animalid, '%i%s' % (session, sfx), phase)
    ameta['phase'] = [phase for _ in np.arange(0, len(ameta))]
    ameta['protocol'] = [protocol for _ in np.arange(0, len(ameta))]
    ameta['experiment'] = [experiment_folder for _ in np.arange(0, len(ameta))]

    return ameta


def assign_phase_by_animal(animalid, animal_meta, create_new=False):
    phasedata = []

    for (animalid, dfn), ameta in animal_meta.sort_values(by=['animalid', 'session']).groupby(['animalid', 'datasource']):
        currmeta = get_phase_from_datafile(animalid, ameta) #, create_new=create_new)

        if currmeta is not None:
            phasedata.append(currmeta)

    phasedata = pd.concat(phasedata, axis=0)

    return phasedata #, exclude_ixs


def assign_phase_by_cohort(cohort, metadata, paradigm='threeport', create_new=False, rootdir='/n/coxfs01/behavior-data'):
   
    #exclude_ixs = []
    phasedata = []
    animal_meta = metadata[metadata['cohort']== cohort]

    for (animalid, dfn), ameta in animal_meta.sort_values(by=['animalid', 'session']).groupby(['animalid', 'datasource']):
        currmeta = get_phase_from_datafile(animalid, ameta) #, create_new=create_new)

        if currmeta is not None:
            phasedata.append(currmeta)

    phasedata = pd.concat(phasedata, axis=0)

    return phasedata #, exclude_ixs

    
def get_phase_data(cohort, paradigm='threeport', create_new=False, verbose=False,
        rootdir='/n/coxfs01/behavior-data'):
    metadata = util.get_metadata(paradigm, rootdir=rootdir, filtered=False, create_meta=False)

    #### Load phase info for cohort
    processed_dir = os.path.join(rootdir, paradigm, 'processed')
    phase_dfile = os.path.join(processed_dir, 'meta', 'phases_%s.pkl' % cohort)
    if os.path.exists(phase_dfile) and create_new is False:
        print("... loading phase data...")
        print(phase_dfile)
        with open(phase_dfile, 'rb') as f:
            phasedata = pkl.load(f, encoding='latin1')
    else:
        phasedata = assign_phase_by_cohort(cohort, metadata, paradigm=paradigm, rootdir=rootdir)
        with open(phase_dfile, 'wb') as f:
            pkl.dump(phasedata, f, protocol=pkl.HIGHEST_PROTOCOL)

    if verbose:
        print(phasedata['phase'].unique())
        print(phasedata.groupby(['phase']).count())
    
    return phasedata

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/behavior-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/behavior-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default=None, 
                      help='Animal ID')
    parser.add_option('-c', '--cohort', action='store', dest='cohort', default=None, 
                      help='cohort, e.g, AL)')
    parser.add_option('-p', '--paradigm', action='store', dest='paradigm', default='threeport', 
            help='paradigm used (default:  threeport')

    parser.add_option('-n', '--nproc', action='store', dest='n_processes', default=1, 
            help='N processes to use (default:  1)')
    
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
            help='flag to assign phase info anew')
    
    (options, args) = parser.parse_args(options)

    return options


def main(options):

    opts = extract_options(options)
    rootdir = opts.rootdir
    paradigm = opts.paradigm
    animalid = opts.animalid
    cohort = opts.cohort
    create_new = opts.create_new
    
    phasedata = get_phase_data(cohort, paradigm=paradigm, create_new=create_new, rootdir=rootdir)

if __name__ == '__main__':
    main(sys.argv[1:])
                  
