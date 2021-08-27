#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:21:01 2021

@author: julianarhee
"""
import os
import glob
import sys
import optparse
import copy
import traceback


import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

import analyze2p.aggregate_datasets as aggr
import analyze2p.extraction.traces as traceutils
import analyze2p.gratings.utils as gutils



def bootstrap_params(dk, responsive_test='ROC', responsive_thr=0.05,
                      trial_epoch='stimulus', n_iterations=100,
                      param_list=['sf', 'size', 'speed'], offset='minsub',
                      n_bootstrap_iters=500, ci=95, at_best_ori=True,
                      response_type='dff', traceid='traces001', n_processes=1,
                      visual_areas=['V1', 'Lm', 'Li']):
    # Get fit dir
    ori_fit_desc = gutils.get_fit_desc(response_type=response_type,
                            responsive_test=responsive_test, 
                            responsive_thr=responsive_thr, 
                            n_bootstrap_iters=n_bootstrap_iters)
    traceid_dir = traceutils.get_traceid_dir(dk, 'gratings', traceid='traces001')
    fitdirs = glob.glob(os.path.join(traceid_dir, 'tuning*', ori_fit_desc))
    if len(fitdirs)==0:
        print("no fits: %s" % dk)
        return None
    # outfile
    fitdir=fitdirs[0]
    tmp_fov_outfile = os.path.join(fitdir, 'results_nonori_params.pkl')
    
    # Get cells in area
    sdata, cells0 = aggr.get_aggregate_info(visual_areas=visual_areas, 
                                            return_cells=True)
    curr_cells = cells0[(cells0.datakey==dk)]
    # Get stimuli
    sdf = aggr.get_stimuli(dk, experiment='gratings')
    # Get neuraldata
    ndf_wide = aggr.get_neuraldf(dk, experiment='gratings', traceid=traceid,
                       epoch=trial_epoch, response_type=response_type,
                       responsive_test=responsive_test, responsive_thr=responsive_thr)
    ndf_long = pd.melt(ndf_wide, id_vars=['config'], 
                  var_name='cell', value_name='response')
    all_cells = curr_cells['cell'].unique()
    ndf = ndf_long[ndf_long['cell'].isin(all_cells)]

    rdf_list = [g for rid, g in ndf.groupby('cell')]

    resdf = pool_bootstrap(rdf_list, sdf, n_iterations=n_iterations, at_best_ori=at_best_ori,
                        offset=offset, n_processes=n_processes)    
    # Get preference index for all cells in FOV that pass
#    ixs_ = ndf.groupby('cell').apply(bootstrap_nonori_index,\
#                             sdf, param_list=param_list, offset=offset,
#                             n_iterations=n_iterations, ci=ci)
#
#
    # Save iter results
    with open(tmp_fov_outfile, 'wb') as f:
        pkl.dump(ixs_, f, protocol=2)
    print("   saved: %s" % tmp_fov_outfile)
    
    return ixs_


def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_


def pool_bootstrap(rdf_list, sdf, n_iterations=100, ci=95, 
                    at_best_ori=True, offset='minsub', 
                    n_processes=1):

    results=[]
    resdf =  None
    terminating = mp.Event()        
    pool = mp.Pool(initializer=initializer, 
                                        initargs=(terminating, ), processes=n_processes)
    try:
        results = pool.map_async(partial(gutils.bootstrap_nonori_index, 
                                sdf=sdf, n_iterations=n_iterations, 
                                at_best_ori=at_best_ori, offset=offset), rdf_list).get() 

    except KeyboardInterrupt:
        print("**interupt")
        pool.terminate()
        print("***Terminating!")
    finally:
        pool.close()
        pool.join()
 
    if len(results) > 0:
        resdf = pd.concat(results, axis=0)
 
       
    return resdf


def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-k', '--datakey', action='store', dest='datakey', default='yyyymmdd_JCxx_fovX', 
                      help='datakey (YYYYMMDD_JCXX_fovX)')

    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    
    # Responsivity params: 
    choices_resptest = ('ROC','nstds', None)
    default_resptest = None 
    parser.add_option('-R', '--response-test', type='choice', choices=choices_resptest,
                      dest='responsive_test', default=default_resptest, 
                      help="Stat to get. Valid choices are %s. Default: %s" % (choices_resptest, str(default_resptest)))
    parser.add_option('-r', '--responsive-thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test threshold (default: p<0.05 for responsive_test=ROC)")

    # Tuning params:
    parser.add_option('-b', '--iter', action='store', dest='n_bootstrap_iters', 
                    default=1000, 
                     help="N bootstrap iterations (default: 1000)")

    parser.add_option('-d', '--response-type', action='store', dest='response_type', 
                    default='dff', 
                      help="Trial response measure to use for fits (default: dff)")
    parser.add_option('-n', '--n-processes', action='store', dest='n_processes', 
                    default=1, help="N processes (default: 1)")

    parser.add_option('-N', '--niter', action='store', dest='n_iterations', 
                    default=100, help="N processes (default: 100)")

    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="Create aggregate fit results file")

    parser.add_option('-E', '--epoch', action='store', dest='trial_epoch', 
                    default='stimulus', help='epoch of trial to use for fitting')

    (options, args) = parser.parse_args(options)

    return options




def main(options):
    opts = extract_options(options)
    rootdir = opts.rootdir

    dk = opts.datakey

    traceid = opts.traceid
    response_type = opts.response_type

    trial_epoch = opts.trial_epoch
    n_bootstrap_iters = int(opts.n_bootstrap_iters)

    responsive_test = opts.responsive_test
    responsive_thr = float(opts.responsive_thr)

    n_processes = int(opts.n_processes)
    create_new = opts.create_new

    n_iterations = int(opts.n_iterations)

    offset='minsub'
    ci=95
    at_best_ori=True

    bootstrap_params(dk, responsive_test=responsive_test, responsive_thr=responsive_thr,
                      trial_epoch=trial_epoch, n_iterations=n_iterations,
                      offset=offset,  ci=ci, at_best_ori=at_best_ori,
                      n_bootstrap_iters=n_bootstrap_iters, 
                      response_type=response_type, traceid=traceid, n_processes=n_processes)


    


if __name__ == '__main__':
    main(sys.argv[1:])
    


