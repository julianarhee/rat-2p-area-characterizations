#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
import subprocess
import json
import glob
import pandas as pd
import _pickle as pkl
import numpy as np

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')

parser.add_argument('-E', '--exp', dest='experiment_type', action='store', default='gratings', help='Experiment type (e.g., gratings)')

parser.add_argument('-e', '--email', dest='email', action='store', default='rhee@g.harvard.edu', help='Email to send log files')
parser.add_argument('-t', '--traceid', dest='traceid', action='store', default='traces001', help='Traceid to use as reference for selecting retino analysis')

parser.add_argument('-R', '--resp-test', dest='responsive_test', action='store', default='nstds', help='Responsive test (default=nstds, options: ROC, nstds, None)')

parser.add_argument('-r', '--resp-thr', dest='responsive_thr', action='store', default=15, help='Responsive test (default=15)')


parser.add_argument('-v', '--area', dest='visual_area', action='store', default=None, help='Visual area to process (default, all)')
parser.add_argument('-k', '--datakeys', nargs='*', dest='included_datakeys', action='append', help='Use like: -k DKEY DKEY DKEY')
#parser.add_argument('-i', '--animalids', nargs='*', dest='animalids', action='append', help='Use like: -k DKEY DKEY DKEY')

parser.add_argument('--new', dest='redo_cell', action='store_true',
                default=False, help='Set flag to rerun all cells.')

args = parser.parse_args()


def info(info_str):
    sys.stdout.write("INFO:  %s \n" % info_str)
    sys.stdout.flush() 

def error(error_str):
    sys.stdout.write("ERR:  %s \n" % error_str)
    sys.stdout.flush() 

def fatal(error_str):
    sys.stdout.write("ERR: %s \n" % error_str)
    sys.stdout.flush()
    sys.exit(1)

def load_metadata(experiment, responsive_test='nstds', responsive_thr=10.,
                  rootdir='/n/coxfs01/2p-data', visual_area=None,
                  aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                  traceid='traces001'):
    from analyze2p.aggregate_datasets import aggregate_data_stats as aggr
    sdata = aggr.get_aggregate_info(traceid=traceid) #, fov_type=fov_type, state=state)
    sdata_exp = sdata[sdata['experiment']==experiment] 
      
    if visual_area is not None:
        sdata_exp = sdata_exp[sdata_exp['visual_area']==visual_area]
 
    return sdata_exp


ROOTDIR = '/n/coxfs01/2p-data'
experiment = args.experiment_type
email = args.email

visual_area = None if args.visual_area in ['None', None] else args.visual_area
traceid = args.traceid
responsive_test = args.responsive_test
responsive_thr = args.responsive_thr
if responsive_test == 'ROC':
    responsive_thr=0.05
redo_cell = args.redo_cell

# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]
logdir = 'LOG__osi_%s_%s' % (str(visual_area), experiment) 
if not os.path.exists(logdir):
    os.mkdir(logdir)

# Remove old logs
old_logs = glob.glob(os.path.join(logdir, '*.err'))
old_logs.extend(glob.glob(os.path.join(logdir, '*.out')))
old_logs.extend(glob.glob(os.path.join(logdir, '*.txt')))
for r in old_logs:
    os.remove(r)

#####################################################################
#                          find XID files                           #
#####################################################################
# Note: the syntax a+=(b) adds b to the array a
# Open log lfile
sys.stdout = open('%s/INFO_%s_%s.txt' % (logdir, piper, experiment), 'w')

def load_metadata(experiment, responsive_test='nstds', responsive_thr=10.,
                  rootdir='/n/coxfs01/2p-data', visual_area=None,
                  aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                  traceid='traces001'):
    #from analyze2p.aggregate_datasets import get_aggregate_info 
    #sdata = aggr.get_aggregate_info(traceid=traceid) #, fov_type=fov_type, state=state)
    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info_assigned.pkl')
    assert os.path.exists(sdata_fpath), "Path does not exist: %s" % sdata_fpath
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f, encoding='latin1')
    sdata_exp = sdata[sdata['experiment']==experiment]  

    if visual_area is not None:
        sdata_exp = sdata_exp[sdata_exp['visual_area']==visual_area]
 
    return sdata_exp

################################################################################
#                               run the pipeline                               #
################################################################################

# -----------------------------------------------------------------
# PER FOV 
# -----------------------------------------------------------------
jobids=[]
dsets = load_metadata(experiment, visual_area=visual_area)

included_datakeys = args.included_datakeys
#animalids = args.animalids

if included_datakeys is not None:
    included_datakeys=included_datakeys[0]
    print("dkeys:", included_datakeys)
    #['20190614_jc091_fov1', '20190602_jc091_fov1', '20190609_jc099_fov1']
    if len(included_datakeys) > 0:
        print(included_datakeys)
        dsets = dsets[dsets['datakey'].isin(included_datakeys)]

if len(dsets)==0:
    fatal("no fovs found.")
info("found %i [%s] datasets to process." % (len(dsets), experiment))

# Run it
for (datakey), g in dsets.groupby(['datakey']):
    mtag = '%s_%s_%s' % (experiment, datakey, visual_area) 
    #
    cmd = "sbatch --job-name={procid}.{mtag} \
            -o '{logdir}/{procid}.{mtag}.out' \
            -e '{logdir}/{procid}.{mtag}.err' \
    /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/slurm/bootstrap_osi.sbatch \
    {datakey} {traceid} {rtest} {rthr} {redo}".format(
        procid=piper, mtag=mtag, logdir=logdir,
        exp=experiment, datakey=datakey, traceid=traceid, 
        rtest=responsive_test, rthr=responsive_thr, redo=redo_cell)
    #
    status, joboutput = subprocess.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    jobids.append(jobnum)
    info("[%s]: %s" % (jobnum, mtag))

info("****done!****")

for jobdep in jobids:
    print(jobdep)
    cmd = "sbatch --job-name={JOBDEP}.checkstatus \
		-o 'log/checkstatus.{EXP}.{JOBDEP}.out' \
		-e 'log/checkstatus.{EXP}.{JOBDEP}.err' \
                  --depend=afternotok:{JOBDEP} \
                  /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/slurm/checkstatus.sbatch \
                  {JOBDEP} {EMAIL}".format(JOBDEP=jobdep, EMAIL=email, EXP=experiment)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = subprocess.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




