#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
#import commands
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

parser.add_argument('-V', '--area', dest='visual_area', action='store', default=None, help='Visual area to process (default, all)')
parser.add_argument('-k', '--datakeys', nargs='*', dest='included_datakeys', action='append', help='Use like: -k DKEY DKEY DKEY')
#parser.add_argument('-i', '--animalids', nargs='*', dest='animalids', action='append', help='Use like: -k DKEY DKEY DKEY')


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


def load_metadata(experiment, visual_area=None,
                  aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
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


# -----------------------------------------------------------------
# ARGS
# -----------------------------------------------------------------

ROOTDIR = '/n/coxfs01/2p-data'
experiment = args.experiment_type
email = args.email

visual_area = None if args.visual_area in ['None', None] else args.visual_area
traceid = args.traceid

dsets = load_metadata(experiment, visual_area=visual_area)
included_datakeys = args.included_datakeys
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

#####################################################################
#                          find XID files                           #
#####################################################################
# Note: the syntax a+=(b) adds b to the array a

# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]

if len(included_datakeys)==1:
    logdir = 'LOG_roc_%s_%s' % (experiment, included_datakeys[0]) 
else:
    logdir = 'LOG_roc_%s_%s' % (experiment, str(visual_area)) 


if not os.path.exists(logdir):
    os.mkdir(logdir)

# Remove old logs
old_logs = glob.glob(os.path.join(logdir, '*.err'))
old_logs.extend(glob.glob(os.path.join(logdir, '*.out')))
old_logs.extend(glob.glob(os.path.join(logdir, '*.txt')))
for r in old_logs:
    os.remove(r)

# Open log lfile
sys.stdout = open('%s/INFO_%s_%s.txt' % (logdir, piper, experiment), 'w')


################################################################################
#                               run the pipeline                               #
################################################################################
cmd_str = '/n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/slurm/bootstrap_roc.sbatch'
jobids=[]

# Run it
for datakey, g in dsets.groupby(['datakey']):
    mtag = '%s_%s_%s' % (experiment, datakey, visual_area) 
    #
    cmd = "sbatch --job-name=ROC.{procid}.{mtag} \
            -o '{logdir}/ROC.{procid}.{mtag}.out' \
            -e '{logdir}/ROC.{procid}.{mtag}.err' \
            {cmd} {datakey} {exp} {traceid}".format(
        procid=piper, mtag=mtag, logdir=logdir, cmd=cmd_str,
        datakey=datakey, exp=experiment, traceid=traceid) 
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




