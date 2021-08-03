#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
import subprocess
import json
import glob
import shutil
import pandas as pd
import _pickle as pkl

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')

parser.add_argument('-A', '--fov', dest='fov_type', action='store', default='zoom2p0x', help='FOV type (e.g., zoom2p0x)')

parser.add_argument('-E', '--exp', dest='experiment_type', action='store', default='rfs', help='Experiment type (e.g., rfs')

parser.add_argument('-t', '--traceid', dest='traceid', action='store', default='traces001', help='traceid (default: traces001)')

parser.add_argument('-e', '--email', dest='email', action='store', default='rhee@g.harvard.edu', help='Email to send log files')

parser.add_argument('-v', '--area', dest='visual_area', action='store', default=None, help='Visual area to process (default, all)')

parser.add_argument('-k', '--datakeys', nargs='*', dest='included_datakeys', action='append', help='Use like: -k DKEY DKEY DKEY')

parser.add_argument('--old-data', dest='old_data_only', action='store_true',
                default=False, help='Set flag to only run datasets before 20190511.')


parser.add_argument('--sphere', dest='do_spherical_correction', action='store_true', help='Run RF fits and eval with spherical correction (saves to: fit-2dgaus_sphr-corr')

parser.add_argument('--fit', dest='redo_fits', action='store_true',
                default=False, help='Set flag to redo all fits.')


parser.add_argument('--neuropil', dest='is_neuropil', action='store_true',
     help='Run fits on NEUROPIL traces')


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

def load_metadata(experiment, rootdir='/n/coxfs01/2p-data', 
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                traceid='traces001', response_type='dff', 
                visual_area=None,
                do_spherical_correction=False,old_data_only=False):
    
    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info_assigned.pkl')
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f, encoding='latin1')
    sdata_exp = sdata[sdata.experiment==experiment]

    if old_data_only:
        sdata_exp['session_int'] = sdata_exp['session'].astype(int)
        sdata_exp = sdata_exp[sdata_exp['session_int']<20190511]

    if visual_area is not None:
        sdata_exp = sdata_exp[sdata_exp['visual_area']==visual_area]
 
    if do_spherical_correction:
        fit_desc = 'fit-2dgaus_%s_sphr' % response_type
    else:
        fit_desc = 'fit-2dgaus_%s-no-cutoff' % response_type        
   
    return sdata_exp


#####################################################################
#                          find XID files                           #
#####################################################################
# Get PID(s) based on name
# Note: the syntax a+=(b) adds b to the array a
ROOTDIR = '/n/coxfs01/2p-data'
FOV = args.fov_type
experiment = args.experiment_type
traceid=args.traceid

visual_area = None if args.visual_area in ['None', None] else args.visual_area

email = args.email
do_spherical_correction = args.do_spherical_correction
rf_correction = 'sphr' if do_spherical_correction else 'reg'
old_data_only = args.old_data_only
redo_fits = args.redo_fits
is_neuropil = args.is_neuropil

# Set up logging
# ---------------------------------------------------------------
# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]

np_str = '_neuropil' if is_neuropil else ''

if visual_area in [None, 'None']:
    logdir = 'LOG__%s%s' % (experiment, np_str)
else:
    logdir = 'LOG__%s%s_%s' % (experiment, np_str, str(visual_area) ) 

# ---------------------------------------------------------------
dsets = load_metadata(experiment, visual_area=visual_area,
                    do_spherical_correction=do_spherical_correction, 
                    old_data_only=old_data_only)
included_datakeys = args.included_datakeys

if included_datakeys is not None:
    included_datakeys=included_datakeys[0]
    print("dkeys:", included_datakeys)
    #['20190614_jc091_fov1', '20190602_jc091_fov1', '20190609_jc099_fov1']
    if len(included_datakeys) > 0:
        print(included_datakeys)
        dsets = dsets[dsets['datakey'].isin(included_datakeys)]
    if len(included_datakeys)==1:
        new_logdir = '%s_%s' % (logdir, included_datakeys[0])
        if not os.path.exists(new_logdir):
            os.makedirs(new_logdir)
        logdir = new_logdir


if not os.path.exists(logdir):
    os.mkdir(logdir)
# Remove old logs
old_logs = glob.glob(os.path.join(logdir, '*.err'))
old_logs.extend(glob.glob(os.path.join(logdir, '*.out')))
old_logs.extend(glob.glob(os.path.join(logdir, '*.txt')))
for r in old_logs:
    os.remove(r)

# Open log lfile
sys.stdout = open('%s/INFO_%s_%s_%s.txt' % (logdir, piper, experiment, rf_correction), 'w')


if len(dsets)==0:
    fatal("no fovs found.")
info("found %i [%s] datasets to process." % (len(dsets), experiment))

################################################################################
#                               run the pipeline                               #
################################################################################
basedir='/n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations'
cmd_str = '%s/analyze2p/slurm/process_rfs.sbatch' % basedir

# Run it
jobids = [] # {}
for (datakey), g in dsets.groupby(['datakey']):
    print("REDO is now: %s" % str(redo_fits))
    mtag = '%s_%s_%s' % (experiment, datakey, visual_area) 

    cmd = "sbatch --job-name={PROCID}.rfs.{MTAG} \
            -o '{LOGDIR}/{PROCID}.{MTAG}.out' \
            -e '{LOGDIR}/{PROCID}.{MTAG}.err' \
            {COMMAND} {DATAKEY} {EXP} {TRACEID} {SPHERE} {NEUROPIL} {REDO}".format(
                PROCID=piper, MTAG=mtag, LOGDIR=logdir, COMMAND=cmd_str, 
                DATAKEY=datakey, EXP=experiment, TRACEID=traceid,
                SPHERE=do_spherical_correction, NEUROPIL=is_neuropil, REDO=redo_fits)
    #info("Submitting PROCESSPID job with CMD:\n%s" % cmd)
    status, joboutput = subprocess.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    jobids.append(jobnum)
    info("[%s]: %s" % (jobnum, mtag))


#info("****done!****")

for jobdep in jobids:
    print(jobdep)
    cmd = "sbatch --job-name={JOBDEP}.checkstatus \
		-o 'log/checkstatus.rfs.{JOBDEP}.out' \
		-e 'log/checkstatus.rfs.{JOBDEP}.err' \
                  --depend=afternotok:{JOBDEP} \
                  {BDIR}/slurm/checkstatus.sbatch \
                  {JOBDEP} {EMAIL}".format(BDIR=basedir, JOBDEP=jobdep, EMAIL=email)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = subprocess.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




