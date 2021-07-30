#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
import subprocess
import json
import glob
import pandas as pd
import dill as pkl

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')

parser.add_argument('-A', '--fov', dest='fov_type', action='store', default='zoom2p0x', help='FOV type (e.g., zoom2p0x)')

parser.add_argument('-E', '--exp', dest='experiment_type', action='store', default='rfs', help='Experiment type (e.g., rfs')

parser.add_argument('-t', '--traceid', dest='traceid', action='store', default='traces001', help='traceid (default: traces001)')

parser.add_argument('-e', '--email', dest='email', action='store', default='rhee@g.harvard.edu', help='Email to send log files')

parser.add_argument('-S', '--sphere', dest='do_spherical_correction', action='store_true', help='Run RF fits and eval with spherical correction (saves to: fit-2dgaus_sphr-corr')

parser.add_argument('-v', '--area', dest='visual_area', action='store', default=None, help='Visual area to process (default, all)')

parser.add_argument('-k', '--datakeys', nargs='*', dest='included_datakeys', action='append', help='Use like: -k DKEY DKEY DKEY')

parser.add_argument('--fit', dest='redo_fits', action='store_true',
                default=False, help='Set flag to redo all fits.')

parser.add_argument('--old-data', dest='old_data_only', action='store_true',
                default=False, help='Set flag to only run datasets before 20190511.')


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


# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:8]
if not os.path.exists('log'):
    os.mkdir('log')

old_logs = glob.glob(os.path.join('log', '*.err'))
old_logs.extend(glob.glob(os.path.join('log', '*.out')))
for r in old_logs:
    os.remove(r)

#####################################################################
#                          find XID files                           #
#####################################################################
# Get PID(s) based on name
# Note: the syntax a+=(b) adds b to the array a
ROOTDIR = '/n/coxfs01/2p-data'
FOV = args.fov_type
EXP = args.experiment_type
traceid=args.traceid

visual_area = None if args.visual_area in ['None', None] else args.visual_area

email = args.email
do_spherical_correction = args.do_spherical_correction
sphr_str = 'sphere' if do_spherical_correction else ''
old_data_only = args.old_data_only
redo_fits = args.redo_fits

# Open log lfile
sys.stdout = open('log/info_%s_%s.txt' % (EXP, sphr_str), 'w')


def load_metadata(experiment, rootdir='/n/coxfs01/2p-data', 
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                traceid='traces001', response_type='dff', 
                visual_area=None,
                do_spherical_correction=False,old_data_only=False):
    
    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info_assigned.pkl')
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f, encoding='latin1')
    sdata = sdata[sdata.experiment==experiment]

    if old_data_only:
        sdata['session_int'] = sdata['session'].astype(int)
        sdata = sdata[sdata['session_int']<20190511]

    if visual_area is not None:
        sdata = sdata[sdata['visual_area']==visual_area]
 
    if do_spherical_correction:
        fit_desc = 'fit-2dgaus_%s_sphr' % response_type
    else:
        fit_desc = 'fit-2dgaus_%s-no-cutoff' % response_type        
   
    meta_list=[]
    for (dk, animalid, session, fov), g in \
            sdata.groupby(['datakey', 'animalid', 'session', 'fov']):
        #rfname = 'gratings' if int(session)<20190511 else e
        meta_list.append(dk)     
        existing_dirs = glob.glob(os.path.join(rootdir, animalid, session, 
                                fov, '*%s*' % experiment, 'traces/%s*' % traceid, 
                                'receptive_fields','%s' % fit_desc))
        for edir in existing_dirs:
            tmp_od = os.path.split(edir)[0]
            tmp_rt = os.path.split(edir)[-1]
            old_dir = os.path.join(tmp_od, '_%s' % tmp_rt)
            if os.path.exists(old_dir):
                continue
            else:
                os.rename(edir, old_dir)    
            info('renamed: %s' % old_dir)

    sdata_exp = sdata[sdata.datakey.isin(meta_list)]

    return sdata_exp


#meta_list = [('JC083', '20190508', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#             ('JC076', '20190420', 'FOV1_zoom2p0x', 'rfs', 'traces001')]
#             ('JC097', '20190616', 'FOV1_zoom2p0x', 'rfs', 'traces001')]

#[('JC097', '20190617', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#            ('JC084', '20190522', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#             ('JC120', '20191111', 'FOV1_zoom2p0x', 'rfs10', 'traces001')]

#meta_list = load_metadata(EXP, do_spherical_correction=do_spherical_correction,
#                old_data_only=old_data_only)
dsets = load_metadata(EXP, visual_area=visual_area,
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

if len(dsets)==0:
    fatal("no fovs found.")
info("found %i [%s] datasets to process." % (len(dsets), EXP))

################################################################################
#                               run the pipeline                               #
################################################################################
basedir='/n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations'
cmd_str = '%s/slurm/process_rfs.sbatch' % basedir

# Run it
jobids = [] # {}
for (datakey), g in dsets.groupby(['datakey']):
    print("REDO is now: %s" % str(redo_fits))
    mtag = '%s_%s_%s' % (EXP, datakey, visual_area) 

    cmd = "sbatch --job-name={PROCID}.rfs.{MTAG} \
            -o 'log/{PROCID}.rfs.{MTAG}.out' \
            -e 'log/{PROCID}.rfs.{MTAG}.err' \
            {COMMAND} {DATAKEY} {EXP} {TRACEID} {SPHERE} {REDO}".format(
                    PROCID=piper, MTAG=mtag, COMMAND=cmd_str, 
                    DATAKEY=datakey, EXP=EXP, TRACEID=traceid,
                    SPHERE=do_spherical_correction, REDO=redo_fits)
    info("Submitting PROCESSPID job with CMD:\n%s" % cmd)
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




