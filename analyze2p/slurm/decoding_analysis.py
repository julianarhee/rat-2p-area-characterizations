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

parser.add_argument('-V', '--area', dest='visual_area', action='store', default=None, help='Visual area to process (default, all)')

parser.add_argument('-k', '--datakeys', nargs='*', dest='included_datakeys', action='append', help='Use like: -k DKEY DKEY DKEY')

parser.add_argument('-X', '--analysis', dest='analysis_type', action='store', default='by_fov', help='Analysis type, default: %s (opts: by_fov)')

parser.add_argument('-T', '--test', dest='test_type', action='store', default=None, help='Test type (opts: None, size_single, size_subset, morph)')

parser.add_argument('--break', dest='break_corrs', action='store_true', default=False, help='Break noise correlations')

parser.add_argument('-S', '--sample-sizes', nargs='+', dest='sample_sizes', default=[], help='Use like: -S 1 2 4')

parser.add_argument('--match-rfs', dest='match_rfs', action='store_true', default=False, help='Match RF size')

parser.add_argument('-O', '--overlap', dest='overlap_thr', action='store', default=None, help='Overlap thr')

parser.add_argument('-R', '--resp-test', dest='responsive_test', default='ROC', action='store', help='Responsive test (nstds or ROC), default=ROC')

parser.add_argument('--epoch', dest='trial_epoch', default='stimulus', action='store', help='Trial epoch to use for metrics (default: stimulus. Can be: plushalf, stimulus)')

parser.add_argument('-N', dest='n_iterations', default=500, action='store', help='Size of bootstrapped distribution  (default=500, NOTE: if morph in test_types, takes forever...)')

parser.add_argument('--shuffle-area', dest='shuffle_visual_area', default=False,action='store_true',  help='Shuffle visual area labels (analysis_type=BY_NCELLS)')

parser.add_argument('-C', '--class-name', dest='class_name', action='store', default='morphlevel',help='Name of class to decode (morphlevel, ori, sf)')

parser.add_argument('-v', '--var-name', dest='variation_name', action='store', default=None, help='Name of transform to split by (e.g., size)')



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
                traceid='traces001', visual_area=None):
    
    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info_assigned.pkl')
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f, encoding='latin1')
    sdata_exp = sdata[sdata.experiment==experiment]

    if visual_area is not None:
        sdata_exp = sdata_exp[sdata_exp['visual_area']==visual_area]
  
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

analysis_type = args.analysis_type
test_type = None if args.test_type in ['None', None] else args.test_type
break_corrs = args.break_corrs

sample_sizes = [int(i) for i in args.sample_sizes]
match_rfs= args.match_rfs
overlap_thr = None if args.overlap_thr in ['None', None] \
                else float(args.overlap_thr)

class_name = args.class_name
variation_name = None if args.variation_name in ['None', None] else str(args.variation_name)

responsive_test = args.responsive_test
trial_epoch = args.trial_epoch
n_iterations = int(args.n_iterations)
shuffle_visual_area = args.shuffle_visual_area


# Set up logging
# ---------------------------------------------------------------
# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]

test_str = 'default' if test_type is None else test_type
corr_str = 'break' if break_corrs else 'intact'
analysis_str = '%s_%s' % (analysis_type, test_str)
shuff_str = 'shuff_' if shuffle_visual_area else ''
if visual_area in [None, 'None']:
    logdir = '%sLOG__%s_%s_%s' % (shuff_str, experiment, analysis_str, corr_str)
else:
    logdir = '%sLOG__%s_%s_%s_%s' %  (shuff_str, experiment, analysis_str, str(visual_area), corr_str) 

#if match_rfs:
if match_rfs is False and overlap_thr is None:
    rf_str = '' #'noRF'
elif match_rfs is True and (overlap_thr is None or overlap_thr==0):
    rf_str = 'matchRF_'
elif match_rfs is True and overlap_thr>0:
    rf_str = 'matchRFoverlap%02d_' % (overlap_thr*10)
else:
    # match_rfs is False and overlap_thr is not None:
    rf_str = 'overlap%02d_' % (overlap_thr*10)

logdir = '%s%s' % (rf_str, logdir)

# ---------------------------------------------------------------
dsets = load_metadata(experiment, visual_area=visual_area)
included_datakeys = args.included_datakeys

if included_datakeys is not None:
    included_datakeys=included_datakeys[0]
    print("dkeys:", included_datakeys)
    #['20190614_jc091_fov1', '20190602_jc091_fov1', '20190609_jc099_fov1']
    if len(included_datakeys) > 0:
        print(included_datakeys)
        dsets = dsets[dsets['datakey'].isin(included_datakeys)]
    if len(included_datakeys)==1:
        new_logdir = '%s_%s' % (included_datakeys[0], logdir)
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
sys.stdout = open('%s/INFO_%s.txt' % (logdir, piper), 'w')


if len(dsets)==0:
    fatal("no fovs found.")
info("found %i [%s] datasets to process." % (len(dsets), experiment))

################################################################################
#                               run the pipeline                               #
################################################################################
basedir='/n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations'
if test_type=='morph_single':
    cmd_str = '%s/analyze2p/slurm/decoding_analysis_morph_single.sbatch' % basedir
elif test_type=='morph':
    cmd_str = '%s/analyze2p/slurm/decoding_analysis_morph.sbatch' % basedir
else:
    if shuffle_visual_area:
        cmd_str = '%s/analyze2p/slurm/decoding_analysis_shuffle_area.sbatch' % basedir
    else:
        cmd_str = '%s/analyze2p/slurm/decoding_analysis.sbatch' % basedir

# Run it
jobids = [] # {}
if analysis_type=='by_ncells':
    # -----------------------------------------------------------------
    # BY NCELLS
    # -----------------------------------------------------------------
    dk=None
    if len(sample_sizes)==0:
        if overlap_thr is not None:
            if overlap_thr>0:
                sample_sizes = [1, 2, 4, 8, 16, 32, 46]
            elif overlap_thr==0:
                sample_sizes0 = [1, 2, 4, 8, 16, 32, 64, 96, 120, 128] 
                #min_ncells = 94 if match_rfs else 141 # greater_than=False 
                min_ncells = 73 if match_rfs else 120
                sample_sizes = [k for k in sample_sizes0 if k<=min_ncells]
                if min_ncells > max(sample_sizes):
                    sample_sizes.append(min_ncells) 
        else:
            sample_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 254] #256] 
    visual_areas = ['V1', 'Lm', 'Li'] if visual_area is None else [visual_area]
    info("Testing %i areas: %s" % (len(visual_areas), str(visual_areas)))
    info("Testing %i sample size: %s" % (len(sample_sizes), str(sample_sizes)))
    for va in visual_areas:
        for n_cells_sample in sample_sizes: #ncells_test:
            mtag = '%s_%s_%s_%i' % (experiment, va, test_str, n_cells_sample) 
            print("MTAG: %s" % mtag)
            #
            cmd = "sbatch --job-name={PROCID}.dcode.{MTAG} \
                -o '{LOGDIR}/{PROCID}.{MTAG}.out' \
                -e '{LOGDIR}/{PROCID}.{MTAG}.err' \
                {CMD} {CLS} {EXP} {VA} {DKEY} {ANALYSIS} {TEST} {CORRS} {NCELLS} {MATCHRF} {OVERLAP} {RTEST} {EPOCH} {NITER} {VARN}".format(
                    PROCID=piper, MTAG=mtag, LOGDIR=logdir, CMD=cmd_str, 
                    CLS=class_name, EXP=experiment, VA=va, DKEY=dk, 
                    ANALYSIS=analysis_type, TEST=test_type, CORRS=break_corrs,
                    NCELLS=n_cells_sample, MATCHRF=match_rfs, OVERLAP=overlap_thr, RTEST=responsive_test, EPOCH=trial_epoch, NITER=n_iterations, VARN=variation_name)
            #info("Submitting PROCESSPID job with CMD:\n%s" % cmd)
            status, joboutput = subprocess.getstatusoutput(cmd)
            jobnum = joboutput.split(' ')[-1]
            jobids.append(jobnum)
            info("[%s]: %s" % (jobnum, mtag))

elif analysis_type=='by_fov':
    n_cells_sample=None
    for (va, dk), g in dsets.groupby(['visual_area', 'datakey']):

        mtag = '%s_%s_%s_%s' % (experiment, va, dk, test_str) 

        cmd = "sbatch --job-name={PROCID}.dcode.{MTAG} \
            -o '{LOGDIR}/{PROCID}.{MTAG}.out' \
            -e '{LOGDIR}/{PROCID}.{MTAG}.err' \
            {CMD} {CLS} {EXP} {VA} {DKEY} {ANALYSIS} {TEST} {CORRS} {NCELLS} {MATCHRF} {OVERLAP} {RTEST} {EPOCH} {NITER} {VARN}".format(
                    PROCID=piper, MTAG=mtag, LOGDIR=logdir, CMD=cmd_str, 
                    CLS=class_name, EXP=experiment, VA=va, DKEY=dk, 
                    ANALYSIS=analysis_type, TEST=test_type, CORRS=break_corrs,
                    NCELLS=n_cells_sample, MATCHRF=match_rfs, OVERLAP=overlap_thr, RTEST=responsive_test, EPOCH=trial_epoch, NITER=n_iterations, VARN=variation_name)
        #info("Submitting PROCESSPID job with CMD:\n%s" % cmd)
        status, joboutput = subprocess.getstatusoutput(cmd)
        jobnum = joboutput.split(' ')[-1]
        jobids.append(jobnum)
        info("[%s]: %s" % (jobnum, mtag))


#info("****done!****")

for jobdep in jobids:
    print(jobdep)
    cmd = "sbatch --job-name={JOBDEP}.checkstatus \
		-o 'log/checkstatus.dcode.{JOBDEP}.out' \
		-e 'log/checkstatus.dcode.{JOBDEP}.err' \
                  --depend=afternotok:{JOBDEP} \
                  {BDIR}/slurm/checkstatus.sbatch \
                  {JOBDEP} {EMAIL}".format(BDIR=basedir, JOBDEP=jobdep, EMAIL=email)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = subprocess.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




