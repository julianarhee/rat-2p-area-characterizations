#!/bin/bash
# nontori_params.sbatch
#
#SBATCH -J osi # A single job name for the array
#SBATCH -p shared # run on cox gpu to use correct env 
#SBATCH -n 4 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-01:00 # Running time of 3 hours
#SBATCH --mem 8192 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o osi_%A_%a.out # Standard output
#SBATCH -e osi_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
#module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/rat2p #pipeline


DATAKEY="$1"
TRACEID="$2"
RTEST="$3"
RTHR="$4"
REDOCELL="$5"

echo "datakey: ${DATAKEY}"
echo "REDO: ${REDOCELL}"
#
if [[ ${REDOCELL} == "True" ]]; then
    echo "REDOING all cells"

    python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/gratings/nonori_params.py -t $TRACEID -R $RTEST -r $RTHR -k $DATAKEY -b 500 -N 100 -E stimulus --new -n 4

 else
    echo "NO redo"
    python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/gratings/nonori_params.py -t $TRACEID -R $RTEST -r $RTHR -k $DATAKEY -b 500 -N 100 -E stimulus -n 4

fi
 
 
#{EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY}".format(

