#!/bin/bash
# process_rfs.sbatch
#
#SBATCH -J procrf # A single job name for the array
#SBATCH -p shared # run on cox gpu to use correct env 
#SBATCH -n 4 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-00:30 # Running time of 3 hours
#SBATCH --mem 8192 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o rfs_%A_%a.out # Standard output
#SBATCH -e rfs_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
#module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/rat2p

DATAKEY="$1"
EXP="$2"
TRACEID="$3"
#SPHERICAL="$4"
NEUROPIL="$4"
REDO="$5"

# run processing on raw data
echo "datakey: ${DATAKEY}"
echo "RF type: ${EXP}"
#echo "SPHERICAL: ${SPHERICAL}"
echo "IS NEUROPIL: ${NEUROPIL}"
echo "REDOFITS: ${REDO}"

#
echo "NO spherical-correction"
if [[ ${NEUROPIL} == "True" ]]; then
    echo "Np..."
    python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/receptive_fields/fit_rfs.py -i $DATAKEY -E $EXP -t $TRACEID -M dff -p 0.5 -n 4 --do-fits --neuropil 
else 
    if [[ ${REDO} == "True" ]]; then
        echo "Re-doing fits"
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/receptive_fields/fit_rfs.py -i $DATAKEY -E $EXP -t $TRACEID -M dff -p 0.5 -n 4 --do-fits --do-eval --all-new
    else
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/receptive_fields/fit_rfs.py -i $DATAKEY -E $EXP -t $TRACEID -M dff -p 0.5 -n 4 --do-eval --all-new
    fi
fi

