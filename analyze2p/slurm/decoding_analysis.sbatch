#!/bin/bash
# decoding_analysis.sbatch
#
#SBATCH -J dcode # A single job name for the array
#SBATCH -p cox #shared # run on cox gpu to use correct env 
#SBATCH -n 4 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-02:00 # Running time of 3 hours
#SBATCH --mem 8192 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o decode_%A_%a.out # Standard output
#SBATCH -e decode_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
module load Anaconda/5.0.1-fasrc01
# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/rat2p

#{TID} {EXP} {VA} {DKEY} {ANALYSIS} {TEST} {CORRS} {NCELLS}".format(

TRACEID="$1"
EXP="$2"
VAREA="$3"
DATAKEY="$4"
ANALYSIS="$5"
TEST="$6"
BREAK="$7"
NCELLS="$8"
MATCHRF="$9"

# run processing on raw data
echo "datakey: ${DATAKEY}"
echo "experiment: ${EXP}"
echo "ANALYSIS: ${ANALYSIS}"
echo "TEST: ${TEST}"
echo "MATCHRF: ${MATCHRF}"
#
if [[ ${BREAK} == "True" ]]; then
    echo "breaking corrs..."
    if [[ ${MATCHRF} == "True" ]]; then
        echo "matching RF size!"
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/decoding.py -V $VAREA -k $DATAKEY -E $EXP -t $TRACEID -d dff -X $ANALYSIS -T $TEST -R nstds -r 10. --epoch plushalf -C 1 -N 500 -n 4 -S $NCELLS --break --rf-size
    else
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/decoding.py -V $VAREA -k $DATAKEY -E $EXP -t $TRACEID -d dff -X $ANALYSIS -T $TEST -R nstds -r 10. --epoch plushalf -C 1 -N 500 -n 4 -S $NCELLS --break
    fi

else
    echo "Corrs intact..."
    if [[ ${MATCHRF} == "True" ]]; then
        echo "matching RF size!"
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/decoding.py -V $VAREA -k $DATAKEY -E $EXP -t $TRACEID -d dff -X $ANALYSIS -T $TEST -R nstds -r 10. --epoch plushalf -C 1 -N 500 -n 4 -S $NCELLS --rf-size
    else
        echo "No match"
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/decoding.py -V $VAREA -k $DATAKEY -E $EXP -t $TRACEID -d dff -X $ANALYSIS -T $TEST -R nstds -r 10. --epoch plushalf -C 1 -N 500 -n 4 -S $NCELLS
    fi
fi
