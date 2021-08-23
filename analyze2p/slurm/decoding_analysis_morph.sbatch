#!/bin/bash
# decoding_analysis_morph.sbatch
#
#SBATCH -J dcode # A single job name for the array
#SBATCH -p shared #shared # run on cox gpu to use correct env 
#SBATCH -n 8 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-04:00 # Running time of 3 hours
#SBATCH --mem 16384 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
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
#{TID} {EXP} {VA} {DKEY} {ANALYSIS} {TEST} {CORRS} {NCELLS} {MATCHRF} {OVERLAP}".format(

CLASSNAME="$1"
EXP="$2"
VAREA="$3"
DATAKEY="$4"
ANALYSIS="$5"
TEST="$6"
BREAK="$7"
NCELLS="$8"
MATCHRF="$9"
OVERLAP="${10}"
RESPTEST="${11}"
EPOCH="${12}"

# run processing on raw data
echo "CLASS: ${CLASSNAME}"
echo "datakey: ${DATAKEY}"
echo "experiment: ${EXP}"
echo "ANALYSIS: ${ANALYSIS}"
echo "TEST: ${TEST}"
echo "MATCHRF: ${MATCHRF}"
echo "OVERLAP: ${OVERLAP}"
echo "RTEST: ${RESPTEST}"
#
if [[ ${BREAK} == "True" ]]; then
    echo "breaking corrs..."
    if [[ ${MATCHRF} == "True" ]]; then
        echo "matching RF size!"
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/decoding.py -V $VAREA -k $DATAKEY -E $EXP  -d dff -X $ANALYSIS -T $TEST -R $RESPTEST --epoch $EPOCH -C 1 -N 500 -S $NCELLS -O $OVERLAP --break --match-rfs --class-name $CLASSNAME -n 8
    else
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/decoding.py -V $VAREA -k $DATAKEY -E $EXP -d dff -X $ANALYSIS -T $TEST -R $RESPTEST --epoch $EPOCH -C 1 -N 500 -S $NCELLS -O $OVERLAP --break --class-name $CLASSNAME -n 8
    fi

else
    echo "Corrs intact..."
    if [[ ${MATCHRF} == "True" ]]; then
        echo "matching RF size!"
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/decoding.py -V $VAREA -k $DATAKEY -E $EXP -d dff -X $ANALYSIS -T $TEST -R $RESPTEST --epoch $EPOCH -C 1 -N 500 -S $NCELLS -O $OVERLAP --match-rfs --class-name $CLASSNAME -n 8
    else
        echo "No match"
        python /n/coxfs01/2p-pipeline/repos/rat-2p-area-characterizations/analyze2p/decoding.py -V $VAREA -k $DATAKEY -E $EXP -d dff -X $ANALYSIS -T $TEST -R $RESPTEST --epoch $EPOCH -C 1 -N 500 -S $NCELLS -O $OVERLAP --class-name $CLASSNAME -n 8
    fi
fi
