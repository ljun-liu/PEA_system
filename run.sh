#!/bin/bash

PYTHON=python3

echo -e "\n $0 $@"

if [[ $# -lt 1 ]] || [[ $# -gt 2 ]]; then
    echo "Usage: $0 </path/to/kaldi/scoring-dir> [</path/to/PEA-results/dir>]"
    echo "Example: $0 data/scoring/system1/score_5  data/system1/PEA_analysis"
    echo
    exit 1;
fi

scoring_dir=$1
if [[ $# -eq 2 ]]; then
    results_dir=$2
else
    results_dir=$scoring_dir/PEA_analysis
    echo " PEA_Results_Dir is not provides, following is used ..."
    echo "$results_dir"
fi

export PYTHONPATH='.'
$PYTHON src/Main.py --scoring_dir $scoring_dir --results_dir $results_dir