#!/bin/bash

# This script takes two input arguments (from_commit) and (to_commit)
# example: gen_diff 8923sdkjf2 HEAD
./git-latexdiff -b --main kernel_ep_uai2015.tex $1 $2 --math-markup=0

