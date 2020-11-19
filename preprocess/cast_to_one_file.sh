#!/bin/bash

filename=/home/mcb/users/zwen8/thesis/data/cleaned_abstracts_sws/$1

# echo $filename

cat $filename | awk -F ':' '{$1=""; print $0}'