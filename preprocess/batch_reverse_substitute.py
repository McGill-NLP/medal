import argparse
import string
import os
import time
from multiprocessing import Pool
from itertools import repeat
from scipy.stats import bernoulli
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Reverse substitution')
    parser.add_argument('-i', '--input_file', required=True, help='path to file containing text')
    parser.add_argument('-n', '--num_of_rows', type=int, help='number of rows to be read from input text file')
    parser.add_argument('-a', '--abbreviation_table', required=True, help='path to abbreviation table')
    parser.add_argument('-o', '--output_file', required=True, help='path to output file')
    parser.add_argument('-p', '--probability', default=0.15, type=float, help='probability of reverse substitution')
    parser.add_argument('--max_cores', default=4, type=int, help='max number of cpu cores')
    return parser.parse_args()

def clean_expansion(text):
    lower = text.lower()
    no_punc = lower.translate(str.maketrans("", "", string.punctuation))
    no_white = " ".join(no_punc.split())
    no_num = no_white.translate(str.maketrans("", "", string.digits))
    return no_num

def reverse_substitute(input_tuple):
    text, expansions_to_ab, p = input_tuple
    expansions = [expans for expans in expansions_to_ab.keys() if expans in text]
    if not expansions:  # no expansion found
        return None
    wcs = [len(expansion.split()) for expansion in expansions]
    first_words = [expansion.split()[0] for expansion in expansions]
    max_wc = np.max(wcs)
    labels = {}
    current_text = text.split()
    for idx, word in enumerate(current_text):
        if word not in first_words: # the word can be the first word
            continue
        if np.sum([' '.join(current_text[idx:idx + wc]) in expansions for wc in range(max_wc, 0, -1)]) == 0: # no expansion detected here
            continue
        current_wcs = [wc for wc in range(max_wc, 0, -1) if ' '.join(current_text[idx:idx + wc]) in expansions] # word counts for all possible expansions
        for wc in current_wcs: # starting with longest
            if wc > len(current_text) - idx: # if remaining text is not long enough
                continue
            if bernoulli.rvs(p, size=1):
                labels[idx] = ' '.join(current_text[idx:idx + wc]) # record location
                current_text[idx] = expansions_to_ab[' '.join(current_text[idx:idx + wc])] # replace with abb
                for _ in range(1, wc):
                    del current_text[idx + 1] # delete remaining expansion
                break # don't search for shorter expansions
    if not labels:
        return None
    return ' '.join(current_text), labels

if __name__ == "__main__":
    args = parse_args()
    print('[' + time.ctime() + ']', 'args parsed')

    if args.num_of_rows:
        text_df = pd.read_csv(args.input_file, header=None, nrows=args.num_of_rows, engine='c')
        print('[' + time.ctime() + ']', args.num_of_rows, 'rows read from', args.input_file)
    else:
        text_df = pd.read_csv(args.input_file, header=None, engine='c')
        print('[' + time.ctime() + ']', 'all rows read from', args.input_file)
    text_df.columns = ['TEXT']

    adam_valid_df = pd.read_csv(args.abbreviation_table, sep='\t')
    print('[' + time.ctime() + ']', 'read abbreviation table from', args.abbreviation_table)

    ab_to_expansions = {ab: adam_valid_df['EXPANSION'][adam_valid_df['PREFERRED_AB'] == ab].to_list() for ab in adam_valid_df['PREFERRED_AB'].unique()}
    expansions_to_ab = {row['EXPANSION']: row['PREFERRED_AB'] for _, row in adam_valid_df.iterrows()}
    print('[' + time.ctime() + ']', 'abbreviation-expansion mapping constructed')

    with Pool(processes=args.max_cores) as pool:
        output_tuples = pool.map(reverse_substitute, zip(text_df['TEXT'], repeat(expansions_to_ab), repeat(args.probability)))
    print('[' + time.ctime() + ']', 'reverse substitute complete')

    abs_id = 0
    abs_ids = []
    rs_texts = []
    locations = []
    true_expansions = []
    print('[' + time.ctime() + ']', 'start parsing outputs')
    for output in tqdm(output_tuples):
        if not output:
            continue
        labels = output[1]
        labels = list(sorted(labels.items()))
        abs_id += 1
        for label in labels:
            abs_ids.append(abs_id)
            rs_texts.append(output[0])
            locations.append(label[0])
            true_expansions.append(label[1])
    output_df = pd.DataFrame({'ABSTRACT_ID': abs_ids, 'TEXT': rs_texts, 'LOCATION': locations, 'LABEL': true_expansions})
    output_df.to_csv(args.output_file, index=None)
    print('[' + time.ctime() + ']', 'output saved to', args.output_file)