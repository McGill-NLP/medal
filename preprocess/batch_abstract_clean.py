import argparse
import string
import os
import time
from multiprocessing import Pool
from itertools import repeat
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Clean extracted abstracts in txt format')
    parser.add_argument('-i', '--input_directory', required=True, help='directory containing extracted abstracts')
    parser.add_argument('-s', '--stop_word_file', required=False, help='file containing stop words to ignore')
    parser.add_argument('-o', '--output_directory', required=True, help='output directory')
    parser.add_argument('--max_cores', default=4, type=int, help='max number of cpu cores')
    return parser.parse_args()

def clean_abstract(input_tuple):
    input_directory, file_name, output_directory, sws = input_tuple

    with open(os.path.join(args.input_directory, file_name), 'r') as file:
        raw_abstracts = file.readlines()
    if len(raw_abstracts) == 1:
        return
    abstracts = {int(raw_abstract.split(':')[0]): ':'.join(raw_abstract.split(':')[1:]).rstrip('\n') for raw_abstract in raw_abstracts}
    cleaned_abstracts = []
    
    for idx, text in abstracts.items():
        lower = text.lower()
        no_punc = lower.translate(str.maketrans("", "", string.punctuation))
        no_white = " ".join(no_punc.split())
        no_num = no_white.translate(str.maketrans("", "", string.digits))
        if sws:
            no_sw = " ".join([word for word in no_num.split() if (word not in sws) and (len(word) > 1)])
        else:
            no_sw = no_num
        cleaned_abstracts.append(":".join([str(idx), no_sw]))

    with open(os.path.join(output_directory, os.path.splitext(os.path.splitext(file_name)[0])[0] + '_cleaned.txt'), 'w') as file:
        file.writelines('\n'.join(cleaned_abstracts) + '\n')

if __name__ == "__main__":
    args = parse_args()

    all_filenames = os.listdir(args.input_directory)
    filenames = [filename for filename in all_filenames if os.path.splitext(filename)[1] == '.txt']
    print('[' + time.ctime() + ']', len(all_filenames), "files found in", args.input_directory)
    print('[' + time.ctime() + ']', len(filenames), "are txt files")

    if args.stop_word_file:
        with open(args.stop_word_file, 'r') as file:
            raw_sws = file.readlines()
        sws = [line.rstrip('\n') for line in raw_sws]
        print('[' + time.ctime() + ']', "stop words read")
    else:
        sws = None  # to maintain consistent function interface

    with Pool(processes=args.max_cores) as pool:
        pool.map(clean_abstract, zip(repeat(args.input_directory), filenames, repeat(args.output_directory), repeat(sws)))
    print('[' + time.ctime() + ']', len(filenames), "abstracts cleaned")