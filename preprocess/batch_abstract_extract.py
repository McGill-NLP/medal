from lxml import etree
import argparse
import os
import time
from multiprocessing import Pool
import tqdm
from itertools import repeat

def parse_args():
    parser = argparse.ArgumentParser(description='Parse xml.gz files and extract abstracts')
    parser.add_argument('input_directory', help='directory containing xml.gz files')
    parser.add_argument('output_directory', help='output directory')
    parser.add_argument('--max_cores', default=4, type=int, help='max number of cpu cores')
    return parser.parse_args()

def extract_abstract(input_tuple):
    input_directory, output_directory, filename = input_tuple
    tree = etree.parse(os.path.join(input_directory, filename))
    root = tree.getroot()
    abstracts = {}
    for idx, child in enumerate(root):
        text = None
        for value in child[0][3]:
            if value.tag == 'Abstract':
                text = value[0].text
                break
        if not text:
            continue
        text = " ".join(text.split())
        if text != "\n" and text:
            abstracts[str(idx)] = text
    output_abstracts = [':'.join([key, value]) for key, value in abstracts.items()]

    with open(os.path.join(output_directory, os.path.splitext(os.path.splitext(filename)[0])[0] + '.txt'), 'w') as file:
        file.writelines('\n'.join(output_abstracts) + '\n')

if __name__ == '__main__':
    args = parse_args()

    # get filenames
    all_filenames = os.listdir(args.input_directory)
    filenames = []
    for filename in all_filenames:
        if os.path.splitext(os.path.splitext(filename)[0])[1] == '.xml' and os.path.splitext(filename)[1] == '.gz':
            filenames.append(filename)
    print('[' + time.ctime() + ']', len(all_filenames), "files found in", args.input_directory)
    print('[' + time.ctime() + ']', len(filenames), "are xml.gz files")

    # parse and save
    with Pool(processes=args.max_cores) as pool:
        pool.map(extract_abstract, zip(repeat(args.input_directory), repeat(args.output_directory), filenames))
    print('[' + time.ctime() + ']', "abstracts parsed and saved")