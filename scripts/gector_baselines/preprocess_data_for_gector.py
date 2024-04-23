import argparse
import os
import csv
from tqdm import tqdm
from random import shuffle, seed

data_files = ['clang8_en.tokenized.csv', 'fce.train.tokenized.csv', 'wi_locness.train.tokenized.csv', 
              'troy-blogs.train.tokenized.csv', 'troy-1bw.train.tokenized.csv']


def is_pair_very_bad(src, tgt):
    # skip deletions 
    if not src.split() or not tgt.split():
        return True
    # skip too long pairs
    if len(src.split()) > 100 or len(tgt.split()) > 100:
        return True
    return False

def read_csv_file(csv_file, headers, return_dict=True):
    sents = []
    with open(csv_file, 'r', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader):
            if return_dict:
                sentence_dict = {}
                for key in headers:
                    sentence_dict[key] = row[key]
                sents.append(sentence_dict)
            else:
                sentence_list = []
                for key in headers:
                    sentence_list.append(row[key])
                sents.append(sentence_list)
    return sents


def write_lines(fn, lines, mode='w'):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])


def main(args):
    headers = ['src', 'tgt']
    all_lines = []
    for fname in data_files:
        input_csv = os.path.join(args.input_data_dir, fname)
        lines = read_csv_file(input_csv, headers=headers)
        pos_ratio = round(100*len([x for x in lines if x['src'] != x['tgt']]) / len(lines),1)
        print(f"Loaded {len(lines)} lines from {fname} with positives ratio {pos_ratio}%.")
        
        if args.downsample_troy is not None and 'troy' in fname:
            # downsample troy examples
            samle_n = int(len(lines) * args.downsample_troy)
            # shuffle lines
            seed(args.seed)
            shuffle(lines)
            sampled_lines = lines[:samle_n]
            print(f"Sampled {len(sampled_lines)} from {len(lines)} lines.")
            lines = sampled_lines[:]

        
        all_lines.extend(lines)
        
    # ensure all lines are ok
    all_lines = [x for x in all_lines if not is_pair_very_bad(x['src'], x['tgt'])]
        
    pos_ratio = round(100*len([x for x in all_lines if x['src'] != x['tgt']]) / len(all_lines),1)
    print(f"Loaded {len(all_lines)} lines overall with positives ratio {pos_ratio}%.")
    
    # shuffle sentences
    seed(args.seed)
    shuffle(all_lines)
    
    # split into train and dev
    total_dev_sents = min(int(args.dev_ratio * len(all_lines)), args.max_dev_lines)
    train_lines = all_lines[total_dev_sents:]
    dev_lines = all_lines[:total_dev_sents]
    print(f"Split data into {len(train_lines)} train lines and {len(dev_lines)} dev lines.")
    
    data_dict = {'train': train_lines, 'dev': dev_lines}
    
    # dump lines
    for partition, lines_dict in data_dict.items():
        for part in ['src', 'tgt']:
            lines = [x[part] for x in lines_dict]
            out_fname = f"{partition}_{part}"
            out_file_name = os.path.join(args.output_directory, out_fname)
            write_lines(out_file_name, lines)
            print(f"Dumped {len(lines)} into {out_fname}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir',
                        help="The directory with all training files in csv format with src/tgt.")
    parser.add_argument('--output_directory', 
                        help="The directory with output data.",
                        default='data/preprocessed_for_gector/')
    parser.add_argument('--seed', type=int, default=2042)
    parser.add_argument('--dev_ratio', type=float, default=0.02)
    parser.add_argument('--max_dev_lines', type=float, default=20000)
    parser.add_argument('--downsample_troy', type=float, 
                        default=0.2)
    args = parser.parse_args()
    main(args)