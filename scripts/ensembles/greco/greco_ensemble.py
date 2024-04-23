"""greco selection of the best hypothesis with heruistic to bias more popular hypothesis"""
import os
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter

import torch
from greco_model import GRECO


def read_lines(fn, skip_strip=False):
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip() or skip_strip]


def write_lines(fn, lines, mode='w'):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])


sys_list = [
    'clc_copy__roberta_base', 'edit_scorer___xlnet_base_', 'gector___roberta_large_exp5_', 
            't5-11b__bea-upe-500-msl-128_', 
            # 'ul2___pretrained-batch-4-acc-2-lr-2e-05-up-500-wup-250-seed-6278_',
            'ul2___batch-4-acc-4-lr-5e-05-up-300-wup-250-seed-1748-databea_',
            'llama2-7b-chat__hf-batch-4-acc-2-lr-1e-05-up-2000-wup-400-seed-1666-gec_sota-clang8-100k_nucle-all_bea-all_fce-10k_troy-20k_tns_',
            # 'ens_m6_dec23'
            'llama-2-13b-chat__hf-batch-4-acc-2-lr-1e-05-up-2000-wup-400-seed-472-gec_sota-gec_sota-clang8-100k_nucle-all_bea-all_fce-10k_troy-20k_tns_',
            # 'ens_m6_jan3',
            'ens_m7_jan18',
            # 'gpt35__0shot-prompt1_',
            # 'gpt4__0shot-prompt1_',
            # 'gpt35__0shot-prompt-cot_',
            ]

def get_top_candidate(source, hyps, sys_names, model, voting_bias=False, m7_bias=0, verbose=0):
    if len(hyps) == 1:
        return hyps[0], sys_names[0]

    
    # score hypothesis
    scores = model.score(source, hyps)
    scores = scores.tolist()
    
    # get sys_weights based on majority voting
    # sys_proportions = [len(x.split("|")) for x in sys_names]
    # sys_proportions = [len(x.split("|")) + (1 if "orig" in x else 0) for x in sys_names]
    sys_proportions = [len(x.split("|")) + (m7_bias if "ens_m" in x else 0) for x in sys_names]
    wf = sum(sys_proportions)
    wn = wf/len(sys_proportions)
    mn = max(sys_proportions)
    # voting_bias = 1
    # sys_proportions = [(x )/wf for x in sys_proportions]
    # sys_proportions = [np.log(x + 1) for x in sys_proportions]
    # sys_proportions = [(x)/wn for x in sys_proportions]
    sys_proportions = [(x)/mn for x in sys_proportions]
    
    if voting_bias:
        scores = [s * sp for s, sp in zip(scores,sys_proportions)]
    # scores = [sp * sp for s, sp in zip(scores,sys_proportions)]

    sorted_hyps = sorted([(s, n, hyp) for hyp, n, s in zip(hyps, sys_names, scores)], reverse=True)
    if verbose > 0:
        print(f"\nOriginal: {source}")
        for i, (s, name, hyp) in enumerate(sorted_hyps):
            print(f"Top {i+ 1} hyp is {name} with {s} score: {hyp}")
        print('\n')
    top_hyp, top_sys = sorted_hyps[0][-1], sorted_hyps[0][-2]  
    return top_hyp, top_sys


def main(args):
    # load model
    device = "cuda"
    model = GRECO('microsoft/deberta-v3-large').to(device)
    model.load_state_dict(torch.load(args.path_to_model))
    print('Model initialization is finished.')
    # run model on dummy example
    source = "He go at school ."
    hyps = ["He goes to school .", "He goes at school .", "He go at school .", "He go .", ]
    sys_names = ['best', 'second|third', 'orig', 'broken']
    _, _ = get_top_candidate(source, hyps, sys_names, model, verbose=1)
        
    # load hypothesis
    sys_dict = {}
    for sys in sys_list:
        file_path = os.path.join(args.input_dir, f"{sys}___{args.evalset_name}.txt")
        lines = read_lines(file_path, skip_strip=True)
        print(f"Loaded {len(lines)} lines from {sys}")
        sys_dict[sys] = lines 
    

    orig_lines = read_lines(args.evalset_file)
    print(f"Loaded {len(orig_lines)} lines from {args.evalset_name}.")
    pred_lines = []
    sys_stats = Counter()
    for idx, orig_line in tqdm(enumerate(orig_lines)):
        source = orig_line

        sys_names_list = ['orig']
        # prepare hyps dict
        hyps_dict = {source: 'orig'}
        for sys_name, lines in sys_dict.items():
            hyp = lines[idx]
            if hyp not in hyps_dict:
                hyps_dict[hyp] = sys_name
            else:
                hyps_dict[hyp] = hyps_dict[hyp] + "|" + sys_name

        hyps, sys_names_list = [], []
        for hyp, sn in hyps_dict.items():
            hyps.append(hyp)
            sys_names_list.append(sn)
            
        # score hyps
        top_cand, top_sys = get_top_candidate(source, hyps, sys_names_list, model, verbose=0)
        pred_lines.append(top_cand)
        
        # get stats
        for sn in top_sys.split("|"):
            sys_stats[sn] += 1
            
        # for debuggin
        # if len(pred_lines) > 10:
        #     break
        
    print(f'\nSystem statistic:')
    print(sys_stats)
    sorted_sys_stats = sorted([(v, k) for k,v in sys_stats.items()], reverse=True)
    for (freq, sn) in sorted_sys_stats:
        print(sn, freq)
        
    # dump preds
    out_file = f"{args.out_greco_pred_prefix}__{args.evalset_name}.txt"
    write_lines(out_file, pred_lines)
    print(f"\nDumped {len(pred_lines)} lines.")
    print(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_model',)
                        # default='models/greco/models/checkpoint.bin')
    parser.add_argument('--input_dir',
                        help="The directory with all system predicitons.")
    parser.add_argument('--evalset_name',
                        default='nucle14_2a')
                        # default='bea-dev')
                        # default='bea-test')
    parser.add_argument('--evalset_file', 
                        default='data/evaluation_sets/nucle14-2a.txt')
                        # default='data/evaluation_sets/bea-dev.txt')
                        # default='data/evaluation_sets/bea-test.txt')
    parser.add_argument('--out_greco_pred_prefix',
                        default='/greco_ens_pred')
    args = parser.parse_args()
    main(args)

