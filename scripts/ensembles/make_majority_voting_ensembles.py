import argparse
import os

from tqdm import tqdm

from annotated_text_utils import AnnotatedTokens, AnnotatedText, align, OnOverlap


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


def is_anns_same(ann1, ann2, check_meta=False):
    s1, e1 = ann1.start, ann1.end
    p1 = ann1.top_suggestion
    s2, e2 = ann2.start, ann2.end
    p2 = ann2.top_suggestion
    if s1 == s2 and e1 == e2:
        if p1 == p2:
            if not check_meta:
                return True
            else:
                et1 = ann1.meta.get('error_type')
                et2 = ann2.meta.get('error_type')
                if et1 == et2:
                    return True
    return False


def is_ann_present_in_ann_list(ann, ann_list, check_meta=False):
    for ann2 in ann_list:
        if is_anns_same(ann, ann2, check_meta):
            return True
    return False


def get_majority_based_combination(system_preds_dict, min_relative_threshold=None):
    ann_tokens_list = []
    for sname, ann_tokens in system_preds_dict.items():
        if sname == 'original':
            continue
        ann_tokens_list.append(ann_tokens)

    comb_ann_tokens = AnnotatedTokens(AnnotatedText(system_preds_dict['original']))
    ann_lists = [x.get_annotations() for x in ann_tokens_list]

    ann_lists_with_counts = []
    for i, ann_list1 in enumerate(ann_lists):
        for ann_idx, ann in enumerate(ann_list1):
            total_counts = 1
            for j, ann_list2 in enumerate(ann_lists):
                if j != i and is_ann_present_in_ann_list(ann, ann_list2):
                    total_counts += 1
            ann_lists_with_counts.append((total_counts, 1/(i+1), 1/(ann_idx+1), ann))

    ann_lists_with_counts = sorted(ann_lists_with_counts, reverse=True)
    
    # filter infrequent corrections
    if min_relative_threshold is not None and min_relative_threshold > 0:
        ann_lists_with_counts = [x for x in ann_lists_with_counts if x[0] > min_relative_threshold]
    

    for freq, idx1, idx2, ann in ann_lists_with_counts:
        comb_ann_tokens.annotate(start=ann.start, end=ann.end,
                                correct_value=ann.top_suggestion,
                                meta=ann.meta,
                                on_overlap=OnOverlap.SAVE_OLD)

    # return combined annotated text
    return comb_ann_tokens


def main(args):
    # load an original sentences
    evalset_name = os.path.basename(args.evalset_path).rstrip('.txt')
    if 'nucle14' in evalset_name:
        evalset_name = 'nucle14'
    orig_lines = read_lines(args.evalset_path)
    print(f"Loaded {len(orig_lines)} original lines from {evalset_name}")
    
    # prepare list of systems
    list_of_system_preds_files = sorted([x for x in os.listdir(args.input_dir) if evalset_name in x])
    sys_file_dict = {}
    for i, fname in enumerate(list_of_system_preds_files):
        [sys_name, _] = fname.rstrip('.txt').split("__")
        print(f"System #{i + 1}. System name: {sys_name}.")
        file_path = os.path.join(args.input_dir, fname)
        sys_file_dict[sys_name] = file_path
        
    # exclude some systems for ablation study
    systems_to_exclude = [
        # single systems  
        # 'CTC-Copy',
        # 'GECToR-2024',
        # 'T5-11B', 
        # 'UL2-20B', 
        # 'EditScorer', 
        # 'Chat-LLaMa-2-13B-FT', 
        # 'Chat-LLaMa-2-7B-FT',
        
        # ensemle systems
        # 'ens_m3',
        # 'ens_m7',
        # 'ens_greco_on_m7',
        # 'gpt_rank_on_clust3, 
        # 'aggr_of_gpt_rank_on_clust3_and_m7',
        # 'aggr_of_gpt_rank_on_clust3_and_greco_on_m7'
        # 'ens_m8_with_greco',
        # 'ens_m9_with_greco_and_gpt',
     ]
    
    
    for sname in systems_to_exclude:
        if sname in sys_file_dict:
            del sys_file_dict[sname]

    # all systems has equal weight
    system_weights_dict = {sname:1 for sname in sys_file_dict.keys()}
    print(system_weights_dict)
    
    # boost weight for some systems
    # uncomment this line to to get ens_m8_with_greco baseline
    # system_weights_dict['ens_greco_on_m7'] = 2
    
    # load all systems predictions
    sys_lines_dict = {}
    for sys_name, file_path in sys_file_dict.items():
        pred_lines = read_lines(file_path, skip_strip=True)
        print(f"Loaded {len(pred_lines)} from {sys_name}")
        assert len(pred_lines) == len(orig_lines)
        sys_lines_dict[sys_name] = pred_lines
        sys_weight = system_weights_dict.get(sys_name, 1)
        if sys_weight > 1:
            for i in range(sys_weight - 1):
                new_sys_name = f"{sys_name}_v{i+2}"
                sys_lines_dict[new_sys_name] = pred_lines
                print(f"Loaded {len(pred_lines)} from {new_sys_name}")
        
    # convert system preds to AnnotatedTokens format
    sentence_preds_dict = {}
    for i, sentence in tqdm(enumerate(orig_lines)):
        sentence_preds_dict[i] = {'original': sentence}
        for sys_name, all_preds in sys_lines_dict.items():
            sys_pred = all_preds[i]
            ann_tokens_pred = align(sentence, sys_pred)
            sentence_preds_dict[i][sys_name] = ann_tokens_pred
    print(f"Preprocessing data is finished.")
    
    # make majority-based ensemble 
    output_ensemble_lines = []
    for i in range(len(orig_lines)):
        sentence_dict = sentence_preds_dict[i]
        ens_pred_tokens = get_majority_based_combination(sentence_dict, min_relative_threshold=args.min_relative_threshold)
        ens_pred_sentence = ens_pred_tokens.get_corrected_text()
        output_ensemble_lines.append(ens_pred_sentence)
        
    # dump predictions
    out_file = os.path.join(args.output_directory, f'tmp_ens_pred_{evalset_name}.txt')
    write_lines(out_file, output_ensemble_lines)
    print(f"Dumped {len(output_ensemble_lines)} ensembled lines.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        # default='data/system_preds/single_systems/',
                        help="The directory with all single system predicitons.")
    parser.add_argument('--output_directory', 
                        # default='data/system_preds/ensemble_systems/',
                        help="The directory with ensemble predicitons output.")
    parser.add_argument('--evalset_path', 
                        # default='data/evaluation_sets/nucle14-2a.txt',
                        # default='data/evaluation_sets/bea-dev.txt',
                        # default='data/evaluation_sets/bea-test.txt',
                        help="Path to the input evalaution sets")
    parser.add_argument('--min_relative_threshold', 
                        default=3)
    args = parser.parse_args()
    main(args)
