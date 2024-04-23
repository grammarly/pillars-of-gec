"""Copied and adapted from official repository: https://github.com/nusnlp/greco/"""
import argparse
import math
import numbers
import os
from operator import itemgetter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from greco_utils import SysCombDataset, edits_to_text, no_conflict, filter_conflict
# from models import get_model
from greco_model import GRECO


def beam_search(dataset, model, beam_size, batch_size, edit_score_w=0.5,
        vote_coef=0, verbose=False):
    def process_queue(queue):
        num_batch = math.ceil(len(queue) / batch_size)
        batched_outputs = []
        for i in range(num_batch):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(queue))
            batch = [q['sentence'] for q, _ in queue[start_idx:end_idx]]
            preds = model.score(source, batch)
            if not isinstance(preds, list):
                preds = preds.squeeze().cpu().tolist()
            if not isinstance(preds, list):
                preds = [preds]
            assert isinstance(preds[0], numbers.Number), \
                "preds is a list of {}, should be list<number>".format(type(preds[0]))
            preds = [math.log(p) for p in preds]
            batched_outputs.extend(preds)
        
        if verbose:
            for q, score in zip(queue, batched_outputs):
                print(' > [{:.2f}] [{:.2f}] {} {}'.format(score['text'], score['edit'],\
                        q[0]['edits'], q[0]['sentence']))

        return batched_outputs

    def clean_beam(beam): # the beam contains List<(seq, score_dict)>
        num_edits = max([b[1]['num_edit'] for b in beam])
        if num_edits > 0:
            edit_scoring = lambda x: 0 if x[1]['num_edit'] == 0 else x[1]['edit'] / x[1]['num_edit']
            sort_key = lambda x: ((1- edit_score_w) * x[1]['text']) + vote_coef * math.log(x[1]['count']/x[1]['used_edit']) + edit_scoring(x) * edit_score_w
        elif vote_coef > 0:
            sort_key = lambda x: x[1]['text'] + vote_coef * math.log(x[1]['count']/x[1]['used_edit'])
        else:
            sort_key = lambda x: x[1]['text']
        sorted_beam = sorted(beam, key=sort_key, reverse=True)
        return sorted_beam[:beam_size]

    dataloader = DataLoader(dataset, 1, shuffle=False)
    result = [None] * len(dataset)
    num_hyp = dataset.num_hyp
    print('Combining {} hypotheses'.format(num_hyp))
    for idx, data in enumerate(tqdm(dataset)):
        source = data['source']
        edits = data['edits']
        hyps = [h for h in data['hyps']]
        assert len(edits) == len(hyps)

        if len(edits) == 0:
            result[idx] = dataset.sources[idx]
            continue
        
        edit_scores = data.get('scores', None)

        beam = []
        queue = [({
            'sentence': source,
            'edits': [],
            'offset': 0,
        }, {
            'text': 0,
            'edit': 0,
            'num_edit': 0,
            'count': (1/num_hyp),
            'used_edit': 1,
        })]
        if verbose:
            print('\n====', '\n# ', source)
        first = True
        for id_edit, (edit, e_count) in enumerate(edits):
            new_queue = []
            e_start, e_end, e_type, rep_token = edit
            for q, score in queue:
                new_score = score.copy()
                new_score['count'] += (e_count/num_hyp)
                new_score['used_edit'] += 1
                if edit_scores is not None:
                    edit_score = edit_scores.get((e_start, e_end, rep_token), None)
                    if edit_score is not None:
                        comp_score = 1 - edit_score # complement score
                        new_score['edit'] += math.log(edit_score)
                        new_score['num_edit'] += 1
                        score['edit'] += math.log(comp_score)
                        score['num_edit'] += 1
                        assert len(q['edits']) <= score['num_edit'] <= len(edits)
                    else:
                        print('[WARNING] edit score for {} is not found'.format(edit))

                if no_conflict(edit, q['edits']):
                    cur_edit = [edit]
                    new_sen, offs = edits_to_text(q['sentence'], cur_edit, q['offset'])
                    new_queue.append(({
                        'sentence': new_sen,
                        'edits': q['edits'] + cur_edit,
                        'offset': offs,
                    }, new_score))

            if first:
                queue.extend(new_queue)
                new_queue = queue.copy()

            if len(queue) >= beam_size or id_edit == len(edits) - 1:
                new_scores = [s for q, s in new_queue]
                text_scores = process_queue(new_queue)
                for s, t in zip(new_scores, text_scores):
                    s['text'] = t
                beam.extend(new_queue)
                queue = clean_beam(beam)
                first = False
        
        beam = clean_beam(beam)
        if verbose and idx <= 4:
            print(beam)
        best_seq_data, highest_score = beam[0]
        result[idx] = best_seq_data['sentence']
    
    return result


def main(args):
    if isinstance(args.data, str):
        data_list = args.data.split()
    else:
        data_list = args.data
    dataset = SysCombDataset(data_list, args.merge_consecutive, args.edit_scores)
    # dataloader = DataLoader(dataset, 1, shuffle=False)
    print(f"Loading data is finished.")

    # model = get_model(args)
    # model.eval()
    # load model
    device = "cuda"
    model = GRECO('microsoft/deberta-v3-large').to(device)
    model.load_state_dict(torch.load(args.path_to_model))
    print('Model initialization is finished.')
    
    
    with torch.no_grad():
        result = beam_search(dataset, model, args.beam_size, args.batch_size,
                    args.score_ratio, args.vote_coef, args.verbose)
        
    # dump to out file
    evalset = data_list[0].split("__")[-1].replace(".m2", "")
    out_file = args.output_prefix + f"__{evalset}.txt"

    with open(out_file, 'w', encoding='utf-8') as out:
        out.write('\n'.join(result))
    print(f"PRED_FILE={out_file}")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+',)
    parser.add_argument('--path_to_model', help='path to the model',)
                        # default='models/greco/models/checkpoint.bin')
    parser.add_argument('--edit_scores', default=None, help='path to file containing scores of each edit')
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--beam_size', type=int, default=16, help="beam size")
    parser.add_argument('--vote_coef', type=float, 
                        # default=0, 
                        default=0.5,
                        help="The voting coefficient during decoding")
    parser.add_argument('--score_ratio', type=float, 
                        default=0, 
                        help="ratio of edit score during decoding")
    parser.add_argument('--verbose', default=False, action='store_true', help='verbose')
    parser.add_argument('--merge_consecutive', default=False, action='store_true', help='merge consecutive edits')
    parser.add_argument('--output_prefix', 
                        help='path to the output file during testing',
                        default='preds/greco_beam_pred',)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
