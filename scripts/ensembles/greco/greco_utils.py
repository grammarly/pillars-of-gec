import json
import random

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from itertools import chain, combinations
import random
from os.path import basename, isdir, isfile, join, splitext

_IGNORE_TYPE = {"noop", "UNK", "Um"}
_EDIT_START = 0
_EDIT_END = 1
_EDIT_TYPE = 2
_EDIT_COR = 3


def multiple_insertion(x, y):
    return x[0] == x[1] == y[0] == y[1]

def intersecting_range(x, y):
    return (x[0] <= y[0] < x[1] and not x[0] == y[1]) or \
            (y[0] <= x[0] < y[1] and not y[0] == x[1])

def no_conflict(edit, selected_edits):
    for selected_edit in selected_edits:
        if multiple_insertion(edit, selected_edit) \
            or intersecting_range(edit, selected_edit):
            return False
    
    return True

def filter_conflict(edits):
    filtered_edits = []
    for edit in edits:
        if no_conflict(edit, filtered_edits):
            filtered_edits.append(edit)
    return sorted(filtered_edits)

def powerset(iterable, shuffle=False):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    edit_nums = list(range(len(s)+1))
    if shuffle:
        random.shuffle(edit_nums)
    return chain.from_iterable(combinations(s, r) for r in edit_nums)

def m2_to_edits(m2_entity):
    m2_lines = m2_entity.split('\n')
    source = m2_lines[0][2:]
    edits = []
    for m2_line in m2_lines[1:]:
        if not m2_line.startswith("A"):
            raise ValueError("{} is not an m2 edit".format(m2_line))
        m2_line = m2_line[2:]
        features = m2_line.split("|||")
        span = features[0].split()
        start, end = int(span[0]), int(span[1])
        error_type = features[1]
        if error_type.strip() in _IGNORE_TYPE:
            continue
        replace_token = features[2]
        edits.append((start, end, error_type, replace_token))
    return {'source': source, 'edits': edits}

def read_m2(filepath):
    with open(filepath, encoding='utf-8') as f:
        m2_entries = f.read().strip().split('\n\n')
    
    return m2_entries

def read_data(src_path, file_path, m2_dir, target_m2=None, filter_idx=None):
    m2_path = join(m2_dir, splitext(basename(file_path))[0] + '.m2')

    if not isfile(m2_path):
        parse_m2(src_path, file_path, m2_path)
    
    hyp_m2 = read_m2(m2_path)
    if filter_idx is not None:
        hyp_m2 = [hyp_m2[i] for i in filter_idx]

    hyp_m2 = [m2_to_edits(m) for m in hyp_m2]

    if target_m2 is not None:
        assert len(target_m2) == len(hyp_m2), \
            "The m2 lengths of target ({}) and hypothesis ({}) are different!"\
                .format(len(target_m2), len(hyp_m2))
        for hyp_entry, trg_entry in zip(hyp_m2, target_m2):
            assert hyp_entry['source'] == trg_entry['source']
            hyp_edits = hyp_entry['edits']
            trg_edits = set([(t[_EDIT_START], t[_EDIT_END], t[_EDIT_COR]) for t in trg_entry['edits']])
            labels = []
            for edit in hyp_edits:
                e_start, e_end, e_type, e_cor = edit
                label = 1 if (e_start, e_end, e_cor) in trg_edits else 0
                labels.append(label)
            hyp_entry['labels'] = labels
    
    return hyp_m2

def sort_edits_no_type(edits, filter=True): # edits: [(start, end, error_type, replace_token)]
    if filter:
        edits = [e for e in edits if e[_EDIT_START] >= 0]
    edits = list(set(edits)) # remove duplicates
    return sorted(edits)

def sort_edits(edits, filter=True): # edits: [(start, end, error_type, replace_token)]
    if filter:
        edits = [e for e in edits if e[_EDIT_TYPE] not in _IGNORE_TYPE]
    edits = list(set(edits)) # remove duplicates
    return sorted(edits)

def edits_to_text(ori, edits, offset=0):
    _edits = sort_edits(edits)
    cor_sent = ori.split()

    offset = offset
    for edit in _edits:
        if edit[_EDIT_TYPE] in _IGNORE_TYPE: continue  # Ignore certain edits
        start = edit[_EDIT_START]
        end = edit[_EDIT_END]
        cor = edit[_EDIT_COR].split()
        len_cor = 0 if len(edit[_EDIT_COR]) == 0 else len(cor)
        cor_sent[start + offset:end + offset] = cor
        offset = offset - (end - start) + len_cor
    result = " ".join(cor_sent)

    return result, offset


def tokenize(tokenizer, data, pad_to_max_length=False, max_length=200, mask_source=False, label_smoothing=0,
        additional_mask=0, label_all_tokens=False):
    padding = "max_length" if pad_to_max_length else False
    tokenized_inputs = tokenizer(
        data['source'], data['hyp'],
        padding=padding,
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    final_labels = []
    final_gaps = []
    final_masks = []
    final_gap_masks = []
    final_hyp_masks = []
    final_g_hyp_masks = []
    for i, (label, gap, mask) in enumerate(zip(data['labels'], data['gap'], data['masks'])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        gap_mask = data.get('gap_mask', [True] * len(gap))[i]
        previous_word_idx = None
        label_ids = []
        gap_ids = []
        gap_mask_ids = []
        mask_ids = []
        hyp_mask_ids = []
        hyp_start_idx = -1
        part_idx = 1
        none_count = 0
        for token_idx, word_idx in enumerate(word_ids):
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function
            if word_idx is None:
                #labels
                if token_idx == 0:
                    label_ids.append(label[0][0])
                else:
                    label_ids.append(-100)

                if previous_word_idx is not None: # skip the first for roberta
                    none_count += 1
                mask_ids.append(False)
                gap_ids.append(-100)
                gap_mask_ids.append(False)
                hyp_mask_ids.append(False)

                continue

            if none_count > 0 and word_idx <= previous_word_idx:
                # print(token_idx, 'w ', word_idx, 'p ', previous_word_idx)
                part_idx += 1 # start of hyp part
                hyp_start_idx = token_idx
                gap_ids[token_idx - 1] = gap[0]
                gap_mask_ids[token_idx - 1] = gap_mask[0]
                none_count = 0 # reset it again

            if word_idx == previous_word_idx and not label_all_tokens:
                label_ids.append(-100)
                gap_ids.append(-100)
                gap_mask_ids.append(False)
                mask_ids.append(False)
                hyp_mask_ids.append(False)
            else:
                try:
                    if mask_source and part_idx <= 1:
                        label_ids.append(-100)
                    else:
                        cur_label = label[part_idx][word_idx]
                        apply_add_mask = False
                        # only apply additional mask on non-edits
                        if additional_mask > 0 and \
                            not mask[part_idx - 1][word_idx]:
                            if torch.rand(1).item() <= additional_mask:
                                cur_label = -100
                                apply_add_mask = True
                        if label_smoothing > 0 and not apply_add_mask:
                            if cur_label == 1:
                                cur_label -= label_smoothing
                            elif cur_label == 0:
                                cur_label += label_smoothing
                            else:
                                raise ValueError("Label contains " + str(cur_label) \
                                    + ", don't use label smoothing")
                        label_ids.append(cur_label)
                    mask_ids.append(mask[part_idx - 1][word_idx]) # mask only have 2 parts, no F0.5 score
                    if part_idx <= 1:
                        gap_ids.append(-100)
                        gap_mask_ids.append(False)
                        hyp_mask_ids.append(False)
                    else:
                        cur_gap = gap[word_idx + 1]
                        apply_add_mask = False
                        if additional_mask > 0 and \
                            not gap_mask[word_idx + 1]:
                            if torch.rand(1).item() <= additional_mask:
                                cur_gap = -100
                                apply_add_mask = True
                        elif label_smoothing > 0 and not apply_add_mask:
                            if cur_gap == 1:
                                cur_gap -= label_smoothing
                            elif cur_gap == 0:
                                cur_gap += label_smoothing
                            else:
                                raise ValueError("Label contains " + str(cur_gap) \
                                    + ", don't use label smoothing")
                        gap_ids.append(cur_gap)
                        gap_mask_ids.append(gap_mask[word_idx + 1])
                        hyp_mask_ids.append(True)
                except:
                    print(data['source'][i], '\n', data['hyp'][i], '\n', part_idx, len(label[1]), len(label[2]), '\n', word_ids)
            previous_word_idx = word_idx if word_idx is not None else previous_word_idx # ignore None

        final_labels.append(label_ids)
        final_masks.append(mask_ids)
        final_gaps.append(gap_ids)
        final_gap_masks.append(gap_mask_ids)
        final_hyp_masks.append(hyp_mask_ids)
        g_hyp_mask_ids = hyp_mask_ids.copy()
        g_hyp_mask_ids[hyp_start_idx - 1] = True
        final_g_hyp_masks.append(g_hyp_mask_ids)

    tokenized_inputs["labels"] = final_labels
    tokenized_inputs["masks"] = final_masks
    tokenized_inputs['gap'] = final_gaps
    tokenized_inputs['gap_mask'] = final_gap_masks
    tokenized_inputs['hyp_mask'] = final_hyp_masks
    tokenized_inputs['g_hyp_mask'] = final_g_hyp_masks
    
    return tokenized_inputs
        # return []


@dataclass
class CustomDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        serialize (:obj:`bool`, `optional`):
            Serialize each item if the dict values are lists instad of singular items.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    serialize: Optional[bool] = True
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = {k: [] for k in features[0].keys()}
        for f in features:
            for k, v in f.items():
                if isinstance(v, list) and self.serialize:
                    for _v in v:
                        examples[k].append(_v)
                else:
                    examples[k].append(v)
        
        size = max([len(x) for x in examples['input_ids']])
        if self.max_length is not None:
            size = max(size, self.max_length)
        if self.pad_to_multiple_of is not None:
            size = ((size // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        
        left_pad = self.tokenizer.padding_side == 'left'
        
        batch = {}
        for k, values in examples.items():
            if k == "attention_mask":
                pad_idx = 0
            elif k == "special_tokens_mask":
                pad_idx = 1
            elif k in ["masks", "gap_mask", "hyp_mask", "g_hyp_mask"]:
                pad_idx = False
            elif k == "token_type_ids":
                pad_idx = self.tokenizer.pad_token_type_id
            else:
                pad_idx = self.tokenizer.pad_token_id
            
            if k == "labels":
                dtype = torch.float
            elif k in ["masks", "gap_mask", "hyp_mask", "g_hyp_mask"]:
                dtype = torch.bool
            else:
                dtype = torch.long
            res = torch.empty((len(values), size), dtype=dtype).fill_(pad_idx)

            def copy_tensor(src, dst):
                assert dst.numel() == src.numel()
                dst.copy_(src)

            for i, v in enumerate(values):
                v = torch.tensor(v)
                copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
            
            batch[k] = res

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        return batch


class TrainDataset(Dataset):
    
    def __init__(self, train_file, tokenizer, mask_source=False,
            max_length=200, mode='list', label_smoothing=0,
            additional_mask=0):
        super().__init__()

        with open(train_file, encoding='utf-8') as f:
            lines = f.readlines()
        data = [json.loads(line.strip()) for line in lines]
        self.key = 'input_ids'
        if isinstance(data[0], list):
            data = self.transform_list_to_dict(data)
        # print('>> data content: ', data[0])
        self.data = [tokenize(tokenizer, d, True, max_length, \
                        mask_source, label_smoothing, additional_mask) \
                        for d in data]
        # print('>> tokenizer: ', len(self.data[0]['input_ids']), '\n', self.data[0]['input_ids'])
        bucket_sizes = [len(b[self.key]) for b in self.data]
        self.bucket_size = max(bucket_sizes)
        assert min(bucket_sizes) == self.bucket_size, "the batch size are not uniform"\
            .format(min(bucket_sizes), self.bucket_size)


    def transform_list_to_dict(self, data):
        formatted_result = []
        first_keys = None
        for batch in data:
            dict_result = {}
            for datum in batch:
                first = len(dict_result.keys()) == 0
                if not first:
                    diff = set(datum.keys()) - set(dict_result.keys())
                    if len(diff) > 0:
                        print('[WARNING] new key {} was not present in previous instance'\
                                .format(k))
                    for k in diff:
                        dict_result[k] = v
                for k, v in datum.items():
                    if k not in dict_result:
                        dict_result[k] = []
                    dict_result[k].append(v)
            formatted_result.append(dict_result)
        return formatted_result


    def __getitem__(self, index: int):        
        return self.data[index]
    

    def __len__(self) -> int:
        return len(self.data)


class SysCombDataset(Dataset):
    
    def __init__(self, files, merge_consecutive=False, edit_scores=None):
        super().__init__()
     
        hyps_m2 = [read_m2(h) for h in files]
        self.num_hyp = len(hyps_m2)
        self.hyp_entities = [[] for _ in range(len(hyps_m2[0]))]
        self.sources = [None for _ in range(len(hyps_m2[0]))]
        
        if edit_scores is not None:
            if merge_consecutive:
                raise NotImplementedError(
                    "Edit scores with edit merging is not implemented yet")
            
            if isinstance(edit_scores, str):
                with open(edit_scores, encoding='utf-8') as f:
                    data = f.readlines()
                edit_scores = [json.loads(d.strip()) for d in data]
            else:
                edit_scores = edit_scores
            self.edit_scores = []
            for sent_edits in edit_scores:
                sent_edit_scores = {}
                for edit in sent_edits:
                    e_start, e_end, e_rep, e_score = edit
                    sent_edit_scores[(e_start, e_end, e_rep)] = e_score
                self.edit_scores.append(sent_edit_scores)
            assert len(self.edit_scores) == len(self.hyp_entities)
        else:
            self.edit_scores = None
        
        for hyp in hyps_m2:
            assert len(hyp) == len(hyps_m2[0]), \
                "Hypothesis length ({}) is different from target ({})"\
                    .format(len(hyp), len(hyps_m2[0]))
            for idx, m2 in enumerate(hyp):
                _hyp_entity = m2_to_edits(m2)
                if merge_consecutive:
                    _hyp_edits = []
                    for edit in sort_edits(_hyp_entity['edits']):
                        e_start, e_end, e_type, e_rep = edit
                        if e_type in _IGNORE_TYPE:
                            continue
                        if merge_consecutive and last_end_idx == e_start:
                            last_edit = _hyp_edits.pop()
                            le_start, le_end, le_type, le_rep = last_edit
                            e_rep = (le_rep + " " + e_rep).strip()
                            e_start = le_start
                            e_type = le_type + "-" + e_type
                        _hyp_edits.append((e_start, e_end, e_type, e_rep))
                        last_end_idx = e_end
                else:
                    _hyp_edits = _hyp_entity['edits']
                self.hyp_entities[idx].extend(_hyp_edits)
                if self.sources[idx] is None:
                    self.sources[idx] = _hyp_entity['source']
                else:
                    assert self.sources[idx] == _hyp_entity['source']
        

    def __getitem__(self, index: int):
        edits = [(e[0], e[1], '', e[3]) for e in self.hyp_entities[index]]
        edits = sort_edits_no_type(edits)
        edit_set = {}
        for e in self.hyp_entities[index]:
            key = (e[0], e[1], e[3])
            if key not in edit_set:
                edit_set[key] = {
                    'type': e[2],
                    'count': 1,
                }
            else:
                edit_set[key]['count'] += 1
            if (edit_set[key]['type'].endswith('UNK') or \
                edit_set[key]['type'].endswith('OTHER')) and not \
                (e[2].endswith('UNK') or e[2].endswith('OTHER')):
                edit_set[key]['type'] = e[2]
        edit_count = []
        for e in edits:
            key = (e[0], e[1], e[3])
            e_type = edit_set[key]['type']
            e_count = edit_set[key]['count']
            new_key = (e[0], e[1], e_type, e[3])
            edit_count.append((new_key, e_count))
        data = {
            'source': self.sources[index],
            'edits': edit_count,
            'hyps': [edits_to_text(self.sources[index], [e])[0] \
                        for e in edits]
        }
        if self.edit_scores is not None:
            data['scores'] = self.edit_scores[index]
            if len(edits) != len(self.edit_scores[index]):
                # the edit set may contain duplicate edits with different edit types
                print('[WARNING] size of edit set ({}) != score set ({})'\
                    .format(len(edits), len(self.edit_scores[index])))
        
        return data

    
    def __len__(self) -> int:
        return len(self.sources)