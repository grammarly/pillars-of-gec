"""Copied from official repository: https://github.com/nusnlp/greco/"""

import functools
import math
import os
import sys

from multiprocessing import Pool
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import (
    InputExample,
    InputFeatures,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    BertPreTrainedModel,
)

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    neuqe_dir = os.path.join(SCRIPT_DIR, 'neuqe')
    sys.path.append(neuqe_dir)
    from neuqe.models.model_utils import set_predictor_arch, set_estimator_arch
    from neuqe.io import io_utils
    neuqe_import_error = None
except Exception as e:
    neuqe_import_error = e


class GRECO(nn.Module):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, lm, tokenizer=None, dropout=0, alpha=1, beta=1, gamma=1, epsilon=0,
        label_weight=None, gap_weight=None, edit_weight=None, freeze_lm=False,
        ranking_loss='naive', estimator_loss='h_listnet', rank_multiplier=1, rank_sample=None):
        super(GRECO, self).__init__()

        self.lm = AutoModel.from_pretrained(lm)
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False
        tokenizer = tokenizer if tokenizer is not None else lm
        add_prefix_space = True if 'roberta' in tokenizer.lower() else False
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, do_lower_case=False, use_fast=True,
            add_prefix_space=add_prefix_space)
        self.dropout = nn.Dropout(dropout)
        self.config = self.lm.config
        hidden_size = self.lm.config.hidden_size
        # legacy F0.5 score projection
        self.score_proj = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.Tanh(),
                            nn.Linear(hidden_size, 1),
                          )
        self.score_proj.apply(self._init_weight)
        self.token_proj = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.Tanh(),
                            nn.Linear(hidden_size, 1),
                            nn.Sigmoid(),
                          )
        self.token_proj.apply(self._init_weight)
        self.gap_proj = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.Tanh(),
                            nn.Linear(hidden_size, 1),
                            nn.Sigmoid(),
                        )
        self.gap_proj.apply(self._init_weight)
        
        self.a = alpha
        self.b = beta
        self.c = gamma
        self.e = epsilon

        self.label_weight = label_weight
        self.gap_weight = gap_weight
        self.edit_weight = edit_weight
        self.rank_mult = rank_multiplier
        self.rank_sample = rank_sample
        self.estimator_loss = estimator_loss

        device_str = 'cpu'
        if torch.cuda.is_available():
            device_str = 'cuda:{}'.format(0)

        self.device = torch.device(device_str)


    def set_bucket_size(self, bucket_size):
        self.bucket_size = bucket_size


    def get_tokenizer(self):
        return self.tokenizer


    def _init_weight(self, module):
        if type(module) == torch.nn.Linear:
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        gap: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        gap_mask: Optional[torch.Tensor] = None,
        hyp_mask: Optional[torch.Tensor] = None,
        g_hyp_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if head_mask is not None:
            outputs = self.lm(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            outputs = self.lm(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs[0]
        score = self.score_proj(sequence_output[:, 0, :])  # take <s> token (equiv. to [CLS])
        sequence_output = self.dropout(sequence_output)
        word_logits = self.token_proj(sequence_output)
        gap_logits = self.gap_proj(sequence_output)

        loss = None
        if labels is not None:
            if self.edit_weight is not None:
                assert masks is not None and gap_mask is not None, "edit weight need masks"
                assert list(gap.shape) == list(gap_mask.shape), "gap size ({}) != gap mask({})"\
                    .format(list(gap.shape), list(gap_mask.shape))

            # F0.5 regression
            loss_fct = nn.MSELoss()
            loss_s = loss_fct(score.squeeze(-1),
                       labels[:,0].to(sequence_output.dtype))

            # word label classification
            if self.label_weight is not None:
                label_weight = torch.ones_like(labels)
                for k, v in self.label_weight.items():
                    _key = torch.full_like(labels, k)
                    _value = torch.full_like(labels, v)
                    _mask = torch.isclose(labels, _key)
                    label_weight = label_weight.where(_mask, _value)
                loss_fct = nn.BCELoss(weight=label_weight, reduction='none')
            elif self.edit_weight is not None:
                label_weight = torch.ones_like(labels, dtype=torch.float)
                _value = torch.full_like(label_weight, self.edit_weight)
                # label_weight = label_weight.where(masks, _value)
                label_weight = torch.where(masks, _value, label_weight)
                loss_fct = nn.BCELoss(weight=label_weight, reduction='none')
            else:
                loss_fct = nn.BCELoss(reduction='none')
            loss_w_all = loss_fct(word_logits.squeeze(-1),
                                labels.to(sequence_output.dtype))
            mask_w = labels != -100
            loss_w_masked = loss_w_all.where(mask_w,
                            torch.tensor(0.0, device=mask_w.device))
            loss_w = loss_w_masked.sum() / mask_w.sum()

            loss = self.a * loss_w

            # gap label classification
            if self.gap_weight is not None:
                gap_weight = torch.ones_like(gap)
                for k, v in self.gap_weight.items():
                    _key = torch.full_like(gap, k)
                    _value = torch.full_like(gap, v)
                    _mask = torch.isclose(gap, _key)
                    gap_weight = gap_weight.where(_mask, _value)
                loss_fct = nn.BCELoss(weight=gap_weight, reduction='none')
            elif self.edit_weight is not None:
                gap_weight = torch.ones_like(gap, dtype=torch.float)
                _value = torch.full_like(gap_weight, self.edit_weight)
                # gap_weight = gap_weight.where(gap_mask, _value)
                gap_weight = torch.where(gap_mask, _value, gap_weight)
                loss_fct = nn.BCELoss(weight=gap_weight, reduction='none')
            else:
                loss_fct = nn.BCELoss(reduction='none')
            loss_g_all = loss_fct(gap_logits.squeeze(-1),
                                gap.to(sequence_output.dtype))
            mask_g = gap != -100
            loss_g_masked = loss_g_all.where(mask_g,
                            torch.tensor(0.0, device=mask_g.device))
            loss_g = loss_g_masked.sum() / mask_g.sum()

            loss += self.b * loss_g

            if self.c > 0 or self.d > 0:
                w_logit_masked = torch.where(hyp_mask,
                            torch.log(word_logits.squeeze(-1)),
                            torch.zeros_like(word_logits.squeeze(-1)))
                g_logit_masked = torch.where(g_hyp_mask,
                                torch.log(gap_logits.squeeze(-1)),
                                torch.zeros_like(gap_logits.squeeze(-1)))
                score_est = torch.div(w_logit_masked.sum(-1) + g_logit_masked.sum(-1),
                                    hyp_mask.sum(-1) + g_hyp_mask.sum(-1)).exp()
                assert len(score_est.shape) == 1, "wrong data format"
                bsz = score_est.shape[0]
                assert bsz % self.bucket_size == 0, \
                        "batch size should be divisible by bucket size"
            
            if self.c > 0:
                num_buckets = bsz // self.bucket_size
                score_table = score_est.view(num_buckets, self.bucket_size)
                
                repetition = torch.arange(self.bucket_size, device=score_table.device)
                _lower_tensor = torch.repeat_interleave(score_table, repetition, dim=-1)
                # make a list of [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
                indices = [j for i in range(1, self.bucket_size) for j in range(i)]
                indices = torch.tensor(indices, device=score_table.device)
                _higher_tensor = torch.index_select(score_table, -1, indices)
                table_length = len(indices)
                
                if self.rank_sample is not None and self.rank_sample < table_length:
                    sample_idx = torch.randint(table_length, (self.rank_sample,),
                                    device=score_table.device)
                    higher_tensor = _higher_tensor.index_select(-1, sample_idx)
                    lower_tensor = _lower_tensor.index_select(-1, sample_idx)
                else:
                    higher_tensor = _higher_tensor
                    lower_tensor = _lower_tensor
                assert lower_tensor.shape == higher_tensor.shape, "Input malformed"
                sigmoid = F.sigmoid(higher_tensor - lower_tensor) * self.rank_mult
                ranking_loss = torch.log(1 + torch.exp(-sigmoid)).sum()
                loss += self.c * ranking_loss

            # F0.5 estimation
            if self.e > 0:
                if self.estimator_loss.lower() == 'listnet':
                    soft_label = F.softmax(labels[:,0], dim=-1)
                    loss_fct = nn.CrossEntropyLoss()
                    loss_e = loss_fct(score_est.unsqueeze(0), soft_label.unsqueeze(0))
                elif self.estimator_loss.lower() == 'h_listnet':
                    bsz = score_est.shape[0]
                    assert bsz % self.bucket_size == 0, \
                            "batch size should be divisible by bucket size"
                    num_buckets = bsz // self.bucket_size
                    score_table = score_est.view(num_buckets, self.bucket_size)
                    soft_label = F.softmax(
                                    labels[:,0].view(num_buckets, self.bucket_size),
                                    dim=-1)
                    loss_fct = nn.CrossEntropyLoss()
                    loss_e = loss_fct(score_table, soft_label)
                else:
                    raise NotImplementedError("Loss {} is unkown".format(self.estimator_loss))
                loss += self.e * loss_e

        output = (word_logits,) + outputs[2:]
        
        return ((loss,) + output) if loss is not None else (score, word_logits, gap_logits)


    def score(self, sources, hyps):
        self.eval()
        self.lm.eval()
        if not isinstance(sources, list):
            sources = [sources]

        if len(sources) == 1:
            sources = sources * len(hyps)
        
        if self.config.model_type == 'xlnet':
            max_len = self.config.d_inner
        else:
            max_len = self.config.max_position_embeddings
        tokenized = False
        texts = self.tokenizer(sources, hyps,
                        padding=True,
                        truncation=True,
                        max_length=max_len-2,
                        is_split_into_words=tokenized,
                        return_tensors="pt")
        
        with torch.no_grad():
            output = self.__call__(texts['input_ids'].to(self.device))
            word_logits = output[1].squeeze(-1)
            gap_logits = output[2]
            if gap_logits is not None:
                gap_logits = gap_logits.squeeze(-1)
            w_masks = []
            g_masks = []
            hyp_idcs = []
            for i in range(word_logits.shape[0]): # batch size
                word_ids = texts.word_ids(batch_index=i)
                assert word_logits[i,:].shape[0] == len(word_ids), "{} != {}".format(word_logits[i,:].shape[0], len(word_ids))
                assert gap_logits[i,:].shape[0] == len(word_ids), "{} != {}".format(word_logits[i,:].shape[0], len(word_ids))
                none_count = 0
                last_word_id = -1
                hyp_start_idx = -1
                label_mask = [False] # mask the CLS
                for w_idx, word_id in enumerate(word_ids[1:]):
                    if word_id is None:
                        label_mask.append(False)
                        none_count += 1
                    elif none_count > 0 and word_id <= last_word_id: # start of hyp
                        hyp_start_idx = w_idx + 1
                        hyp_idcs.append(hyp_start_idx)
                        label_mask.append(True)
                        none_count = 0
                    else:
                        if hyp_start_idx < 0 or last_word_id == word_id:
                            label_mask.append(False)
                        else: # hyp
                            label_mask.append(True)
                    last_word_id = word_id if word_id is not None else last_word_id #ignore None
                gap_mask = label_mask.copy()
                gap_mask[hyp_start_idx - 1] = True
                assert len(label_mask) == len(word_ids)
                w_masks.append(label_mask)
                g_masks.append(gap_mask)
            
            w_masks = torch.tensor(w_masks, device=word_logits.device)
            word_logits = torch.log(word_logits)
            gap_logits = torch.log(gap_logits)
            w_logit_masked = word_logits.where(w_masks,
                            torch.tensor(0.0, device=word_logits.device))
            g_masks = torch.tensor(g_masks, device=gap_logits.device)
            g_logit_masked = gap_logits.where(g_masks,
                            torch.tensor(0.0, device=gap_logits.device))
            output = torch.div(w_logit_masked.sum(-1) + g_logit_masked.sum(-1),
                                w_masks.sum(-1) + g_masks.sum(-1)).exp()
        
        return output


class ModelArgsWrapper():
    def __init__(self, args=None):
        super(ModelArgsWrapper, self).__init__()
        if args is not None:
            for k, v in args.items():
                setattr(self, k, v)


    def assign_properties(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class GPT2(nn.Module):
    def __init__(self, model_id='gpt2-large'):
        super(GPT2, self).__init__()
        
        device_str = 'cpu'
        if torch.cuda.is_available():
            device_str = 'cuda:{}'.format(0)

        self.device = torch.device(device_str)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


    def score(self, source, hyp, score_src=False):
        hyp_score, _ = self._score(hyp)
        if not score_src:
            return hyp_score
        else:
            src_score, _ = self._score(source)
            return hyp_score - src_score


    def _score(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.n_positions
        stride = 512

        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        if len(nlls) == 0:
            return torch.tensor(0), torch.tensor(-1)
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        prob = torch.exp(-1 * torch.stack(nlls).sum() / end_loc)

        return prob, ppl


class MLM(nn.Module):
    def __init__(self, model_id):
        super(GPT2, self).__init__()
        
        device_str = 'cpu'
        if torch.cuda.is_available():
            device_str = 'cuda:{}'.format(0)

        self.device = torch.device(device_str)
        self.model = AutoModelForMaskedLM.from_pretrained(
                        model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)


    def score(self, source, hyp):
        src_score, _ = self._score(source)
        hyp_score, _ = self._score(hyp)
        return hyp_score - src_score


    def _score(self, sentence):
        tensor_input = self.tokenizer.encode(sentence,
                            return_tensors='pt').to(self.device)
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1,
                            self.tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(
                    masked_input != self.tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = model(masked_input, labels=labels).loss
        
        return torch.exp(-1 * loss)


class VERNET(nn.Module):


    def __init__(self, bert_model, checkpoint, inference_model):
        super(VERNET, self).__init__()
        
        device_str = 'cpu'
        if torch.cuda.is_available():
            device_str = 'cuda:{}'.format(0)

        self.device = torch.device(device_str)

        model = AutoModel.from_pretrained(bert_model)
        model = model.to(self.device)
        self.max_len = 120

        
        model_args = ModelArgsWrapper()

        dict_model_args = {
            'bert_hidden_dim': 768,
            'bert_pretrain': bert_model,
            'max_len': self.max_len,
            'evi_num': 1,
        }
        for k, v in dict_model_args.items():
            setattr(model_args, k, v)
        model = inference_model(model, model_args)
        model.load_state_dict(torch.load(checkpoint)['model'])
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)


    def tok2int_sent(self, example, tokenizer, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""
        src_tokens = example[0]
        hyp_tokens = example[1]

        src_tokens = tokenizer.tokenize(src_tokens)
        src_tokens = src_tokens[:max_seq_length]
        hyp_tokens = tokenizer.tokenize(hyp_tokens)
        hyp_tokens = hyp_tokens[:max_seq_length]

        tokens = ["[CLS]"] + src_tokens + ["[SEP]"]
        input_seg = [0] * len(tokens)
        input_label = [0] * len(tokens)

        tokens = tokens + hyp_tokens  + ["[SEP]"]
        for token in hyp_tokens:
            if "##" in token:
                input_label.append(0)
            else:
                input_label.append(1)
        input_label.append(1)
        input_seg = input_seg + [1] * (len(hyp_tokens) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)


        max_len = max_seq_length * 2 + 3
        padding = [0] * (max_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        input_seg += padding
        input_label += padding

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(input_seg) == max_len
        assert len(input_label) == max_len
        return input_ids, input_mask, input_seg, input_label


    def tok2int_list(self, examples):
        """Loads a data file into a list of `InputBatch`s."""
        inps = list()
        msks = list()
        segs = list()
        labs = list()
        for example in examples:
            input_ids, input_mask, input_seg, input_lab = self.tok2int_sent(
                example, self.tokenizer, self.max_len)
            inps.append(input_ids)
            msks.append(input_mask)
            segs.append(input_seg)
            labs.append(input_lab)
        return inps, msks, segs, labs
    
    
    def score(self, sources, hyps):
        if not isinstance(sources, list):
            sources = [sources]

        if len(sources) == 1:
            sources = sources * len(hyps)
        examples = []
        for src, hyp in zip(sources, hyps):
            examples.append([src, hyp])
        self.eval()
        self.model.eval()
        with torch.no_grad():
            inp_tensor, msk_tensor, seg_tensor, lab_tensor = self.tok2int_list(examples)
            inp_tensor = torch.LongTensor(inp_tensor).to(self.device)
            msk_tensor = torch.LongTensor(msk_tensor).to(self.device)
            seg_tensor = torch.LongTensor(seg_tensor).to(self.device)
            lab_tensor = torch.LongTensor(lab_tensor).to(self.device)

            prob = self.model(inp_tensor, msk_tensor, seg_tensor, score_flag = False)
            prob = prob.view(-1, self.max_len * 2 + 3, 4)
            prob = prob[:, :, :2]
            prob = F.softmax(prob, -1)
            prob = prob[:, :, 1].squeeze(-1)
            prob = torch.sum(prob * lab_tensor.float(), 1) / torch.sum(lab_tensor.float(), 1)
            
            return prob


class SOME(nn.Module):
    def __init__(self, args_dict):
        super(SOME, self).__init__()

        device_str = 'cpu'
        if torch.cuda.is_available():
            device_str = 'cuda:{}'.format(0)

        self.device = torch.device(device_str)
        self.args = ModelArgsWrapper(args_dict)
        self.model_g = BertForSequenceClassification.from_pretrained(self.args.g_dir)
        self.model_f = BertForSequenceClassification.from_pretrained(self.args.f_dir)
        self.model_m = BertForSequenceClassification.from_pretrained(self.args.m_dir)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_type)

        
    def convert_examples_to_features(
        self,
        examples,
        tokenizer,
        max_length=None,
        task=None,
        label_list=None,
        output_mode=None,
    ):
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample):
            if example.label is None:
                return None
            elif output_mode == 'classification':
                return label_map[example.label]
            elif output_mode == 'regression':
                return float(example.label)
            raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        return features


    def create_example(self, src, pred, task):
        examples = []
        if task == 'ssreg':
            for i, (s, p) in enumerate(zip(src, pred)):
                examples.append(
                    InputExample(guid=i, text_a=s, text_b=p, label=None)
                )
        elif task == 'sreg':
            for i, p in enumerate(pred):
                examples.append(
                    InputExample(guid=i, text_a=p, text_b=None, label=None)
                )
        return examples


    def create_dataset(self, src, pred, task=None):
        # load examples and convert to features
        examples = self.create_example(src, pred, task=task)
        tokenizer = self.tokenizer
        features = self.convert_examples_to_features(
            examples,
            tokenizer,
            label_list=[None],
            max_length=128,
            output_mode='regression',
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        data = {
            'input_ids': all_input_ids.to(self.device),
            'attention_mask': all_attention_mask.to(self.device),
        }

        if self.args.model_type == 'distilbert' or \
            ('bert' not in self.args.model_type and 'xlnet' not in self.args.model_type):
            all_token_type_ids = None
        else:
            all_token_type_ids = all_token_type_ids.to(self.device)
        data['token_type_ids'] = all_token_type_ids

        return data


    def predict(self, task):
        if task == 'grammer':
            model = self.model_g
            pred_dataset = self.data_sreg
        elif task == 'fluency':
            model = self.model_f
            pred_dataset = self.data_sreg
        elif task == 'meaning':
            model = self.model_m
            pred_dataset = self.data_ssreg

        model.to(self.device)

        preds = None
        model.eval()

        with torch.no_grad():                
            outputs = model(**pred_dataset)
            logits = outputs[:2][0]

        preds = logits.detach().cpu().numpy()
        preds = np.squeeze(preds, axis=-1)

        return preds


    def add(self, src, pred):
        if not isinstance(src, list):
            src = [src]
        if len(src) == 1:
            src = src * len(pred)
        # make dataset for sreg and ssreg
        self.data_sreg = self.create_dataset(src, pred, task='sreg')
        self.data_ssreg = self.create_dataset(src, pred, task='ssreg')


    def min_max_normalize(self, x, x_min=1, x_max=4):
        return (x - x_min) / (x_max - x_min)

    
    def score(self, sources, hyps):
        self.add(sources, hyps)

        # normalize
        score_g = [self.min_max_normalize(x) for x in self.predict(task='grammer')]
        score_f = [self.min_max_normalize(x) for x in self.predict(task='fluency')]
        score_m = [self.min_max_normalize(x) for x in self.predict(task='meaning')]
        
        assert len(score_g) == len(score_f) == len(score_m)

        # calc gfm score
        scores = []
        for g, f, m in zip(score_g, score_f, score_m):
            scores.append(
                self.args.weight_g * g + self.args.weight_f * f + self.args.weight_m * m
            )

        return scores


class NeuQE(nn.Module):

    def __init__(self, pred_model_path, est_model_path):
        super(NeuQE, self).__init__()
        device_str = 'cpu'
        if torch.cuda.is_available():
            device_str = 'cuda:{}'.format(0)

        self.device = torch.device(device_str)

        pred_checkpoint = torch.load(pred_model_path)
        pred_args = pred_checkpoint['args']
        Predictor = set_predictor_arch(pred_args.architecture)
        self.predictor = Predictor(pred_args).to(self.device)
        self.predictor.load_state_dict(pred_checkpoint['state_dict'])
        self.predictor.eval()
        
        est_checkpoint = torch.load(est_model_path)
        est_args = est_checkpoint['args']
        Estimator = set_estimator_arch(est_args.architecture)
        self.estimator = Estimator(
                            est_args,
                            pred_model=self.predictor
                        ).to(self.device)
        est_model_state = self.estimator.state_dict()
        est_model_state.update(est_checkpoint['state_dict'])
        self.estimator.load_state_dict(est_model_state)
        self.estimator.eval()

        src_vocab, trg_vocab = pred_checkpoint['vocab']
        self.vocab = (src_vocab,trg_vocab)


    def test(self, test_samples, test_scores=None):
        self.estimator.eval()
        sample_idx = 0
        loss = 0
        total_loss_value = 0
        est_criterion = torch.nn.MSELoss
        out_scores = []
        if isinstance(test_samples, list):
            sample_as_batch = test_samples
        elif isinstance(test_samples, tuple):
            sample_as_batch = [test_samples]
        else:
            raise ValueError("sample: ", sample_as_batch)

        pred_input = io_utils.create_predictor_input(sample_as_batch, self.vocab)
        #extract source sentence tokens and target sentence tokens from input
        source = pred_input[0]
        target = pred_input[1]
        source_mask = pred_input[2]
        target_mask = pred_input[3]

        # convert to autograd Variables
        source_input = torch.tensor(source, device=self.device)
        source_mask_input = torch.tensor(source_mask, device=self.device)
        target_ref = torch.tensor(target, device=self.device)
        target_ref_mask = torch.tensor(target_mask, device=self.device)
        target_length = target_ref.size()[0]

        model_input = (source_input, source_mask_input, target_ref, target_ref_mask)
        est_score, log_probs= self.estimator(model_input)

        out_scores = est_score.data # 
        if test_scores:
            scores_ref = torch.FloatTensor([test_scores[sample_idx]], device=self.device)
            est_loss = est_criterion(est_score, scores_ref)
            total_loss_value += (est_loss.data[0])

        sample_idx += 1
        assert len(out_scores) == len(test_samples), \
            "{} != {}".format(len(out_scores), len(test_samples))
        
        if test_scores:
            avg_loss = total_loss_value / len(test_samples)
        else:
            avg_loss = None
        return out_scores, avg_loss



    def score(self, sources, hyps):
        if not isinstance(sources, list):
            sources = [sources]
        if len(sources) == 1:
            sources = sources * len(hyps)
        test_samples = [(src.split(), hyp.split()) for src, hyp in zip(sources, hyps)]
        test_score, _ = self.test(test_samples)
        # print(test_score, test_samples)

        return test_score


def get_model(args):
    device_str = 'cpu'
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)

    device = torch.device(device_str)
    
    if 'greco' in args.model.lower():
        model = GRECO(args.lm_model).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    elif 'gpt' in args.model.lower():
        model = GPT2()
    elif 'mlm' in args.model.lower():
        model = MLM(args.lm_model)
    elif 'vernet' in args.model.lower():
        from VERNet.model.models import inference_model
        lm_model = args.lm_model or 'google/electra-base-discriminator'
        model_dir = args.checkpoint or 'VERNet/checkpoints/electra_model/model.best.pt'
        model = VERNET(lm_model, model_dir, inference_model)
    elif 'some' in args.model.lower():
        if args.checkpoint is not None:
            model_dir = args.checkpoint
        else:
            model_dir = 'checkpoints/some'
        model_args = {
            'model_type': 'bert-base-cased',
            'g_dir': os.path.join(model_dir, 'grammer'),
            'f_dir': os.path.join(model_dir, 'fluency'),
            'm_dir': os.path.join(model_dir, 'meaning'),
            'weight_g': 0.55,
            'weight_f': 0.43,
            'weight_m': 0.02,
        }
        model = SOME(model_args)
    elif 'neuqe' in args.model.lower():  
        if neuqe_import_error is not None:
            raise ImportError("Failed to import NueQE modules.\n{}".format(neuqe_import_error))
        
        if args.checkpoint is not None:
            model_dir = args.checkpoint
        else:
            model_dir = 'checkpoints/neuqe'
        if ':' in args.model:
            model_ver = args.model.split(':')[1].upper()
        else:
            model_ver = 'RC'
        pred_name = '{}nn_predictor'.format(model_ver[0].lower())
        pred_model = os.path.join(model_dir, pred_name)
        pred_model = os.path.join(pred_model, 'model.best.pt')
        est_model = os.path.join(model_dir, 'm2scores.{}.pt'.format(model_ver))
        print('== NeuQE  ==\npredictor:{}\n estimator:{}'.format(pred_model, est_model))
        model = NeuQE(pred_model, est_model)
    else:
        raise NotImplementedError("{} model is not yet implemented"\
            .format(args.model))
    
    model.eval()

    return model