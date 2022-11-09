""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os,io
import copy
import json
from torch.utils.data import TensorDataset
from processors.common_processor_inter import DataProcessor
from processors.commmon_examples import LabelInputFeatures,LabelInputExample
import jieba
import jieba.posseg as pseg
logger = logging.getLogger(__name__)

class AutoNerProcessor(DataProcessor):


    def __init__(self,args):
        super().__init__()
        self.args = args

    '''
    auto recongize the data format and change it into datasets
    format 1: bmes
    format 2: json

    '''
    def get_examples_common(self,data_dir,tag):
        if f'{tag}.json' in os.listdir(data_dir):
            return self._create_json_examples(self._read_ner_json(os.path.join(data_dir, f"{tag}.json")), tag)
        elif f'{tag}.char.bmes' in os.listdir(data_dir):
            return self._create_bmes_examples(self._read_ner_text(os.path.join(data_dir, f"{tag}.char.bmes")), tag)
        else:
            raise ValueError('Invalid data directory %s' % data_dir)

    def get_labels(self,data_dir):
        """See base class."""
        return io.open(os.path.join(data_dir, 'labels.txt'), encoding='utf-8').read().strip().split("\n")

    def get_train_examples(self, data_dir):
        return self.get_examples_common(data_dir, "train")
    
    def get_dev_examples(self, data_dir):
        return self.get_examples_common(data_dir, "dev")

    def get_test_examples(self, data_dir):
        return self.get_examples_common(data_dir, "test")

    
    def _create_bmes_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(LabelInputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

    def _create_json_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(LabelInputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

    def convert_examples_to_features(self,examples,label_list,max_seq_length,tokenizer,max_utterance_length=64,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            if isinstance(example.text_a,list):
                example.text_a = "".join(example.text_a)
            tokens = [i for i in example.text_a]
            label_ids = [label_map[x] for x in example.labels]
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            
            label_ids += [label_map['O']]
            



            need_nature_feature = False
            if self.args.get('nature_embed') and self.args.nature_embed.get('do_nature_embed'):
                nature_list = io.open(os.path.join(self.args.base.model_name_or_path,'nature_list.txt'),'r').read().split('\n')
                nature_dict = {nature_list[i]:i for i in range(len(nature_list))}
                need_nature_feature = True
                nature_list = []
                words_nature = pseg.cut(''.join(tokens))
                expand_w_n = [[word, flag] for word,flag in words_nature]
                for item in expand_w_n:
                    chars = item[0]
                    nature = item[1]
                    
                    nature_list.append(f'B-{nature}')
                    
                    for _ in range(1,len(chars)):
                        nature_list.append(f'I-{nature}')

            tokens += [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            if need_nature_feature:
                nature_list.append('B-END')

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [label_map['O']]
                segment_ids += [cls_token_segment_id]
                if need_nature_feature:
                    nature_list += ['B-START']
            else:
                tokens = [cls_token] + tokens
                label_ids = [label_map['O']] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                if need_nature_feature:
                    nature_list = ['B-START'] + nature_list

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if need_nature_feature:
                nature_ids = [nature_dict[i] for i in nature_list]
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(label_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token] * padding_length) + label_ids
                if need_nature_feature:
                    nature_ids = [nature_dict.get('B-PAD')] * padding_length + nature_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token] * padding_length
                if need_nature_feature:
                    nature_ids += [nature_dict.get('B-PAD')] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            
            if need_nature_feature:
                    assert len(nature_ids) == max_seq_length
            if ex_index < 5  and self.args.base.print_data_examples:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
                if need_nature_feature:
                    logger.info("nature_ids: %s", " ".join([str(x) for x in nature_ids]))

            features.append(LabelInputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                        segment_ids=segment_ids, label_ids=label_ids,nature_feature_ids=nature_ids if need_nature_feature else None))
        return features

    def train_collate_fn(self,batch):
        """
        batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        if self.args.get('nature_embed') and self.args.nature_embed.get('do_nature_embed'):
            all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels,all_nature = map(torch.stack, zip(*batch))
            max_len = max(all_lens).item()
            all_input_ids = all_input_ids[:, :max_len]
            all_nature_ids = all_nature[:,:max_len]
            all_attention_mask = all_attention_mask[:, :max_len]
            all_token_type_ids = all_token_type_ids[:, :max_len]
            all_labels = all_labels[:,:max_len]
            return all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_lens,all_nature_ids
        else:
            all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
            max_len = max(all_lens).item()
            all_input_ids = all_input_ids[:, :max_len]
            all_attention_mask = all_attention_mask[:, :max_len]
            all_token_type_ids = all_token_type_ids[:, :max_len]
            all_labels = all_labels[:,:max_len]
            return all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_lens
    
    def dev_collate_fn(self,batch):
        return self.train_collate_fn(batch)

    def predict_collate_fn(self,batch):
        return self.train_collate_fn(batch)
    
    def reduce_to_tensorDataset(self,features,data_type):

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        if self.args.get('nature_embed') and self.args.nature_embed.get('do_nature_embed'):
            all_nature_ids = torch.tensor([f.nature_feature_ids for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids,all_nature_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
        return dataset




if __name__=='__main__':
    print()
    