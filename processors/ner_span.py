""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """
from re import sub
import torch
import logging
import os,io
import copy
import json

from processors.common_processor_inter import DataProcessor
from processors.ner_common_processors import EntityTagExtractor
from .commmon_examples import SubjectInputExample,SubjectInputFeature
from torch.utils.data import TensorDataset
logger = logging.getLogger(__name__)

ete = EntityTagExtractor()


class AutoSpanProcessor(DataProcessor):


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
        return io.open(os.path.join(data_dir, 'span_labels.txt'), encoding='utf-8').read().strip().split("\n")

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
            text_a = line['words']
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            subject = ete.get_entities(labels,id2label=None,markup='bios')
            examples.append(SubjectInputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

    def _create_json_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = line['labels']
            subject = ete.get_entities(labels,id2label=None,markup='bios')
            examples.append(SubjectInputExample(guid=guid, text_a=text_a, subject=subject))
        return examples





    def train_collate_fn(self,batch):
        """
        batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        

        all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens = map(torch.stack, zip(*batch))

        max_len = max(all_lens).item()
        all_input_ids = all_input_ids[:, :max_len]
        all_input_mask = all_input_mask[:, :max_len]
        all_segment_ids = all_segment_ids[:, :max_len]
        all_start_ids = all_start_ids[:,:max_len]
        all_end_ids = all_end_ids[:, :max_len]

        return all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens


    def dev_collate_fn(self,batch):
        '''
        评估时之前的步骤并没有讲feature处理为dataset，因为span有subjects,所以dev模式到这里时是list
        '''
        all_lens = [f.input_len for f in batch]
        max_len = max(all_lens)
        all_input_ids = torch.tensor([f.input_ids[:max_len] for f in batch], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask[:max_len] for f in batch], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids[:max_len] for f in batch], dtype=torch.long)
        all_start_ids = torch.tensor([f.start_ids[:max_len] for f in batch], dtype=torch.long)
        all_end_ids = torch.tensor([f.end_ids[:max_len] for f in batch], dtype=torch.long)
        subjects = [f.subjects for f in batch]
        return all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens,subjects

    def predict_collate_fn(self,batch):
        return self.train_collate_fn(batch)


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
        label2id = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            textlist = example.text_a
            subjects = example.subject
            if isinstance(textlist,list):
                textlist = " ".join(textlist)
            tokens = tokenizer.tokenize(textlist)
            start_ids = [0] * len(tokens)
            end_ids = [0] * len(tokens)
            subjects_id = []
            for subject in subjects:
                label = subject[0]
                start = subject[1]
                end = subject[2]
                start_ids[start] = label2id[label]
                end_ids[end] = label2id[label]
                subjects_id.append((label2id[label], start, end))
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                start_ids = start_ids[: (max_seq_length - special_tokens_count)]
                end_ids = end_ids[: (max_seq_length - special_tokens_count)]
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
            tokens += [sep_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            if cls_token_at_end:
                tokens += [cls_token]
                start_ids += [0]
                end_ids += [0]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                start_ids = [0]+ start_ids
                end_ids = [0]+ end_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                start_ids = ([0] * padding_length) + start_ids
                end_ids = ([0] * padding_length) + end_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                start_ids += ([0] * padding_length)
                end_ids += ([0] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(start_ids) == max_seq_length
            assert len(end_ids) == max_seq_length

            if ex_index < 5  and self.args.base.print_data_examples:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
                logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))

            features.append(SubjectInputFeature(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    start_ids=start_ids,
                                    end_ids=end_ids,
                                    subjects=subjects_id,
                                    input_len=input_len))
        return features

    
    def reduce_to_tensorDataset(self,features,data_type):

        if data_type == 'dev':
            return features
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
        all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens)
            
        

        
        return dataset








