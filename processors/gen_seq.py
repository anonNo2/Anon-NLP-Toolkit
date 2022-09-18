""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os,io
import copy
import json
from torch.utils.data import TensorDataset
from data_augments.cls_enhancer import ClsEnhancer
from processors.common_processor_inter import DataProcessor
from processors.commmon_examples import GenDialogueInputExample, LabelInputFeatures

logger = logging.getLogger(__name__)



class AutoGenBaseProcessor(DataProcessor):


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
            return self._create_gen_examples(self._read_gen_json(os.path.join(data_dir, f"{tag}.json")), tag)
        elif f'{tag}.txt' in os.listdir(data_dir) :
            return self._create_gen_examples(self._read_gen_text(os.path.join(data_dir, f"{tag}.txt")), tag)
        
        
        else:
            raise ValueError('Invalid data directory %s' % data_dir)

    def get_labels(self,data_dir):
        """See base class."""
        return []
    ## TEMP /S
    def get_train_examples(self, data_dir):
        return self.get_examples_common(data_dir, "train")
    
    def get_dev_examples(self, data_dir):
        return self.get_examples_common(data_dir, "dev")

    def get_test_examples(self, data_dir):
        return self.get_examples_common(data_dir, "test")
    ## TEMP /E

    def _create_gen_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        lines = lines[:1000 if set_type == 'train' else 100]
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(GenDialogueInputExample(guid=guid, chapter=line))
        self.args.context[f'{set_type}_raw_data'] = lines
        return examples


    def train_collate_fn(self,batch):
        """
        batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
        max_len = max(all_lens).item()
        all_input_ids = all_input_ids[:, :max_len]
        all_attention_mask = all_attention_mask[:, :max_len]
        all_token_type_ids = all_token_type_ids[:, :max_len]
        all_labels = all_labels
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
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
        return dataset






class AutoGenDialogueProcessor(AutoGenBaseProcessor):


    def __init__(self,args):
        super().__init__(args)
        self.args = args

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
        sep_id = tokenizer.sep_token_id
        cls_id = tokenizer.cls_token_id
        sep_tok = tokenizer.sep_token
        cls_tok = tokenizer.cls_token
        features = []

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            chapter = example.chapter
            if "\r\n" in chapter:
                utterances = chapter.split("\r\n")
            else:
                utterances = chapter.split("\n")
            
            reduce_max_len_utterances = []
            reduce_max_len_utterances_toks = []
            reduce_max_len_utterances_ids = []

            for idx in range(len(utterances)):
                candi_sent = utterances[idx]
                
                if len(candi_sent) > max_utterance_length - 2:
                    candi_sent = candi_sent[:max_utterance_length - 2]
                reduce_max_len_utterances.append(candi_sent)
                # reduce_max_len_utterances_toks.append(tokenizer.tokenize(candi_sent) + [sep_tok])
                reduce_max_len_utterances_toks.append([i for i in candi_sent] + [sep_tok])
                reduce_max_len_utterances_ids.append(tokenizer.convert_tokens_to_ids([i for i in candi_sent]) + [sep_id])
            
            for step in range(1,len(reduce_max_len_utterances)):

                inputs_text_ids = reduce_max_len_utterances_ids[:step]
                inputs_text_toks = reduce_max_len_utterances_toks[:step]
                
                labels_ids = [cls_id] + reduce_max_len_utterances_ids[step]
                labels_tok = [cls_tok] + reduce_max_len_utterances_toks[step]

                history_start_index = 1
                filter_history_sent = []
                filter_history_sent_ids = []
                input_ids = [cls_id]  # 每个dialogue以[CLS]开头
                input_toks = [cls_tok]
                segment_ids = [cls_token_segment_id]
                # 逻辑是先从最后一位往前加句子，加下一句如果总数超了max_len就停止
                # （ps） gpt的 generate_text_by_input 方法里history回溯那里写错了，不能从history的头部开始回溯，应该从尾部，否则我们想要的是 BCD->E,会得到ABC->E
                for rev_idx in range(len(inputs_text_ids)-1,-1,-1):
                    this_turn_toks = inputs_text_toks[rev_idx]
                    
                    this_turn_ids = inputs_text_ids[rev_idx]
                    
                    if history_start_index + len(this_turn_ids)  > max_seq_length:
                        break
                    filter_history_sent.append(this_turn_toks)
                    filter_history_sent_ids.append(this_turn_ids)
                    history_start_index += len(this_turn_ids)
                
                filter_history_sent.reverse()
                filter_history_sent_ids.reverse()
                for his_idx in range(len(filter_history_sent)):
                    input_ids.extend(filter_history_sent_ids[his_idx])
                    input_toks.extend(filter_history_sent[his_idx])
                    segment_ids.extend(len(filter_history_sent_ids[his_idx]) * [sequence_a_segment_id])
                
                # 可以通过这两行记录src和target，调试时可用
                # content_line = json.dumps({'src':input_toks,'tgt':labels_tok},ensure_ascii=False)
                # ids_line = json.dumps({'src':input_ids,'tgt':labels_ids},ensure_ascii=False)
                input_lens = len(input_ids)
                input_mask = [1 if mask_padding_with_zero else 0] * input_lens
                

                

                if cls_token_at_end:
                    input_ids = input_ids[1:] + input_ids[:1]
                    segment_ids += segment_ids[1:] + segment_ids[:1]
                
                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - input_lens
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    labels_ids = ([-100] * (max_utterance_length - len(labels_ids))) + labels_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                    
                else:
                    input_ids += [pad_token] * padding_length
                    labels_ids += ([-100] * (max_utterance_length - len(labels_ids))) 
                    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                    segment_ids += [pad_token_segment_id] * padding_length
                    

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                
                if ex_index < 1:
                    logger.info("*** Example ***")
                    logger.info("guid: %s", example.guid)
                    logger.info("tokens: %s", " ".join([str(x) for x in input_toks]))
                    logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                    logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                    logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                    logger.info("label_tokens: %s", " ".join([str(x) for x in labels_tok]))
                    logger.info("label_ids: %s", " ".join([str(x) for x in labels_ids]))
                    

                features.append(LabelInputFeatures(input_ids=input_ids, input_mask=input_mask,
                                            segment_ids=segment_ids, label_ids=labels_ids,input_len=input_lens))

    
                
                
        return features

    



class AutoGenQAProcessor(AutoGenBaseProcessor):


    def __init__(self,args):
        super().__init__(args)
        self.args = args

    @classmethod
    def _read_gen_text(self,input_file):
        all_lines = io.open(input_file,'r').read()
        if "\r\n" in all_lines:
            chapter_data = all_lines.split("\r\n")
        else:
            chapter_data = all_lines.split("\n")
        
        
        
        return chapter_data

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
        sep_id = tokenizer.sep_token_id
        cls_id = tokenizer.cls_token_id
        sep_tok = tokenizer.sep_token
        cls_tok = tokenizer.cls_token
        features = []

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            
            chapter = example.chapter
            splitor_ = chapter.split('\t')
            question = splitor_[0]
            answer = splitor_[1][:max_utterance_length - 2]

            if '@' in question:
                questions = [i[:max_utterance_length - 2] for i in question.split('@')]
            else:
                questions = [question[:max_utterance_length - 2]]

            

            if self.args.gen_special.muti_question_strategy == 0:
                questions.sort(key=lambda x:len(x),reverse=True)
                questions = questions[:1]
            
            answer_ids = tokenizer.convert_tokens_to_ids([i for i in answer]) + [sep_id]
            for quest_str in questions:
                input_ids = [cls_id] + tokenizer.convert_tokens_to_ids([i for i in quest_str]) + [sep_id]
        
                labels_ids = [cls_id] + answer_ids
                # quest_str最后还要拼接一个sep，这个也算上
                segment_ids = [cls_token_segment_id] + (len(quest_str) + 1) * [sequence_a_segment_id]

                
                input_lens = len(input_ids)
                input_mask = [1 if mask_padding_with_zero else 0] * input_lens
                

                if cls_token_at_end:
                    input_ids = input_ids[1:] + input_ids[:1]
                    segment_ids += segment_ids[1:] + segment_ids[:1]
                
                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - input_lens
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    labels_ids = ([-100] * (max_utterance_length - len(labels_ids))) + labels_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                    
                else:
                    input_ids += [pad_token] * padding_length
                    labels_ids += ([-100] * (max_utterance_length - len(labels_ids))) 
                    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                    segment_ids += [pad_token_segment_id] * padding_length
                    

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                
                if ex_index < 1:
                    logger.info("*** Example ***")
                    logger.info("guid: %s", example.guid)
                    logger.info("tokens: %s", " ".join([str(x) for x in quest_str]))
                    logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                    logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                    logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                    logger.info("label_tokens: %s", " ".join([str(x) for x in answer]))
                    logger.info("label_ids: %s", " ".join([str(x) for x in labels_ids]))
                    

                features.append(LabelInputFeatures(input_ids=input_ids, input_mask=input_mask,
                                            segment_ids=segment_ids, label_ids=labels_ids,input_len=input_lens))

    
                
                
        return features



if __name__=='__main__':
    
    print()