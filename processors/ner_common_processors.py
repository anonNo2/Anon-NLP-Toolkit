import csv
import json
from tracemalloc import start
import torch
from transformers import BertTokenizer
import pandas as pd
import copy





class EntityTagExtractor(object):

    def get_entity_bios(self,seq,id2label):
        """Gets entities from sequence.
        note: BIOS
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            # >>> get_entity_bios(seq)
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("S-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[2] = indx
                chunk[0] = tag.split('-')[1]
                chunks.append(chunk)
                chunk = (-1, -1, -1)
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks

    def get_entity_bio(self,seq,id2label):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_bio(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
                chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx

                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks

    def get_entities(self,seq,id2label,markup='bios'):
        '''
        :param seq:
        :param id2label:
        :param markup:
        :return:
        '''
        assert markup in ['bio','bios']
        if markup =='bio':
            return self.get_entity_bio(seq,id2label)
        else:
            return self.get_entity_bios(seq,id2label)

    def bert_extract_item(self,start_logits, end_logits):
        
        # start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
        # end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]

        start_pred_batch = [i[1:-1] for i in torch.argmax(start_logits, -1).cpu().numpy()]
        end_pred_batch = [i[1:-1] for i in torch.argmax(end_logits, -1).cpu().numpy()]
        result = []
        for idx in range(len(start_pred_batch)):
            start_pred = start_pred_batch[idx]
            end_pred = end_pred_batch[idx]
            S = []
            for i, s_l in enumerate(start_pred):
                if s_l == 0:
                    continue
                for j, e_l in enumerate(end_pred[i:]):
                    if s_l == e_l:
                        S.append((s_l, i, i + j))
                        break
            result.append(S)

        return result


entity_tag_extractor = EntityTagExtractor()