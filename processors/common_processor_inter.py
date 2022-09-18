
import pandas as pd
import json,io,os


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self,data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_gen_json(self,input_file):
        all_contents = io.open(input_file,'r').read()
        chapter_data = json.loads(all_contents)
        return chapter_data


    @classmethod
    def _read_gen_text(self,input_file):
        all_lines = io.open(input_file,'r').read()
        if "\r\n" in all_lines:
            chapter_data = all_lines.split("\r\n\r\n")
        else:
            chapter_data = all_lines.split("\n\n")
        
        return chapter_data

    @classmethod
    def _read_ner_text(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})
        return lines

    @classmethod
    def _read_ner_json(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label',None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key,value in label_entities.items():
                        for sub_name,sub_index in value.items():
                            for start_index,end_index in sub_index:
                                assert  ''.join(words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-'+key
                                else:
                                    labels[start_index] = 'B-'+key
                                    labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_cls_dataframe(self,cls_dataframe):
        lines = []
        for idx in range(len(cls_dataframe)):
            text = cls_dataframe['text'][idx]
            label = cls_dataframe['label'][idx] if 'label' in cls_dataframe else -1
            lines.append({'text':text,'label':label})
        return lines
    
    @classmethod
    def _read_cls_json(self,input_file):
        
        cls_dataframe = pd.read_json(input_file)
        return self._read_cls_dataframe(cls_dataframe)
    
    @classmethod
    def _read_cls_xls(self,input_file):
        
        cls_dataframe = pd.read_excel(input_file)
        return self._read_cls_dataframe(cls_dataframe)
    
    @classmethod
    def _read_cls_csv(self,input_file):
        
        cls_dataframe = pd.read_csv(input_file)
        return self._read_cls_dataframe(cls_dataframe)
