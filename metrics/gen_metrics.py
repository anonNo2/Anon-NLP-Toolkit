import torch
from collections import Counter
from processors.ner_common_processors import entity_tag_extractor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class SeqGenScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.name_list = [self.id2label[i] for i in range(len(self.id2label))]
        self.reset()

    def reset(self):
        self.labels = []
        self.predict = []

    

    def result(self):

       return classification_report(self.labels, self.predict, digits=4, target_names=self.name_list,output_dict=True)


    def update(self, labels, preds):
        '''
        labels: [0,1,0,1,....]
        preds: [1,0,1,0,.....]

        :param labels:
        :param preds:
        :return:
        
        '''
        self.labels.extend(labels)
        self.predict.extend(preds)
        

