


from asyncio.log import logger
import random
from statistics import mean
from data_augments.enhancer_inter import Enhancer
from processors.ner_common_processors import LabelInputExample


class ClsEnhancer(Enhancer):


    '''
    (DM,EDA,Back_Translation,Over_Sample)
    '''

    def __init__(self,context_dict):
        super().__init__()
        self.aug_params = context_dict.data_augments
        self.logger = context_dict.context.logger
        self.aug_function_dict = {
            'DM':self.dynamic_mask,
            'EDA':self.eda,
            'Back_Translation':self.back_translation,
            'Over_Sample':self.over_sample
        }
        self.labels_num_dict = {}

    def dynamic_mask(self,data_items):
        pass

    def eda(self,data_items):
        def reduce_func(new_items):
            new_items.guid = new_items.guid + '_eda'
            randint = random.randint(0, 3)
            

        pass

    def back_translation(self,data_items):
        pass

    def over_sample(self,data_items):
        def reduce_func(new_items):
            new_items.guid = new_items.guid + '_oversample'
            return new_items
        self.common_aug_process(data_items,reduce_func)
        
    

    def common_aug_process(self,data_items,reduce_func):
        baseline = self.aug_params.aug_baseline
        if baseline == 'MAX':
            baseline = max(self.labels_num_dict.values())
        elif baseline == 'MEAN':
            baseline = mean(self.labels_num_dict.values())
        
        baseline = int(baseline)
        new_data_items = []
        for label in self.labels_num_dict.keys():
            this_label_items = [i for i in data_items if i.labels == label]
            if len(this_label_items) >= baseline:
                new_data_items.extend(this_label_items[:baseline])
            else:
                for _ in range(baseline - len(this_label_items)):
                    data_item_idx = random.randint(0,len(this_label_items)-1)
                    new_items = LabelInputExample(this_label_items[data_item_idx].guid,this_label_items[data_item_idx].text_a,this_label_items[data_item_idx].labels)
                    new_items = reduce_func(new_items)
                    new_data_items.append(new_items)
                new_data_items.extend(this_label_items)
        return new_data_items


    def analysis_data_detail(self,data_items):
        all_labels = [i.labels for i in data_items]
        labels_num_dict = {}
        for label in set(all_labels):
            labels_num_dict[label] = len([i for i in all_labels if i == label])
            self.logger.info(f'{label}:{labels_num_dict[label]}')
        self.logger.info(f'总数据条数:{len(all_labels)}')
        self.labels_num_dict = labels_num_dict


    def augments_data(self,data_items):
        self.logger.info(f'是否增强:{self.aug_params.do_aug},增强方法:{self.aug_params.aug_function},增强目标基准线:{self.aug_params.aug_baseline}')
        if not self.aug_params.do_aug:
            return data_items
        self.logger.info(f'增强前数据分布:')
        self.analysis_data_detail(data_items)
        self.logger.info('=' * 30)
        self.logger.info('开始进行数据增强')

        reduce_data_items = self.aug_function_dict[self.aug_params.aug_function](data_items)
        self.logger.info('=' * 30)
        self.logger.info(f'增强后数据分布:')
        self.analysis_data_detail(reduce_data_items)
        return reduce_data_items
