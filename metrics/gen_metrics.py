import torch
from collections import Counter
from callback.progressbar import ProgressBar
from metrics.gen_metrics_tools import cal_metrics
from processors.ner_common_processors import entity_tag_extractor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io,os,json
from tqdm import tqdm




class GenScore(object):
    def __init__(self, context_dict):
        self.context_dict = context_dict
        self.context = context_dict.context
        self.logger = self.context.logger
        self.correct_nums = 0
        self.all_nums = 0
        self.reset()

    def reset(self):
        self.correct_nums = 0
        self.all_nums = 0

    

    def result(self):

       return {'acc':self.correct_nums * 1.0 / self.all_nums}


    def update(self, labels, preds,ignore_index=-100):
        '''
        labels: [0,1,0,1,....]
        preds: [1,0,1,0,.....]

        :param labels:
        :param preds:
        :return:
        
        '''
        n_correct, n_word = self.calculate_acc(logit=preds,labels=labels,ignore_index=ignore_index)

        self.correct_nums += n_correct
        self.all_nums += n_word
    

    def calculate_acc(self,logit, labels, ignore_index=-100):
        logits_contiguous = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels_contiguous = labels[..., 1:].contiguous().view(-1)

        _, logits_contiguous = logits_contiguous.max(dim=-1)  # 对于每条数据，返回最大的index
        # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
        non_pad_mask = labels_contiguous.ne(ignore_index)
        n_correct = logits_contiguous.eq(labels_contiguous).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        return n_correct, n_word

    def calculate_machine_metrics(self,datafile_path,output_dir,data_type):
        metric_scores = cal_metrics(datafile_path)
        info = "\n" + "\n".join([f'{k}:{v}' for k,v in metric_scores.items()])
        self.logger.info(info)
        io.open(os.path.join(output_dir,f'{data_type}_machine_scores.json'),'w').write(json.dumps(metric_scores,ensure_ascii=False,indent=4))


    

    


class GPTQAGenScore(GenScore):
    def __init__(self, context_dict):
        super().__init__(context_dict)
        self.context_dict = context_dict
        self.context = context_dict.context
        self.logger = self.context.logger
        self.correct_nums = 0
        self.all_nums = 0
        self.gen_max_len =  self.context_dict.base.eval_max_seq_len
        self.reset()


    

    
    def get_machine_metric_datas(self,model,output_dir,data_type):
        '''
        生成机器指标（bleu,gleu,rouge）所需的数据
        
        '''
        model_to_gen = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        tokenizer = self.context.tokenizer
        
        model_to_gen.eval()
        data_loader = self.context[f'{data_type}_dataloader']
        results = []
        pbar = ProgressBar(n_total=len(data_loader), desc=f"Evaluating Machine Metrics-ckpt:{self.context.global_step}")
        
        for step, batch in enumerate(data_loader):
            
            batch = tuple(t.to(self.context.device) for t in batch)
            input_ids = batch[5]

            
            
            output = model_to_gen.generate(input_ids,decoder_start_token_id=tokenizer.cls_token_id,
                                                eos_token_id=tokenizer.sep_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                top_k=1,
                                                max_new_tokens=self.gen_max_len).cpu().numpy()
            gen_texts = [tokenizer.decode(j for j in i[1:-1] if j not in [tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.pad_token_id,tokenizer.unk_token_id]).replace(' ','') for i in output]
            answers_texts = [tokenizer.decode(j for j in i[1:-1] if j not in [tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.pad_token_id,tokenizer.unk_token_id,-100]).replace(' ','') for i in batch[6]]
            for idx in range(len(gen_texts)):
                results.append({'ori':answers_texts[idx],'gen':gen_texts[idx]})
            pbar(step)
                
        
        data_file_path = os.path.join(output_dir,f'{data_type}_machine_metric_data.json')
        io.open(data_file_path,'w').write(json.dumps(results,ensure_ascii=False, indent=4))
        return data_file_path
    

    def generate_text_by_input(self,text,history_text,model,tokenizer):
        
        history_text.append(text)
        temp_choosen_h_mark = []

        history = [tokenizer.encode(i, add_special_tokens=False) for i in history_text]


        input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
        history_start_index = 1
        filter_history_sent_ids = []
        for rev_idx in range(len(history)-1,-1,-1):
            
            this_turn_ids = history[rev_idx][:self.context_dict.base.eval_max_utterance_len] + [tokenizer.sep_token_id]
            
            if history_start_index + len(this_turn_ids)  > self.context_dict.base.eval_max_seq_len:
                break
            
            filter_history_sent_ids.append(this_turn_ids)
            history_start_index += len(this_turn_ids)
        filter_history_sent_ids.reverse()

        for sent_ids in filter_history_sent_ids:
            input_ids.extend(sent_ids)
            temp_choosen_h_mark.append(tokenizer.convert_ids_to_tokens(sent_ids))
            
            

        input_ids = torch.tensor(input_ids).long().to(model.device)
        input_ids = input_ids.unsqueeze(0)

        # 最多生成max_len个token
        output = model.generate(input_ids,decoder_start_token_id=tokenizer.cls_token_id,eos_token_id=tokenizer.sep_token_id,top_k=1,max_new_tokens=self.gen_max_len).cpu().numpy()[0]
        
        text = ''.join([i for i in tokenizer.decode(output[1:-1])]).replace(' ', '')
        return text



class T5QAGenScore(GenScore):
    def __init__(self, context_dict):
        super().__init__(context_dict)
        self.context_dict = context_dict
        self.context = context_dict.context
        self.logger = self.context.logger
        self.correct_nums = 0
        self.all_nums = 0
        self.gen_max_len =  self.context_dict.base.eval_max_seq_len
        self.reset()
    

    def generate_text_by_input(self,text,history_text,model,tokenizer):
        
        history_text.append(text)
        temp_choosen_h_mark = []

        history = [tokenizer.encode(i, add_special_tokens=False) for i in history_text]


        input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
        history_start_index = 1
        filter_history_sent_ids = []
        for rev_idx in range(len(history)-1,-1,-1):
            
            this_turn_ids = history[rev_idx][:self.context_dict.base.eval_max_utterance_len] + [tokenizer.sep_token_id]
            
            if history_start_index + len(this_turn_ids)  > self.context_dict.base.eval_max_seq_len:
                break
            
            filter_history_sent_ids.append(this_turn_ids)
            history_start_index += len(this_turn_ids)
        filter_history_sent_ids.reverse()

        for sent_ids in filter_history_sent_ids:
            input_ids.extend(sent_ids)
            temp_choosen_h_mark.append(tokenizer.convert_ids_to_tokens(sent_ids))
            
            

        input_ids = torch.tensor(input_ids).long().to(model.device)
        input_ids = input_ids.unsqueeze(0)

        # 最多生成max_len个token
        output = model.generate(input_ids,decoder_start_token_id=tokenizer.cls_token_id,eos_token_id=tokenizer.sep_token_id,top_k=1,max_new_tokens=self.gen_max_len).cpu().numpy()[0]
        
        text = ''.join([i for i in tokenizer.decode(output[1:-1])]).replace(' ', '')
        return text

    

    def get_machine_metric_datas(self,model,output_dir,data_type):
        '''
        生成机器指标（bleu,gleu,rouge）所需的数据
        
        '''
        model_to_gen = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        tokenizer = self.context.tokenizer
        
        model_to_gen.eval()
        data_loader = self.context[f'{data_type}_dataloader']
        results = []
        pbar = ProgressBar(n_total=len(data_loader), desc=f"Evaluating Machine Metrics-ckpt:{self.context.global_step}")
        for step, batch in enumerate(data_loader):
            
            batch = tuple(t.to(self.context.device) for t in batch)
            input_ids = batch[0]
            output = model.generate(input_ids,decoder_start_token_id=tokenizer.cls_token_id,eos_token_id=tokenizer.sep_token_id,top_k=1,max_new_tokens=self.gen_max_len).cpu().numpy()
            gen_texts = [tokenizer.decode(j for j in i[1:-1] if j not in [tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.pad_token_id,tokenizer.unk_token_id]).replace(' ','') for i in output]
            answers_texts = [tokenizer.decode(j for j in i[1:-1] if j not in [tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.pad_token_id,tokenizer.unk_token_id,-100]).replace(' ','') for i in batch[3]]
            for idx in range(len(gen_texts)):
                results.append({'ori':answers_texts[idx],'gen':gen_texts[idx]})
            pbar(step)
                
        
        data_file_path = os.path.join(output_dir,f'{data_type}_machine_metric_data.json')
        io.open(data_file_path,'w').write(json.dumps(results,ensure_ascii=False, indent=4))
        return data_file_path




class T5DialogueGenScore(GenScore):
    def __init__(self, context_dict):
        super().__init__(context_dict)
        self.context_dict = context_dict
        self.context = context_dict.context
        self.logger = self.context.logger
        self.correct_nums = 0
        self.all_nums = 0
        self.gen_max_len =  self.context_dict.base.eval_max_utterance_len
        self.reset()


    
    def generate_text_by_input(self,text,history_text,model,tokenizer):
        
        history_text.append(text)
        temp_choosen_h_mark = []

        history = [tokenizer.encode(i, add_special_tokens=False) for i in history_text]


        input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
        history_start_index = 1
        filter_history_sent_ids = []
        for rev_idx in range(len(history)-1,-1,-1):
            
            this_turn_ids = history[rev_idx][:self.context_dict.base.eval_max_utterance_len] + [tokenizer.sep_token_id]
            
            if history_start_index + len(this_turn_ids)  > self.context_dict.base.eval_max_seq_len:
                break
            
            filter_history_sent_ids.append(this_turn_ids)
            history_start_index += len(this_turn_ids)
        filter_history_sent_ids.reverse()

        for sent_ids in filter_history_sent_ids:
            input_ids.extend(sent_ids)
            temp_choosen_h_mark.append(tokenizer.convert_ids_to_tokens(sent_ids))
            
            

        input_ids = torch.tensor(input_ids).long().to(model.device)
        input_ids = input_ids.unsqueeze(0)

        # 最多生成max_len个token
        output = model.generate(input_ids,decoder_start_token_id=tokenizer.cls_token_id,eos_token_id=tokenizer.sep_token_id,top_k=1,max_new_tokens=self.gen_max_len).cpu().numpy()[0]
        
        text = ''.join(tokenizer.decode(output[1:-1])).replace(' ', '')
        return text

    

    def get_machine_metric_datas(self,model,output_dir,data_type):
        '''
        生成机器指标（bleu,gleu,rouge）所需的数据
        A_ori->B_gen
        A_ori,B_ori->C_gen
        A_ori,B_ori,C_ori->D_gen

        最后输出
        B_ori,B_gen
        C_ori,C_gen
        D_ori,D_gen
        '''
        model_to_gen = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        self.context.tokenizer
        
        raw_paras = self.context_dict.context[f'{data_type}_raw_data']
        results = []
        for single_para_idx in tqdm(range(len(raw_paras))):
            single_para = raw_paras[single_para_idx]
            single_lines = single_para.split('\t')
            lines_nums = len(single_lines)
            for step in range(1,lines_nums):
                inputs_text = single_lines[:step]
                history = inputs_text[:-1]
                text = inputs_text[-1]
                gen_text_tok = self.generate_text_by_input(text, history, model_to_gen, self.context.tokenizer)
                gen_text = "".join(gen_text_tok)
                ori_text = single_lines[step]
                results.append({'ori':ori_text,'gen':gen_text})
        
        data_file_path = os.path.join(output_dir,f'{data_type}_machine_metric_data.json')
        io.open(data_file_path,'w').write(json.dumps(results,ensure_ascii=False, indent=4))
        return data_file_path

    

    





