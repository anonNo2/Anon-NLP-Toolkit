

from pickletools import optimize
from callback import optimizater
from callback.progressbar import ProgressBar
from executive_devices.devices_inter import ExecutiveDevice
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
import os,json,io
import numpy as np
from metrics.ner_metrics import SeqEntityScore, SpanEntityScore
from processors.ner_common_processors import entity_tag_extractor



class NerExecDevice(ExecutiveDevice):
    
    def get_eval_tags(self,model,logits,inputs):
        raise NotImplementedError()
    
    def get_model_input(self,batch,type):
        return {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3] if type != 'predict' else None}
    
    def get_metrics(self,context_config):
        raise NotImplementedError()

    
    def eval_step(self,context_config,prefix=""):

        self.eval_preparatory(context_config)
        
        args = context_config.base
        context = context_config.context
        model = context.model
        logger = context.logger
        metric = self.get_metrics(context_config)
        eval_output_dir = args.output_dir
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        eval_dataset = context.eval_dataset
        context.eval_batch_size = args.per_gpu_eval_batch_size * max(1, context.n_gpu)
        # Note that DistributedSampler samples randomly
        
        eval_dataloader = context.eval_dataloader
        # Eval!
        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {context.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        if isinstance(model, nn.DataParallel):
            model = model.module
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            eval_loss,nb_eval_steps = self.eval_batch_step(context_config,metric,batch,eval_loss,nb_eval_steps)
            pbar(step)
            
        logger.info("\n")
        eval_loss = eval_loss / nb_eval_steps
        eval_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        logger.info(f"***** Eval results {prefix} *****", )
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
        logger.info(f"***** Entity results {prefix} *****")
        for key in sorted(entity_info.keys()):
            logger.info(f"******* {key} results ********")
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            logger.info(info)
        return results

    def predict_step(self,context_config,prefix=""):

        args = context_config.base
        context = context_config.context
        model = context.model
        logger = context.logger
        
        pred_output_dir = context.output_dir
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)
        
        self.predict_preparatory(context_config)
        # Eval!
        logger.info(f"***** Running prediction {prefix} *****")
        logger.info(f"  Num examples = {len(context.test_dataset)}")
        logger.info(f"  Batch size = 1")
        results = []
        output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
        pbar = ProgressBar(n_total=len(context.test_dataloader), desc="Predicting")

        if isinstance(model, nn.DataParallel):
            model = model.module
        for step, batch in enumerate(context.test_dataloader):
            model.eval()
            json_d = self.predict_batch_step(context_config,batch,step)
            results.append(json_d)
            pbar(step)
        logger.info("\n")
        with open(output_predict_file, "w") as writer:
            for record in results:
                writer.write(json.dumps(record) + '\n')

    def train_epoch(self,context_config,pbar):
        args = context_config.base
        context = context_config.context
        model = context.model
        
        optimizer = context.optimizer
        scheduler = context.scheduler
        for step, batch in enumerate(context.train_dataloader):
            # Skip past any already trained steps if resuming training
            if context.steps_trained_in_current_epoch > 0:
                context.steps_trained_in_current_epoch -= 1
                continue
            model.train()
            loss = self.train_batch_step(context_config,batch,optimizer)
            
            pbar(step, {'loss': loss.item()})
            context.tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                context.global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and context.global_step % args.logging_steps == 0:
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        self.eval_step(context_config,args.task_name)
                if args.local_rank in [-1, 0] and args.save_steps > 0 and context.global_step % args.save_steps == 0:
                    self.save_ckpt(context_config,model)

    def train_batch_step(self,context_config,batch,optimizer):
        context = context_config.context
        model = context.model
        args = context_config.base
        
        batch = tuple(t.to(context.device) for t in batch)
        inputs = self.get_model_input(batch,'train')
        if context_config.model != "distilbert":
            # XLM and RoBERTa don"t use segment_ids
            inputs["token_type_ids"] = (batch[2] if context_config.model in ["bert", "xlnet"] else None)
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
        if context.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        if context_config.adversarial.do_adv:
            context.fgm.attack()
            loss_adv = model(**inputs)[0]
            if context.n_gpu>1:
                loss_adv = loss_adv.mean()
            loss_adv.backward()
            context.fgm.restore()
        return loss

    def eval_batch_step(self,context_config,metric,batch,eval_loss,nb_eval_steps):
        context = context_config.context
        model = context.model
        batch = tuple(t.to(context.device) for t in batch)
        with torch.no_grad():
            inputs = self.get_model_input(batch,'eval')
            if context_config.model != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if context_config.model in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = self.get_eval_tags(model,logits,inputs)
        if context.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = batch[4].cpu().numpy().tolist()
        
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(context.id2label[out_label_ids[i][j]])
                    temp_2.append(context.id2label[tags[i][j]])
        return eval_loss,nb_eval_steps

    def predict_batch_step(self,context_config,batch,step):
        
        context = context_config.context
        model = context.model
        batch = tuple(t.to(context.device) for t in batch)

        with torch.no_grad():
            inputs = self.get_model_input(batch,'predict')
            if context_config.model != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if context_config.model in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = self.get_eval_tags(model,logits,inputs)
            
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = entity_tag_extractor.get_entities(preds, context.id2label, context_config.ner_special.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([context.id2label[x] for x in preds])
        json_d['entities'] = label_entities
        return json_d

    

class SoftmaxNerExecDevice(NerExecDevice):

    def get_metrics(self,context_config):
        return SeqEntityScore(context_config.context.id2label, markup=context_config.ner_special.markup)
    
    def get_eval_tags(self,model,logits,inputs):
        tags = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        return tags

    def get_optimizer_grouped_parameters(self,context_config):
        args = context_config.base
        context = context_config.context
        model = context.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        return optimizer_grouped_parameters
    
    
class SpanNerExecDevice(NerExecDevice):


    def get_metrics(self,context_config):
        return SpanEntityScore(context_config.context.id2label)
    

    def get_eval_tags(self,model,logits,inputs):
        tags = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        return tags

    def get_model_input(self,batch,type):
        return {"input_ids": batch[0], "attention_mask": batch[1],
                      "start_positions": batch[3], "end_positions": batch[4]}

    def eval_batch_step(self,context_config,metric,batch,eval_loss,nb_eval_steps):
        context = context_config.context
        model = context.model
        args = context_config.base
        subjects = batch[6]
        batch = [batch[i].to(context.device) for i in [0,1,2,3,4]]

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "start_positions": batch[3], "end_positions": batch[4]}
            if context_config.model != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if context_config.model in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        tmp_eval_loss, start_logits, end_logits = outputs[:3]
        R = entity_tag_extractor.bert_extract_item(start_logits, end_logits)
        T = subjects
        metric.update(true_subject_batch=T, pred_subject_batch=R)
        if context.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        return eval_loss,nb_eval_steps


    def predict_batch_step(self,context_config,batch,step):

        args = context_config.base
        context = context_config.context
        model = context.model
        


        batch = tuple(t.to(context.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "start_positions": None, "end_positions": None}
            if context_config.model != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if context_config.model in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        start_logits, end_logits = outputs[:2]
        R = entity_tag_extractor.bert_extract_item(start_logits, end_logits)
        if R:
            label_entities = [[[context.id2label[x[0]], x[1], x[2]] for x in item] for item in R]
        else:
            label_entities = []
        json_d = {}
        json_d['id'] = step
        json_d['entities'] = label_entities
        return json_d

    def get_optimizer_grouped_parameters(self,context_config):
        args = context_config.base
        context = context_config.context
        model = context.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        bert_parameters = model.bert.named_parameters()
        start_parameters = model.start_fc.named_parameters()
        end_parameters = model.end_fc.named_parameters()
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.learning_rate},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': args.learning_rate},

            {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': 0.001},
            {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': 0.001},

            {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': 0.001},
            {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': 0.001},
        ]
        return optimizer_grouped_parameters
    
    

class CrfNerExecDevice(NerExecDevice):

    def get_eval_tags(self,model,logits,inputs):
        if isinstance(model, nn.DataParallel):
            model = model.module
        tags =  model.crf.decode(logits, inputs['attention_mask'])
        tags  = tags.squeeze(0).cpu().numpy().tolist()
        return tags
    
    def get_metrics(self,context_config):
        return SeqEntityScore(context_config.context.id2label, markup=context_config.ner_special.markup)
    
    
    
class BertBilstmCrfNerExecDevice(CrfNerExecDevice):

    def get_optimizer_grouped_parameters(self,context_config):
        args = context_config.base
        context = context_config.context
        model = context.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(model.bert.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        bilstm_param_optimizer = list(model.bilstm.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': context_config.ner_special.crf.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': context_config.ner_special.crf.crf_learning_rate},
            # 添加bilstm的参数优化器内容
            {'params': [p for n, p in bilstm_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': context_config.ner_special.crf.crf_learning_rate},
            {'params': [p for n, p in bilstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': context_config.ner_special.crf.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': context_config.ner_special.crf.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': context_config.ner_special.crf.crf_learning_rate}
        ]
        return optimizer_grouped_parameters


class BertCrfNerExecDevice(CrfNerExecDevice):

    def get_optimizer_grouped_parameters(self,context_config):
        args = context_config.base
        context = context_config.context
        model = context.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(model.bert.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': context_config.ner_special.crf.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': context_config.ner_special.crf.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': context_config.ner_special.crf.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': context_config.ner_special.crf.crf_learning_rate}
        ]
        return optimizer_grouped_parameters
