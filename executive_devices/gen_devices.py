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
from metrics.cls_metrics import SeqClsScore
from metrics.ner_metrics import SeqEntityScore, SpanEntityScore
from processors.ner_common_processors import entity_tag_extractor




class GenExecDevice(ExecutiveDevice):
    
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
        eval_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        logger.info(f"***** Eval results {prefix} *****", )
        info = "\n" + "\n".join([f'{k}:{v}' for k,v in results.items()])
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
        out_label_ids = np.argmax(inputs['labels'].cpu().numpy(),axis=-1).tolist()
        
        metric.update(tags,out_label_ids)
        
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
            
        preds_label = context.id2label[tags[0]]
        
        json_d = {}
        json_d['id'] = step
        json_d['tag'] = preds_label
        
        return json_d




class T5PegasusExecDevice(GenExecDevice):

    def get_metrics(self,context_config):
        return SeqClsScore(context_config.context.id2label)
    
    def get_eval_tags(self,model,logits,inputs):
        tags = np.argmax(logits.cpu().numpy(), axis=1).tolist()
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
    