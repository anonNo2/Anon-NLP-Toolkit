'''
Author: anon
Date: 2022-07-21 19:50:38
LastEditors: anon
LastEditTime: 2022-07-21 20:01:13
FilePath: /Anon-NLP-Toolkit/Main_Entry.py
Description: 

Copyright (c) 2022 by anon/Ultrapower, All Rights Reserved. 
'''
import glob
import logging
import os,io
import json
from random import shuffle
import time
from typing import Any
from shutil import copyfile
import munch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from callback.t5_tokenizer import T5PegasusTokenizer
from executive_devices.cls_devices import SoftmaxClsExecDevice
from executive_devices.gen_devices import GPTQAExecDevice, T5PegasusExecDevice, T5QAExecDevice
from executive_devices.ner_devices import BertBilstmCrfNerExecDevice,BertCrfNerExecDevice,SoftmaxNerExecDevice,SpanNerExecDevice
from models.bert_for_cls import BertForNormalCls
from processors.cls_seq import AutoClsProcessor
from models.bert_for_ner import BertBiLSTMCrfForNer, BertCrfForNer, BertSoftmaxForNer, BertSpanForNer
from processors.gen_seq import AutoGenDecoderQAProcessor, AutoGenEncoderDecoderDialogueProcessor, AutoGenEncoderDecoderQAProcessor
from processors.ner_seq import AutoNerProcessor
from processors.ner_span import AutoSpanProcessor
from tools.common import seed_everything,json_to_text
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from transformers.models.mt5.modeling_mt5 import MT5Config
from tools.config import load_cfg
from tools.logger_entry import GlobalLogger
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config,OpenAIGPTLMHeadModel
Main_Struct_Dict = {
    'NER':{
        'bert':{
            'bertBiLSTMCrf': (BertConfig, BertBiLSTMCrfForNer,BertTokenizer,AutoNerProcessor,BertBilstmCrfNerExecDevice),
            'bertCrf': (BertConfig, BertCrfForNer, BertTokenizer,AutoNerProcessor,BertCrfNerExecDevice),
            'bertSoftmax': (BertConfig, BertSoftmaxForNer, BertTokenizer,AutoNerProcessor,SoftmaxNerExecDevice),
            'bertSpan': (BertConfig, BertSpanForNer,BertTokenizer,AutoSpanProcessor,SpanNerExecDevice),
        },
    },
    'CLS':{
        'bert':{
            'bertSoftmax':(BertConfig, BertForNormalCls, BertTokenizer,AutoClsProcessor,SoftmaxClsExecDevice),
        }
    },
    'GEN':{
        'T5-Pegasus': {
            'T5-ConditionalGeneration-Dialogue':(MT5Config, MT5ForConditionalGeneration, T5PegasusTokenizer,AutoGenEncoderDecoderDialogueProcessor,T5PegasusExecDevice),
            'T5-ConditionalGeneration-QA':(MT5Config, MT5ForConditionalGeneration, T5PegasusTokenizer,AutoGenEncoderDecoderQAProcessor,T5QAExecDevice)
        },
        'GPT': {
            'GPT-ConditionalGeneration-QA': (GPT2Config,GPT2LMHeadModel,BertTokenizer,AutoGenDecoderQAProcessor,GPTQAExecDevice)
        }
    }
}






class MainController():

    def __init__(self,conf_path) :
        self.conf_path = conf_path
        self.config = load_cfg(conf_path)
        if not os.path.exists(self.config.base.output_dir):
            os.makedirs(self.config.base.output_dir)
        
        self.importance_ref = Main_Struct_Dict[self.config.type][self.config.model][self.config.struct]
        self.data_preduce_func = {
            'NER':self.ner_task_data_prepare,
            'CLS':self.cls_task_data_prepare,
            'GEN':self.generate_task_data_prepare
        }
        self.config['context'] = munch.Munch({})

    def load_examples(self, data_type='train'):
        args = self.config
        processor =args.context.processor
        
        label_list = processor.get_labels(args.base.data_dir)
        if data_type == 'train':
            examples = processor.get_train_examples(args.base.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.base.data_dir)
        else:
            examples = processor.get_test_examples(args.base.data_dir)
        
        return examples,label_list

    
    def covert_examples_to_tensors(self, task, tokenizer,examples, label_list, data_type='train',shard_start_idx=0,shard_size=0):
        '''
        load datafile and cache it
        加载并缓存数据文件
        '''
        args = self.config
        logger = self.logger
        if args.base.local_rank not in [-1, 0] :
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        processor =args.context.processor
        tag = args.struct
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.base.data_dir, '{}-{}_cached_{}-{}_{}_{}_{}'.format(
            shard_start_idx,
            shard_size,
            tag,
            data_type,
            list(filter(None, args.base.model_name_or_path.split('/'))).pop(),
            str(args.base.train_max_seq_len if data_type == 'train' else args.base.eval_max_seq_len),
            str(task)))
        if os.path.exists(cached_features_file) and not args.base.overwrite_cache:
            logger.info(
                f"从缓存中加载数据 {cached_features_file}",
                f"Loading features from cached file {cached_features_file}"
                )
            features = torch.load(cached_features_file)
        else:
            logger.info(
                f"从数据集文件中加载数据 {args.base.data_dir}",
                f"Creating features from dataset file at {args.base.data_dir}"

            )

            features = processor.convert_examples_to_features(examples=examples,
                                                    tokenizer=tokenizer,
                                                    label_list=label_list,
                                                    max_seq_length=args.base.train_max_seq_len if data_type == 'train' \
                                                        else args.base.eval_max_seq_len,
                                                    max_utterance_length=args.base.get('train_max_utterance_len',0) if data_type == 'train' \
                                                        else args.base.get('eval_max_utterance_len',0),
                                                    cls_token_at_end=bool(args.model in ["xlnet"]),
                                                    pad_on_left=bool(args.model in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model in ['xlnet'] else 0,
                                                    )
            if args.base.local_rank in [-1, 0]:
                
                logger.info(f"将特征文件保存在{cached_features_file}",f"Saving features into cached file {cached_features_file}")
                # 暂时，因为这个存取似乎比解析还慢
                torch.save(features, cached_features_file)
        if args.base.local_rank == 0 :
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        # Convert to Tensors and build dataset
        return processor.reduce_to_tensorDataset(features,data_type)
        
    
    def common_preduce_step(self):
        '''
        做前置的准备步骤，不同的模型，不同的任务，在进行训练之前的公共的准备步骤
        Do pre-preparation steps, different models, different tasks, common preparation steps before training
        '''
        args = self.config
        tag = args.model + '_' + args.struct
        if not os.path.exists(args.base.output_dir):
            os.mkdir(args.base.output_dir)
        args.context.output_dir = os.path.join(args.base.output_dir,tag,args.base.task_name)
        if not os.path.exists(args.context.output_dir):
            os.makedirs(args.context.output_dir)
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        log_file= os.path.join(self.config.context.output_dir,f'{tag}-{self.config.type}-{time_}.log')
        self.logger = GlobalLogger(language=self.config.language,log_file=log_file)
        self.logger.print_config(self.config)
        args.context.logger = self.logger
        copyfile(self.conf_path,self.config.context.output_dir + f'/train_config.yml')



        if os.path.exists(args.context.output_dir) and os.listdir(
                args.context.output_dir) and args.base.do_train and not args.base.overwrite_output_dir:
            
            if not args.base.need_init_model_evaluate:
                raise ValueError(
                    self.logger.glo(
                    "输出目录({})已经存在了，你可以在配置中使用overwrite_output_dir=true来进行覆盖",
                    "Output directory ({}) already exists and is not empty. Use overwrite_output_dir = true to overcome.")
                    .format(
                        args.context.output_dir
                        ))
        # 如果需要，设置远程调试
        # Setup distant debugging if needed
        if args.base.server_ip and args.base.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(args.base.server_ip, args.base.server_port), redirect_output=True)
            ptvsd.wait_for_attach()
        # 配置gpu
        # Setup CUDA, GPU & distributed training
        if args.base.local_rank == -1 or args.base.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.base.no_cuda else "cpu")
            args.context.n_gpu = torch.cuda.device_count()
        else:  
            # 初始化分布式训练
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.base.local_rank)
            device = torch.device("cuda", args.base.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.context.n_gpu = 1
        args.context.device = device
        self.logger.info(
            f"Process rank: {args.base.local_rank}, device: {device}, n_gpu: {args.context.n_gpu}, distributed training: {bool(args.base.local_rank != -1)}, 16-bits training: {args.base.fp16}"
            )
        # 设置种子
        # Set seed
        seed_everything(args.base.seed)
    

    def load_processor_and_config(self):
        args = self.config
        config_class, _, _,preprocessor_class,_ = self.importance_ref
        args.context.processor = preprocessor_class(args)
        label_list = args.context.processor.get_labels(args.base.data_dir)
        args.context.label_list = label_list
        args.context.config = config_class.from_pretrained(args.base.model_name_or_path,num_labels=len(label_list),)
        

    def load_model(self):
        args = self.config
        context = self.config.context
        logger = self.logger
        # 在分布式训练时，保证只有主进程加载预训练语言模型和词表
        # Make sure only the first process in distributed training will download model & vocab
        if args.base.local_rank not in [-1, 0]:
            torch.distributed.barrier() 
 

        _, model_class, tokenizer_class,_,exe_device_class = self.importance_ref
        
        args.context.tokenizer = tokenizer_class.from_pretrained(args.base.model_name_or_path,
                                                    do_lower_case=args.base.do_lower_case,)

        model_path = args.base.model_name_or_path

        if args.base.continue_train:
            
            last_output_dir = context.output_dir
            ckpt_dir = os.path.join(last_output_dir,'checkpoint_dir')
            shard_dir = os.path.join(last_output_dir,'shard_model_dir')
            max_global_step = 0
            load_dir = ''
            for _, dirs, _ in os.walk(ckpt_dir):
                for dir in dirs:
                    ckpt_step = int(dir.replace('checkpoint-',''))
                    if ckpt_step > max_global_step:
                        max_global_step = ckpt_step
                        load_dir = os.path.join(ckpt_dir,dir)
            
            for _, dirs, _ in os.walk(shard_dir):
                for dir in dirs:
                    shard_step = int(dir.split('-step#')[1].replace('#',''))
                    if shard_step > max_global_step:
                        max_global_step = shard_step
                        load_dir = os.path.join(shard_dir,dir)
            
            
            if load_dir:
                logger.info('注意：满足继续训练条件，将开启继续训练模式')
                model_path = load_dir
                context.continue_step = max_global_step
                self.logger.info(f" 加载模型将更为为:{load_dir}")
                args.context.reload_model_path = model_path

            

        
        args.context.model = model_class.from_pretrained(model_path, config= args.context.config)
        args.context.model_class = model_class
        args.context.exe_device = exe_device_class()
        
        # 在分布式训练时，保证只有主进程加载预训练语言模型和词表
        # Make sure only the first process in distributed training will download model & vocab
        if args.base.local_rank == 0:
            torch.distributed.barrier()
        
        args.context.model.to(args.context.device)
        
    def load_dev_and_predict_datasets(self):
        args = self.config
        
        if args.base.do_eval or args.base.evaluate_during_training :
            examples,label_list = self.load_examples(data_type='dev')
            args.context.eval_dataset = self.covert_examples_to_tensors(args.base.task_name, args.context.tokenizer,examples,label_list, data_type='dev')
        if args.base.do_predict:
            examples,label_list = self.load_examples(data_type='predict')
            args.context.test_dataset = self.covert_examples_to_tensors(args.base.task_name, args.context.tokenizer,examples,label_list, data_type='predict')
    


    def load_datasets_complete(self):
        args = self.config
        if args.base.do_train:

            examples,label_list = self.load_examples(data_type='train')
            self.config.context.shard_start = 0
            self.config.context.shard_end = len(examples)
            self.config.context.all_examples_size = len(examples)
            args.context.train_dataset = self.covert_examples_to_tensors(args.base.task_name, args.context.tokenizer,examples,label_list, data_type='train')
        self.load_dev_and_predict_datasets()
       
                  
    
    def ner_task_data_prepare(self):
        '''
        ner 数据前置准备工作
        ner dataset prepare
        '''
        args = self.config
        label_list = args.context.label_list
        args.context.label_list = label_list
        args.context.id2label = {i: label for i, label in enumerate(label_list)}
        args.context.label2id = {label: i for i, label in enumerate(label_list)}
        args.context.num_labels = len(label_list)
        args.context.config.loss_type = args.ner_special.loss_type
        args.context.config.soft_label = True
        
        
    
    def cls_task_data_prepare(self):
        '''
        分类 数据前置准备工作
        classification dataset prepare
        '''
        args = self.config
        label_list = args.context.label_list
        args.context.label_list = label_list
        args.context.id2label = {i: label for i, label in enumerate(label_list)}
        args.context.label2id = {label: i for i, label in enumerate(label_list)}
        args.context.num_labels = len(label_list)
        args.context.config.loss_type = args.cls_special.loss_type
        pass

    def generate_task_data_prepare(self):
        '''
        生成 数据前置准备工作
        generate dataset prepare
        '''
        pass

    def main_running_step(self):
        
        args = self.config
        logger = self.logger
        model = self.config.context.model
        tokenizer = self.config.context.tokenizer
        
        if args.base.do_train:
            global_step, tr_loss = args.context.exe_device.train_step(self.config)
            self.logger.info(f" global_step = {global_step}, average loss = {tr_loss}")
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.base.do_train and (args.base.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            save_train_final_results = os.path.join(args.context.output_dir,'shard_model_dir',f'shard_#{args.context.shard_start}-{args.context.shard_end}#-{args.context.all_examples_size}-step#{args.context.global_step}#')
            if not os.path.exists(save_train_final_results) and args.base.local_rank in [-1, 0]:
                os.makedirs(save_train_final_results)
            logger.info(f"Saving model checkpoint to {save_train_final_results}")
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(save_train_final_results)
            torch.save(self.config.context.optimizer.state_dict(), os.path.join(save_train_final_results, "optimizer.pt"))
            torch.save(self.config.context.scheduler.state_dict(), os.path.join(save_train_final_results, "scheduler.pt"))
            tokenizer.save_vocabulary(save_train_final_results)
        # Evaluation
        results = {}
        if args.base.do_eval and args.base.local_rank in [-1, 0]:
            
            checkpoints = [args.context.output_dir]
            if args.base.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.context.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info(f"Evaluate the following checkpoints: {checkpoints}")
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
                self.config.context.model =  self.importance_ref[1].from_pretrained(checkpoint, config=args.context.config)
                self.config.context.model.to(args.context.device)
                result = args.context.exe_device.eval_step(self.config,prefix)
                if global_step:
                    result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
                results.update(result)
            output_eval_file = os.path.join(args.base.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

        if args.base.do_predict and args.base.local_rank in [-1, 0]:
            
            checkpoints = [args.context.output_dir]
            if args.base.predict_checkpoints > 0:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.context.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
                checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.base.predict_checkpoints)]
            logger.info(f"Predict the following checkpoints: {checkpoints}")
            for checkpoint in checkpoints:
                prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
                self.config.context.model = self.importance_ref[1].from_pretrained(checkpoint, config=args.context.config)
                self.config.context.model.to(args.context.device)
                args.context.exe_device.predict_step(self.config,prefix)
            

        
            
        
        
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        # 不同任务可公用的前置准备部分 common prepare step
        self.common_preduce_step()

        self.load_processor_and_config()
        # 不同任务的数据预处理部分 Data preprocessing for different tasks
        self.data_preduce_func[self.config.type]()
        # 加载模型和数据
        self.load_model()

        if 'big_data' in self.config and self.config.big_data.need_split_shard:
            task_name = self.config.base.task_name
            tokenizer = self.config.context.tokenizer
            
            examples,label_list = self.load_examples(data_type='train')
            self.config.context.all_examples_size = len(examples)
            shard_size = self.config.big_data.shard_size
            self.load_dev_and_predict_datasets()
            if self.config.big_data.shuffle_before_split:
                shuffle(examples)
            for idx in range(self.config.big_data.start_shard_idx,len(examples),shard_size):
                shard_examples = examples[idx:idx + shard_size]
                self.config.context.train_dataset = self.covert_examples_to_tensors(task_name, tokenizer,
                                                                            shard_examples,label_list, 
                                                                            data_type='train',
                                                                            shard_start_idx=idx,
                                                                            shard_size=shard_size
                                                                            )
                self.config.context.shard_start = idx
                self.config.context.shard_end = idx + shard_size
                self.main_running_step()
            # 不同任务的主运行部分 Main running step for different tasks
            
        else:
            self.load_datasets_complete()
            # 不同任务的主运行部分 Main running step for different tasks
            self.main_running_step()

    
        if 'results_record' in self.config.context:
            io.open(os.path.join(self.config.context.output_dir,'results_record.json'),'w').write(json.dumps(self.config.context.results_record,ensure_ascii=False,indent=4))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='指定配置')
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('conf_file', type=str, help='配置文件')

    args = parser.parse_args()

    MainController(args.conf_file)()
