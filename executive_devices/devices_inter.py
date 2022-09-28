from asyncio.log import logger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch
from callback.adversarial import FGM
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
import os

class ExecutiveDevice():

    '''
    公共准备流程/ Start
    '''
    def predict_preparatory(self,context_config):
        '''
        推理前的公共准备部分
        '''
        args = context_config.base
        context = context_config.context

        context.test_sampler = RandomSampler(context.test_dataset) if args.local_rank == -1 else DistributedSampler(context.test_dataset)
        context.test_dataloader = DataLoader(context.test_dataset, sampler=context.test_sampler, batch_size=1,
                                    collate_fn=context.processor.predict_collate_fn)

    def eval_preparatory(self,context_config):
        '''
        评估前的公共准备部分
        '''
        args = context_config.base
        context = context_config.context

        context.eval_batch_size = args.per_gpu_eval_batch_size * max(1, context.n_gpu)
        context.eval_sampler = RandomSampler(context.eval_dataset) if args.local_rank == -1 else DistributedSampler(context.eval_dataset)
        context.eval_dataloader = DataLoader(context.eval_dataset, sampler=context.eval_sampler, batch_size=context.eval_batch_size,
                                    collate_fn=context.processor.dev_collate_fn)

    def train_preparatory(self,context_config,optimizer_grouped_parameters,global_step=0,init_shard=False):
        '''
        训练前的公共准备部分
        '''
        args = context_config.base
        context = context_config.context
        logger = context.logger
        model = context.model

        context.train_batch_size = args.per_gpu_train_batch_size * max(1, context.n_gpu)
        context.train_sampler = RandomSampler(context.train_dataset) if args.local_rank == -1 else DistributedSampler(context.train_dataset)
        context.train_dataloader = DataLoader(context.train_dataset, sampler=context.train_sampler, batch_size=context.train_batch_size,
                                    collate_fn=context.processor.train_collate_fn)
        if args.max_steps > 0:
            context.t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(context.train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            context.t_total = len(context.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


        if context_config.big_data.need_split_shard:
        # if False:
            # 分片初始
            if init_shard:
                shard_warmup_steps = int(context_config.big_data.big_data_total_step * args.warmup_proportion)
                context.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
                context.scheduler = get_linear_schedule_with_warmup(context.optimizer, num_warmup_steps=shard_warmup_steps,
                                                            num_training_steps=context_config.big_data.big_data_total_step)
                logger.info(f'分片训练初始化优化器，预热步数:{shard_warmup_steps},总步数:{context_config.big_data.big_data_total_step}')
            else:
                logger.info(f'当前步数:{global_step},不重新初始化优化器')

        else:
            # 非分片初始
            
            context.warmup_steps = int(context.t_total * args.warmup_proportion)
            context.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            context.scheduler = get_linear_schedule_with_warmup(context.optimizer, num_warmup_steps=context.warmup_steps,
                                                        num_training_steps=context.t_total)
            logger.info(f'常规初始化优化器，预热步数:{context.warmup_steps},总步数:{context.t_total}')

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            context.optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            context.scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
            

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        # multi-gpu training (should be after apex fp16 initialization)
        if context.n_gpu > 1:
            context.model = torch.nn.DataParallel(model)
        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            context.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
        

        # Train!
        logger.info(f"***** Running training {context.shard_start}~{context.shard_end}*****")
        logger.info(f"  Num examples = {len(context.train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")

        total_train_batch_size = context.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {context.t_total}")

        
        steps_trained_in_current_epoch = 0
        
        if args.continue_train and global_step == 0:
            context.global_step = context.continue_step
        else:
            context.global_step = global_step
        

        context.steps_trained_in_current_epoch = steps_trained_in_current_epoch
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(context.train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(context.train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {global_step}")
            logger.info(f"  Will skip the first %d steps in the first epoch {steps_trained_in_current_epoch}")
        
        if args.save_steps==-1 and args.logging_steps==-1:
            args.logging_steps=len(context.train_dataloader)
            args.save_steps = len(context.train_dataloader)

    '''
    公共准备流程/ End
    '''

    '''
    train/eval/predict 定义 /Start
    '''
    def train_step(self,context_config):
        '''
        训练部分
        1.进行公共准备
        2.进行模型梯度置0
        3.创建ProgressBar监控进度
        4.开始开启训练
            4.1.开启当前epoch训练
            4.2.当前epoch结束后进行显存清理（如果使用gpu）
        '''
        args = context_config.base
        context = context_config.context

        logger = context.logger

        global_step = context.get('global_step',0)
        init_shard = global_step == 0
        
        self.train_preparatory(context_config,self.get_optimizer_grouped_parameters(context_config),global_step=global_step,init_shard=init_shard)


        model = context.model
        
        context.tr_loss, context.logging_loss = 0.0, 0.0
        if context_config.adversarial.do_adv:
            context.fgm = FGM(model, emb_name=context_config.adversarial.adv_name, epsilon=context_config.adversarial.adv_epsilon)
        model.zero_grad()
        train_dataloader = context.train_dataloader
        
        pbar = ProgressBar(n_total=len(train_dataloader), desc=f'Training-{context.shard_start}~{context.shard_end}', num_epochs=int(args.num_train_epochs))
        
        for epoch in range(int(args.num_train_epochs)):
            pbar.reset()
            pbar.epoch_start(current_epoch=epoch+1)
            self.train_epoch(context_config,pbar)
            if 'cuda' in str(context.device):
                torch.cuda.empty_cache()
        return context.global_step, context.tr_loss / context.global_step
    
    def eval_step(self,context_config,prefix=""):
        raise NotImplementedError()
    
    def predict_step(self,context_config,prefix=""):
        raise NotImplementedError()
    
    def train_epoch(self,context_config,pbar):
        raise NotImplementedError()
    
    def train_batch_step(self,context_config,batch,optimizer):
        raise NotImplementedError()

    def eval_batch_step(self,context_config,metric,batch,eval_loss,nb_eval_steps):
        raise NotImplementedError()

    def predict_batch_step(self,context_config,batch,step):
        raise NotImplementedError()

    '''
    train/eval/predict 定义 /End
    '''

    '''
    公共方法 /Start
    '''
    def save_ckpt(self,context_config,model):
        '''
        保存检查点
        '''
        
        context = context_config.context
        logger = context.logger
        # Save model checkpoint
        output_dir = os.path.join(context_config.context.output_dir,"checkpoint_dir" ,"checkpoint-{}".format(context_config.context.global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)

        logger.info(f"Saving model checkpoint to {output_dir}")
        context.tokenizer.save_vocabulary(output_dir)
        torch.save(context.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(context.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info(f"Saving optimizer and scheduler states to {output_dir}",)

    def get_optimizer_grouped_parameters(self,context_config):
        '''
        获取优化器组参数
        context_config: 总配置
        '''
        raise NotImplementedError()

    def get_eval_tags(self,model,logits,inputs):
        '''
        获取这一batch的推理tag
        model: 模型
        logits: 分析结果
        inputs: 输入
        '''
        raise NotImplementedError()
    
    def get_model_input(self,batch,type):
        '''
        获取模型输入
        batch: batch原始数据
        type: 数据类型(train,eval,predict)
        '''
        raise NotImplementedError()
    
    def get_metrics(self,context_config):
        '''
        获取评估器
        context_config: 总配置
        '''
        raise NotImplementedError()
    
    '''
    公共方法 /End
    '''