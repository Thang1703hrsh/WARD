import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime
import gc

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from training.arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model, get_distillation_schedule

from methods.distillm import forward_kl, reverse_kl, js_distance, tv_distance
from methods.distillm import skewed_forward_kl, skewed_reverse_kl
from methods.distillm import SampleGenerator, ReplayBuffer
from methods.distillm.projector import Projector
from methods.distillm.losses import get_fdd_loss, dtw_distillation_loss

from methods.distillm2.losses import get_distillm2_loss_split

from rouge_metric import compute_metrics

from peft import PeftModel

from wandb_logger import init_wandb, log_metrics, finish_wandb

torch.set_num_threads(4)

# FDD helpers (same as fdd_finetune.py)

class CustomsQwen3Attention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, **kwargs):
        kwargs["output_attentions"] = False
        return self.original(**kwargs)

class CustomsOPTAttention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, **kwargs):
        kwargs["output_attentions"] = False
        return self.original(**kwargs)

class CustomsGPT2Attention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, hidden_states, **kwargs):
        kwargs["output_attentions"] = False
        return self.original(hidden_states, **kwargs)

def customize_model_attention(args, model, is_teacher=True):
    """
    Disable output_attentions for layers NOT in layer mapping (same as fdd_finetune.py)
    """
    # Try to use teacher_layer_mapping / student_layer_mapping
    args_layer_mapping = args.teacher_layer_mapping if is_teacher else args.student_layer_mapping

    # Prefer model type for teacher if exists
    model_type = getattr(args, "teacher_model_type", None) if is_teacher else getattr(args, "model_type", None)
    if model_type is None:
        model_type = args.model_type

    if model_type == "qwen":
        for i, layer in enumerate(model.model.layers[:-1]):
            if (i + 1) not in args_layer_mapping:
                layer.self_attn = CustomsQwen3Attention(layer.self_attn)
    elif model_type == "opt":
        for i, layer in enumerate(model.model.decoder.layers[:-1]):
            if (i + 1) not in args_layer_mapping:
                layer.self_attn = CustomsOPTAttention(layer.self_attn)
    elif model_type == "gpt2":
        for i, layer in enumerate(model.transformer.h[:-1]):
            if (i + 1) not in args_layer_mapping:
                layer.attn = CustomsGPT2Attention(layer.attn)
    else:
        raise NotImplementedError(f"customize_model_attention: unsupported model_type={model_type}")

def soft_label_distill_loss(student_logits, teacher_logits, mask, distill_temperature=2.0):
    student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)
    loss = F.kl_div(student_probs, teacher_probs, reduction="none").sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss

def get_fdd_loss(args, t_hiddens, s_hiddens, mask, student, teacher):
    """
    Same as fdd_finetune.py:
    - traj_loss: KL on logits of selected hidden states
    - der_loss : cosine similarity on delta log-probs (trajectory derivative)
    """
    i = 0
    traj_loss, der_loss = 0.0, 0.0
    pre_s_hidden_logs, pre_t_hidden_logs = None, None

    for s_idx, t_idx in zip(args.student_layer_mapping, args.teacher_layer_mapping):
        s_hidden = s_hiddens[s_idx]
        t_hidden = t_hiddens[t_idx]

        # OPT needs project_out alignment (same as fdd_finetune.py)
        if args.model_type == "opt":
            s_decoder_proj = student.module.model.model.decoder.project_out
            if s_decoder_proj is not None:
                s_hidden = s_decoder_proj(s_hidden)

            t_decoder_proj = teacher.model.decoder.project_out
            if t_decoder_proj is not None:
                t_hidden = t_decoder_proj(t_hidden)

        s_hidden_logits = student.module.lm_head(s_hidden)
        t_hidden_logits = teacher.lm_head(t_hidden)

        traj_loss = traj_loss + soft_label_distill_loss(s_hidden_logits, t_hidden_logits, mask)

        s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1)
        t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1)

        if i > 0:
            delta_hidden_student = s_hidden_logs - pre_s_hidden_logs
            delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs
            cos_sim = F.cosine_similarity(delta_hidden_student, delta_hidden_teacher, dim=-1, eps=1e-5)
            cos_sim_loss = 1 - cos_sim
            cos_sim_loss = (cos_sim_loss * mask).sum() / mask.sum()
            der_loss = der_loss + cos_sim_loss

        pre_s_hidden_logs, pre_t_hidden_logs = s_hidden_logs, t_hidden_logs
        i += 1

    if i == 0:
        return torch.tensor(0.0)
    if i == 1:
        return traj_loss / i
    return traj_loss / i + der_loss / (i - 1)


def get_teacher_model(args, tokenizer, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        customize_model_attention(args, model, is_teacher=True)

        model.resize_token_embeddings(len(tokenizer))
        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model



def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, tokenizer, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, tokenizer, device)
    model.resize_token_embeddings(len(tokenizer))
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    
    # Check if using DistiLLM-2 Arrow format
    if "distillm2" in args.type:
        # Load Arrow datasets for DistiLLM-2
        from datasets import load_from_disk
        print_rank(f"Loading DistiLLM-2 Arrow dataset from {args.data_dir}")
        raw_datasets = load_from_disk(args.data_dir)
        
        # Create wrapper class with collate method
        class DistiLLM2Dataset:
            def __init__(self, dataset, tokenizer, max_length, max_prompt_length=256):
                self.dataset = dataset
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.max_prompt_length = max_prompt_length
                # Extract answers for evaluation
                self.answers = [[item['chosen']] for item in dataset]
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                sample = self.dataset[idx]
                sample["idx"] = idx
                return sample
            
            def move_to_device(self, model_data, no_model_data, gen_data, device):
                for k in model_data:
                    model_data[k] = model_data[k].to(device)

                for k in no_model_data:
                    if isinstance(no_model_data[k], torch.Tensor):
                        no_model_data[k] = no_model_data[k].to(device)

                for k in gen_data:
                    gen_data[k] = gen_data[k].to(device)

                return model_data, no_model_data, gen_data
            
            def collate(self, examples):
                # Tokenize prompts and responses (chosen = teacher, rejected = student)
                prompts = [ex['prompt'] for ex in examples]
                chosen = [ex['chosen'] for ex in examples]
                rejected = [ex['rejected'] for ex in examples]
                examples_idx = [ex['idx'] for ex in examples]
                
                # Tokenize inputs - use max_prompt_length for prompts to ensure consistent gen_data size
                prompt_tokens = self.tokenizer(prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_prompt_length)
                
                # Tokenize full sequences (prompt + response)
                chosen_full = [p + c for p, c in zip(prompts, chosen)]
                rejected_full = [p + r for p, r in zip(prompts, rejected)]
                
                chosen_tokens = self.tokenizer(chosen_full, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                rejected_tokens = self.tokenizer(rejected_full, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                
                # Find the max length between chosen and rejected to pad them equally
                max_len = max(chosen_tokens['input_ids'].shape[1], rejected_tokens['input_ids'].shape[1])
                
                # Pad chosen and rejected to the same length
                def pad_to_length(tensor, target_length, pad_value):
                    if tensor.shape[1] >= target_length:
                        return tensor[:, :target_length]
                    padding = torch.full((tensor.shape[0], target_length - tensor.shape[1]), pad_value, dtype=tensor.dtype)
                    return torch.cat([tensor, padding], dim=1)
                
                chosen_input_ids = pad_to_length(chosen_tokens['input_ids'], max_len, self.tokenizer.pad_token_id)
                chosen_attention_mask = pad_to_length(chosen_tokens['attention_mask'], max_len, 0)
                rejected_input_ids = pad_to_length(rejected_tokens['input_ids'], max_len, self.tokenizer.pad_token_id)
                rejected_attention_mask = pad_to_length(rejected_tokens['attention_mask'], max_len, 0)
                
                # Create labels by masking prompt tokens with -100
                chosen_labels = chosen_input_ids.clone()
                rejected_labels = rejected_input_ids.clone()
                
                # Mask prompt tokens in labels (only compute loss on response)
                for i in range(len(examples)):
                    prompt_len = (prompt_tokens['attention_mask'][i] == 1).sum().item()
                    # Find actual prompt length in the full sequence (accounting for tokenizer merging)
                    # Use a safe approach: mask first max_prompt_length tokens
                    chosen_labels[i, :prompt_len] = -100
                    rejected_labels[i, :prompt_len] = -100
                
                # Concatenate chosen and rejected for efficient forward pass
                concatenated_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
                concatenated_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
                concatenated_labels = torch.cat([chosen_labels, rejected_labels], dim=0)
                
                # Create model_batch and no_model_batch with concatenated inputs
                model_batch = {
                    'input_ids': concatenated_input_ids,
                    'attention_mask': concatenated_attention_mask,
                }
                
                no_model_batch = {
                    'label': concatenated_labels,
                    'attention_mask': concatenated_attention_mask,
                    'batch_size': len(examples),  # Store original batch size to split chosen/rejected later
                    'examples_idx': examples_idx
                }
                
                # Create gen_data for evaluation - use fixed max_prompt_length to match LMTrainDataset format
                gen_data = {
                    'input_ids': prompt_tokens['input_ids'],
                    'attention_mask': prompt_tokens['attention_mask'],
                }
                
                return model_batch, no_model_batch, gen_data
        
        if args.do_train:
            data["train"] = DistiLLM2Dataset(raw_datasets["train"], tokenizer, args.max_length, args.max_prompt_length)
            print_rank("train num", len(data["train"]))
            if args.do_valid:
                # Use DistiLLM-2 dev.json if exists, else fall back to Dolly valid
                dev_json_path = os.path.join(args.data_dir, "dev.json")
                if os.path.exists(dev_json_path):
                    import json as _json
                    with open(dev_json_path) as _f:
                        dev_raw = _json.load(_f)
                    if args.dev_num > 0:
                        dev_raw = dev_raw[:args.dev_num]
                    from datasets import Dataset as HFDataset
                    dev_hf = HFDataset.from_list(dev_raw)
                    data["dev"] = DistiLLM2Dataset(dev_hf, tokenizer, args.max_length, args.max_prompt_length)
                    print_rank("dev num (DistiLLM-2 dev.json)", len(data["dev"]))
                else:
                    data["dev"] = LMTrainDataset(args, tokenizer, args.gt_data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
                    print_rank("dev num (Dolly gt-data-dir)", len(data["dev"]))
        elif args.do_eval:
            data["test"] = DistiLLM2Dataset(raw_datasets["test"] if "test" in raw_datasets else raw_datasets["train"], tokenizer, args.max_length, args.max_prompt_length)
    else:
        # Use standard LMTrainDataset for other distillation methods
        if args.do_train:
            data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
            print_rank("train num", len(data["train"]))
            if args.do_valid:
                data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
        elif args.do_eval:
            data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
        else:
            raise ValueError("Do train and do eval must set one")
        
    # pre-trained dataset (OpenWebText)
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def pt_loss(args, model, model_batch, no_model_batch):
    loss_mask = (no_model_batch["label"] != -100).int()
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


def get_distil_loss(args, no_model_batch, logits, teacher_logits):
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "distillm2" in args.type or "distillm_v2" in args.type:
            from methods.distillm2.losses import get_distillm2_loss
            labels = no_model_batch["label"]
            attention_mask = no_model_batch.get("attention_mask", None)
            gradual_beta = getattr(args, 'gradual_beta', False)
            distil_loss = get_distillm2_loss(
                student_logits=logits,
                teacher_logits=teacher_logits,
                labels=labels,
                attention_mask=attention_mask,
                loss_type='distillm_v2',
                global_step=getattr(args, 'current_global_step', None),
                max_steps=getattr(args, 'total_iters', None),
                gradual_beta=gradual_beta,
            )
        elif "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        else:
            raise NotImplementedError
    return distil_loss

# def get_distil_loss(args, teacher_logits, no_model_batch, logits):
#     if args.model_parallel:
#         raise NotImplementedError
#     else:
#         if "sfkl" in args.type:
#             distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
#         elif "srkl" in args.type:
#             distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
#         elif "jsd" in args.type:
#             distil_loss = js_distance(logits, teacher_logits, no_model_batch)
#         elif "tvd" in args.type:
#             distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
#         elif "fkl" in args.type or args.type == "kd":
#             distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
#         elif "rkl" in args.type:
#             distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
#         else:
#             raise NotImplementedError
#     return distil_loss


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def finetune(
    args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, 
    optimizer: AdamW, lr_scheduler, 
    dataset, device, 
    teacher_model=None, velocity_field=None, projector=None, update_velocity_dict={}
):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    
    if "distillm2" in args.type:
        gen_dataloader = DataLoader(
            dataset["train"], batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate
        )
        # idx_to_update = []
        # new_sgos = {}
    
    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(
        dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)
        
    student_generator = SampleGenerator(args, tokenizer)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_contra_loss, total_dtw_loss, total_fdd_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else None
    if args.do_valid:
        prev_avg_loss, rouge_results = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
        best_val_loss, best_rouge = prev_avg_loss, rouge_results.get("rougeL", None)
    else:
        prev_avg_loss = 999999
        best_val_loss, best_rouge = 999999, -1
    best_val_iter = -1
    replay_buffer = ReplayBuffer(args)
    torch.cuda.empty_cache()
    
    output_hidden_states = False
    if "fdd" in args.type or "dtw" in args.type:
        teacher_schedule, student_schedule = get_distillation_schedule(
            args.num_teacher_layers,
            args.num_student_layers,
            args.num_distill_layers
        )
        output_hidden_states = True
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        cur_t = (epoch+1)/args.epochs # for curriculum-based contra target
        
        # generate new SGOs
        
            
            # update 'm' value used in computing adaptive skew coefficients
            # aggregate over all diff of log of batches from prev. epoch
            # args.logp_logq = torch.mean(torch.tensor(args.logp_logq_list)).item()
            # args.logq_logp = torch.mean(torch.tensor(args.logq_logp_list)).item()
            # args.logp_logq_list, args.logq_logp_list = [], []
            
            # def update_sgos(example, idx):
            #     if idx in idx_to_update:
            #         example['rejected'] = new_sgos[idx]
            #     return example
            # dataset["train"].dataset.map(update_sgos, with_indices=True)
            # idx_to_update = []
            # new_sgos = {}
        
        model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            
            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                except:
                    pt_train_iter = iter(pt_train_dataloader)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)
            
            torch.cuda.synchronize()
            st_time = time.time()
            
            
            if "adaptive" in args.type:
                # sampling ratio:
                samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            
            # data generation
            if args.student_gen:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)
                    
                # elif "adaptive" in args.type and "distillm2" in args.type \
                #     and r < samp_threshold:
                #     # handles adaptive SGO generation
                #     examples_idx = no_model_batch["examples_idx"] # to assign new SGOs
                #     batch_size = no_model_batch["batch_size"]
                    
                #     # generate similar to adaptive approach in DistiLLM (v1)
                #     new_model_batch = student_generator.run_sample(model, gen_data, with_responses=True)
                    
                #     # modify batch data with new generated data
                #     no_model_batch["label"][batch_size//2:] = new_model_batch.pop("no_model_batch")
                #     model_batch["input_ids"][batch_size//2:] = new_model_batch["input_ids"]
                #     model_batch["attention_mask"][batch_size//2:] = new_model_batch["attention_mask"]
                    
                #     # update sgo in dataset
                #     new_sgo = new_model_batch["response"]
                #     idx_to_update.extend(examples_idx)
                #     for (idx, sgo) in zip(examples_idx, new_sgo):
                #         new_sgos[idx] = sgo
                    
                elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):

                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    if args.model_type in ["opt"]:
                        model_batch.pop('position_ids')
                        
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)
                    
                model.train()

            output_hidden_states = "fdd" in args.type or "dtw" in args.type
            outputs = model(**model_batch, output_hidden_states=output_hidden_states, use_cache=False)
            logits = outputs.logits

            if args.model_parallel:
                raise NotImplementedError
            else:
                lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(
                        **model_batch,
                        output_hidden_states=output_hidden_states,
                        use_cache=False
                    )
                    teacher_logits = teacher_outputs.logits

                distil_loss = get_distil_loss(args, no_model_batch, logits, teacher_logits)
                if torch.isnan(distil_loss) or torch.isinf(distil_loss):
                    distil_loss = torch.tensor(0.0, device=logits.device)

                if "fdd" in args.type:
                    fdd_loss = get_fdd_loss(
                        args,
                        teacher_outputs.hidden_states,
                        outputs.hidden_states,
                        model_batch["attention_mask"],
                        model,
                        teacher_model
                    )
                else:
                    fdd_loss = torch.tensor(0.0).to(distil_loss.device)

                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * (distil_loss + fdd_loss)
            else:
                loss = lm_loss

            contra_loss = torch.tensor(0.0).to(loss.device)

            if "dtw" in args.type:
                hidden_states = outputs.hidden_states
                mask = model_batch["attention_mask"].float()

                if args.dtw_gamma_steps and args.dtw_gamma_steps > 0:
                    gamma_start = args.dtw_gamma_start if args.dtw_gamma_start is not None else args.dtw_gamma
                    gamma_end = args.dtw_gamma_end if args.dtw_gamma_end is not None else args.dtw_gamma
                    progress = min(1.0, float(global_step) / float(args.dtw_gamma_steps))
                    dtw_gamma = gamma_start + (gamma_end - gamma_start) * progress
                else:
                    dtw_gamma = args.dtw_gamma

                unit_ids = None
                importance_weights = None
                if args.dtw_unitization:
                    unit_ids = no_model_batch.get("unit_ids", None)
                if args.dtw_importance_weights == "teacher_entropy":
                    with torch.no_grad():
                        teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
                        entropy = -(teacher_probs * torch.log(teacher_probs + 1e-9)).sum(dim=-1)
                    importance_weights = 1.0 / (entropy + 1e-6)

                dtw_loss = dtw_distillation_loss(
                    hidden_states,
                    teacher_outputs.hidden_states,
                    student_schedule,
                    teacher_schedule,
                    mask,
                    projector=projector,
                    window_size=args.dtw_window,
                    gamma=dtw_gamma,
                    distance=args.dtw_distance,
                    normalize=args.dtw_normalize,
                    use_divergence=args.dtw_use_divergence,
                    band_source=args.dtw_band_source,
                    band_width=args.dtw_band_width,
                    band_penalty=args.dtw_band_penalty,
                    band_center_blend=args.dtw_band_center_blend,
                    band_entropy_coef=args.dtw_band_entropy_coef,
                    band_warmup_steps=args.dtw_band_warmup_steps,
                    current_step=global_step,
                    unit_ids=unit_ids,
                    importance_weights=importance_weights,
                )
                if torch.isnan(dtw_loss) or torch.isinf(dtw_loss):
                    dtw_loss = torch.tensor(0.0, device=loss.device)
                loss += args.dtw_weight * dtw_loss
            else:
                dtw_loss = torch.tensor(0.0).to(loss.device)
                
            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)
                
            model.backward(loss)
            model.step()
             
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size

            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss / (args.log_interval * args.gradient_accumulation_steps)
            
            global_contra_loss = 0
            if "contra" in args.type:
                dist.all_reduce(contra_loss, dist.ReduceOp.SUM, group=dp_group)
                global_contra_loss = contra_loss.item() / dp_world_size
                total_contra_loss += global_contra_loss / (args.log_interval * args.gradient_accumulation_steps)

            global_dtw_loss = 0
            if "dtw" in args.type:
                dist.all_reduce(dtw_loss, dist.ReduceOp.SUM, group=dp_group)
                global_dtw_loss = dtw_loss.item() / dp_world_size
                total_dtw_loss += global_dtw_loss / (args.log_interval * args.gradient_accumulation_steps)

            global_fdd_loss = 0
            if "fdd" in args.type:
                dist.all_reduce(fdd_loss, dist.ReduceOp.SUM, group=dp_group)
                global_fdd_loss = fdd_loss.item() / dp_world_size
                total_fdd_loss += global_fdd_loss / (args.log_interval * args.gradient_accumulation_steps)
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss / (args.log_interval * args.gradient_accumulation_steps)
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_contra_loss, log_dtw_loss, log_fdd_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | contra_loss: {:.4f} | dtw_loss: {:.4f} | fdd_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    log_contra_loss,
                    log_dtw_loss,
                    log_fdd_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, global_contra_loss, global_dtw_loss, global_fdd_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss,
                    total_distil_loss,
                    total_contra_loss,
                    total_dtw_loss,
                    total_fdd_loss,
                    total_time / (args.log_interval))
                # print_rank("*" * 100)
                print_rank(log_str)
                # print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                
                # Log to wandb (only rank 0)
                if dist.get_rank() == 0:
                    metrics = {
                        "train/loss": total_loss,
                        "train/distil_loss": total_distil_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    }
                    if "contra" in args.type:
                        metrics["train/contra_loss"] = total_contra_loss
                    if "dtw" in args.type:
                        metrics["train/dtw_loss"] = total_dtw_loss
                    if "fdd" in args.type:
                        metrics["train/fdd_loss"] = total_fdd_loss
                    log_metrics(metrics, step=global_step)

                total_loss, total_distil_loss, total_contra_loss, total_dtw_loss, total_fdd_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


            # Evaluation
            del logits, outputs
            torch.cuda.empty_cache()
            if args.do_valid \
                and args.eval_interval \
                and global_step % args.eval_interval == 0 \
                and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss, curr_rouge_results = evaluate(args, tokenizer, model, dataset["dev"], "dev", global_step, device, adaptive_threshold)
                curr_em, curr_rouge = curr_rouge_results.get("exact_match", None), curr_rouge_results.get("rougeL", None)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss
                    
                # Checkpointing
                improved = (curr_avg_loss < best_val_loss)
                if args.eval_gen:
                    assert curr_rouge is not None and best_rouge is not None
                    improved = (curr_rouge > best_rouge)
                if args.save and args.save_interval and global_step % args.save_interval == 0 \
                    and step % args.gradient_accumulation_steps == 0 \
                    and improved:
                    best_val_loss = curr_avg_loss
                    if args.eval_gen:
                        best_rouge = curr_rouge
                    best_val_iter = global_step
                    save_dir_path = os.path.join(args.save, str(global_step))
                    if args.model_parallel:
                        raise NotImplementedError
                    else:
                        if args.only_save_last:
                            best_val_iter = -1
                        elif dist.get_rank() == 0:
                            os.makedirs(save_dir_path, exist_ok=True)
                            print_rank(f"Model save to {save_dir_path}")
                            tokenizer.save_pretrained(save_dir_path)
                            model.module.save_pretrained(save_dir_path, safe_serialization=False)
                    dist.barrier()
                model.train()
            torch.cuda.empty_cache()

            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
                args.current_global_step = global_step
            step += 1
            
            if global_step > args.total_iters:
                break
            
        optimizer.zero_grad()
        if "distillm2" in args.type and epoch < (args.epochs-1) and (epoch+1) % args.generate_sgo_interval == 0:
            model.eval()
            print(f"Generating new SGOs at epoch {epoch}")
            # form 'rejected' responses for Dataset obj complying with loaded Arrow dataset in prepare_dataset
            # maybe load prompts from dataset["train"].dataset or sth
            new_sgo = []
            # student generate
            for model_batch, no_model_batch, gen_data in tqdm(gen_dataloader, desc="Generating"):
                dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
                # handle ordering, currently dataloader randomly samples not in order of original dataset (done)
                gen_results = student_generator.run_sample(model, gen_data, responses_only=True)
                new_sgo.extend(gen_results["response"])
            
            # modify 'rejected' responses in dataset obj
            # dataset["train"].dataset: the dataset; modify rejected responses in here with new SGOs
            # do not need to recreate DistributedSampler or DataLoader, dataset is still the same (same size), only contents modified
            dataset["train"].dataset = dataset["train"].dataset.remove_columns("rejected").add_column("rejected", new_sgo)
            print("Generation complete!")
            
        torch.cuda.empty_cache()
    return model, best_val_iter


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
        references = dataset.answers
        responses = responses[:len(references)]
        res = compute_metrics(responses, references)
    else:
        res = {}
    
    if get_rank() == 0:
        if args.eval_gen:
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
        # Log to wandb (only rank 0)
        if dist.get_rank() == 0:
            metrics = {f"eval/{split}_loss": avg_loss}
            if res:
                for key, val in res.items():
                    metrics[f"eval/{split}_{key}"] = val
            log_metrics(metrics, step=epoch)
        
    return all_loss / step, res


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    args.logp_logq = None
    args.logq_logp = None
    args.logp_logq_list = []
    args.logq_logp_list = []
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    # Initialize wandb (only on rank 0) - reads from wandb_config.yaml if key not provided
    if dist.get_rank() == 0:
        wandb_name = args.wandb_name or f"{args.ckpt_name}-{args.type}"
        wandb_config = {
            "type": args.type,
            "model": args.ckpt_name,
            "teacher": args.teacher_ckpt_name if hasattr(args, 'teacher_ckpt_name') else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
        }
        # Pass wandb_key (can be None, will auto-load from YAML)
        init_wandb(args.wandb_project, wandb_name, wandb_config, args.wandb_key, args.base_path)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]    
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer, ds_config, device, set_optim=args.do_train)
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, tokenizer, device)
    else:
        teacher_model = None
    
    velocity_field = None
    projector = None

    if projector is None and args.projector_path is not None:
        projector = Projector(d_student=args.d_student, d_teacher=args.d_teacher)
        projector.load_state_dict(torch.load(
            args.projector_path,
            map_location=f"cuda:{device}",
            weights_only=True
        ))
        projector.to(device)
        projector.eval()

    if teacher_model:
        teacher_model.resize_token_embeddings(len(tokenizer))
    
    update_velocity_dict = {}

    if args.do_train:
        model, best_val_iter = finetune(
            args, tokenizer, model, 
            optimizer, lr_scheduler, 
            dataset, device, 
            teacher_model, velocity_field, projector, update_velocity_dict
        )
        # Copy best saved checkpoint out to root dir of this model type
        if args.save and dist.get_rank() == 0:
            if best_val_iter == -1:
                if isinstance(model.module, PeftModel):
                    model.module.save_pretrained(args.save, safe_serialization=False, save_embedding_layers=False)
                else:
                    model.module.save_pretrained(args.save, safe_serialization=False)
            else:
                best_ckpt_path = os.path.join(args.save, str(best_val_iter))
                import shutil
                for filename in os.listdir(best_ckpt_path):
                    src_path = os.path.join(best_ckpt_path, filename)
                    dst_path = os.path.join(args.save, filename)

                    if os.path.isfile(src_path):  # only copy files
                        shutil.copy2(src_path, dst_path)  # overwrite if exists
            tokenizer.save_pretrained(args.save)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
    
    # Finish wandb
    if dist.get_rank() == 0:
        finish_wandb()
        
    
if __name__ == "__main__":
    main()