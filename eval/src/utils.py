import os
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from src.models.hs_edit_models import EditQwen2ForCausalLM, EditLlamaForCausalLM


def load_only_generation_config(model_name_or_path):
    config_path = os.path.join(model_name_or_path, "generation_config.json")
    assert os.path.exists(config_path), f"generation config {config_path} not found."
    generation_config = json.load(open(config_path, "r"))
    return generation_config


def load_model(model_name_or_path, load_method):
    if load_method != "sde" or "olmo" in model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)
    else:
        if "qwen" in model_name_or_path.lower():
            model = EditQwen2ForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        elif "llama" in model_name_or_path.lower():
            model = EditLlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        else:
            raise ValueError(f"model_name_or_path {model_name_or_path} not found in EditModel.")
    # model.to(torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    return model, tokenizer, model_config

import sys
sys.path.insert(0, '../hylat/src')

import transformers
from model_decoder_mas_dual import HyLAT, ModelArguments, DataArguments, TrainingArguments
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
import argparse
import torch

def load_latent_model(model_name_or_path, conf_path):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = parser.parse_yaml_file(conf_path, allow_extra_keys=True)

    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen", "olmo"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError
    
    model = HyLAT(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    
    # model_state = model.state_dict()    
    # embed_key = ["codi.base_model.model.model.embed_tokens.weight","codi.base_model.model.lm_head.weight"]
    # for key in embed_key:
    #     new_embed = model_state[key]   # [128260, 2048]        
    #     state_dict[key] = new_embed
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    
    tokenizer_path = model_args.model_name_or_path 
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            model = model.to('cuda')
        else:
            model = model.cuda()

    model.to(torch.bfloat16)
    
    model.eval()
    model_config = model.codi.config
    return model, tokenizer, model_config


import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import re

def extract_assistant_replies(text):
    pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n\s*(.*?)\s*<\|eot_id\|>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        pattern = r'<\|im_start\|>user\n\s*(.*?)\s*<\|im_end\|>'
        matches = re.findall(pattern, text, re.DOTALL)
    return [reply.strip() for reply in matches]

def count_token(df, tokenizer):
    cnt = []
    for d in tqdm(df):
        tmp = 0
        for key in (k for k in d if k.startswith("agent_")):
            for msg in d[key]["history"]:
                if msg["role"] == "assistant":
                    tmp += len(tokenizer(msg["content"])["input_ids"])
        cnt.append(tmp)
    return np.mean(cnt)

def count_cipher(df, tokenizer):
    cnt = []
    for d in tqdm(df):
        tmp = 0
        for key in (k for k in d if k.startswith("agent_")):
            texts = extract_assistant_replies(d[key]["history"])
            for text in texts:
                tmp += len(tokenizer(text)["input_ids"])
        cnt.append(tmp)
    return np.mean(cnt)

def count_single_token(df, tokenizer):
    cnt = []
    for d in tqdm(df):
        tmp=0
        for msg in d["agent_0"]["history"]:
            if msg["role"]=="assistant":
                tmp+=len(tokenizer(msg["content"])["input_ids"])
        cnt.append(tmp)
    return np.mean(cnt)

def count_trust_game(df, tokenizer):
    cnt = []
    for d in tqdm(df):
        tmp=0
        for msg in d["trustor_history"]:
            if msg["role"]=="assistant":
                tmp+=len(tokenizer(msg["content"])["input_ids"])
        for msg in d["trustee_history"]:
            if msg["role"]=="assistant":
                tmp+=len(tokenizer(msg["content"])["input_ids"])
        cnt.append(tmp)
    return np.mean(cnt)    

def count_trust_game_cipher(df, tokenizer):
    cnt = []
    for d in tqdm(df):
        tmp = 0
        text = d["trustor_history"]
        texts = extract_assistant_replies(text)
        for text in texts:
            tmp+=len(tokenizer(text)["input_ids"])
        text = d["trustee_history"]
        texts = extract_assistant_replies(text)
        for text in texts:
            tmp+=len(tokenizer(text)["input_ids"])       
        cnt.append(tmp)
    return np.mean(cnt) 

