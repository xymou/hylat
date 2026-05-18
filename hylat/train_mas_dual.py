# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import copy
import logging
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial
import argparse
from train import *

from src.model_decoder_mas_dual import (
    HyLAT,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)

IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def extract_short_answer(text):
    text = text.strip()
    
    # 1. 优先提取 \boxed{...} 里的内容
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    # 2. 提取 "The answer is:" 或 "The answer is" 后面紧跟的内容
    #    取最后一次出现，避免CoT中间出现的干扰
    answer_is = re.findall(
        r'[Tt]he answer is[:\s]+([^\n.]+)',
        text
    )
    if answer_is:
        candidate = answer_is[-1].strip().rstrip('.')
        # 如果提取出来的内容不超过一个合理长度，认为是有效答案
        if len(candidate) <= 50:
            return candidate

    # 3. 提取选项字母 A/B/C/D/E（单独出现或在括号里）
    letter = re.search(r'\(([A-E])\)|(?<!\w)([A-E])(?!\w)', text)
    if letter:
        return (letter.group(1) or letter.group(2)).strip()
    
    # 4. True/False
    # if re.search(r'\bTrue\b', text, re.IGNORECASE):
    #     return 'True'
    # if re.search(r'\bFalse\b', text, re.IGNORECASE):
    #     return 'False'    
    # 5. #### 分隔符（GSM8K格式）
    if '####' in text:
        val = text.split('####')[-1].strip().replace(',', '')
        # 只取数字部分
        num = re.search(r'-?\d+\.?\d*', val)
        if num:
            return num.group()
    
    # 6. 纯数字
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1]
    
    # 7. fallback：返回原始文本，让调用方决定是否跳过
    return text    

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs,num_items_in_batch=None):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        #"ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        if step % self.args.logging_steps == 0:
            self.log({"loss": loss.item(), "ce_loss": outputs["ce_loss"], "distill_loss": outputs["distill_loss"], 
                      "ref_ce_loss": outputs["ref_ce_loss"],"decode_loss": outputs["decode_loss"], "loss_detail": outputs["loss_detail"]})
            # self.log({"loss": loss.mean().item(), "ce_loss": outputs["ce_loss"].mean().item(), "distill_loss": outputs["distill_loss"].mean().item(), 
            #           "ref_ce_loss": outputs["ref_ce_loss"].mean().item(),"decode_loss": outputs["decode_loss"].mean().item()})
        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,#training_args.model_max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def train(args):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = parser.parse_yaml_file(args.conf, allow_extra_keys=True)

    ##########################
    #       Peft Model       #
    ##########################
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


    model = HyLAT(model_args, training_args, lora_config)
    if model_args.stage1_ckpt_dir:
        try:
            state_dict = load_file(os.path.join(model_args.stage1_ckpt_dir, "model.safetensors"))
        except Exception:
            state_dict = torch.load(os.path.join(model_args.stage1_ckpt_dir, "pytorch_model.bin"))
        
        model_state = model.state_dict()    
        embed_key = ["codi.base_model.model.model.embed_tokens.weight","codi.base_model.model.lm_head.weight"]#, "decoder.base_model.model.model.embed_tokens.weight", "decoder.base_model.model.lm_head.weight"]
        for key in embed_key:
            new_embed = model_state[key]   # [128260, 2048]        
            state_dict[key] = new_embed
        model.load_state_dict(state_dict, strict=False)
        model.codi.tie_weights()
    
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #         token=model_args.token,
    #         cache_dir=training_args.cache_dir,
    #         model_max_length=training_args.model_max_length,
    #         padding_side="right",
    #         use_fast=False,
    #     )
    tokenizer = model.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        #answer_prompt = torch.tensor([464, 3280, 318, 25])
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception:
            breakpoint()       
            
            
        
    def add_chat_template_suffix_llama(
        text: str,
    ):
        suffix = f"{{content}}<|eot_id|>"
        return suffix.format(content=text)
    

    def add_chat_template_prefix(text, add_system, model_name):
        if 'qwen' in model_name:
            add_chat_template_prefix_func = add_chat_template_prefix_qwen
        elif 'llama' in model_name:
            add_chat_template_prefix_func = add_chat_template_prefix_llama
        elif 'gemma' in model_name:
            add_chat_template_prefix_func = add_chat_template_prefix_gemma
        elif "ministral" in model_name:
            add_chat_template_prefix_func = add_chat_template_prefix_ministral 
        elif "olmo" in model_name:
            add_chat_template_prefix_func = add_chat_template_prefix_olmo       
        else:
            raise NotImplementedError(f"{model_name} not supported!")
        return add_chat_template_prefix_func(text, add_system)
        
    def add_chat_template_suffix(text, model_name):
        if 'qwen' in model_name:
            add_chat_template_suffix_func = add_chat_template_suffix_qwen
        elif 'llama' in model_name:
            add_chat_template_suffix_func = add_chat_template_suffix_llama
        elif 'gemma' in model_name:
            add_chat_template_suffix_func = add_chat_template_suffix_gemma
        elif "ministral" in model_name:
            add_chat_template_suffix_func = add_chat_template_suffix_ministral         
        elif "olmo" in model_name:
            add_chat_template_suffix_func = add_chat_template_suffix_olmo  
        else:
            raise NotImplementedError(f"{model_name} not supported!")     
        return add_chat_template_suffix_func(text) 
    
    def build_agent_context(
        dialogue,
        tokenizer,
        bot_token,
        eot_token,
        latent_token, 
        num_latent,
        model_name
        ):        
        OTHER_TEMPLATE=[
            """These are the solutions to the problem from other agents: {content}\n\nUsing the responses from other agents as additional information, can you give an answer to the original question?\nExamine your solution and that other agents step by step.\nMake sure to state your answer at the end of the response.""",
            """These are the responses from other agents: {content}\n\nWhat is your answer to the original question? Make sure to state your answer at the end of the response."""
        ]
        
        other_answer = dialogue["answer"]
        other_cot = dialogue["cot"]
        other_latent = latent_token *num_latent
        
        template = random.choice(OTHER_TEMPLATE)
        # latent
        response = bot_token+ other_latent + eot_token +other_answer
        response = template.format(content = response)
        context = add_chat_template_prefix(response, add_system=False, model_name=model_name)
        # text
        text_context = template.format(content=f"{other_cot} {other_answer}")
        text_context = add_chat_template_prefix(text_context, add_system=False, model_name=model_name)
        content_tokens = tokenizer.encode(context, add_special_tokens=False)
        text_context_tokens = tokenizer.encode(text_context, add_special_tokens=False)
        return content_tokens, text_context_tokens

    def preprocess_mas(
        dialogue_list: Sequence[list],
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
        latent_id: int, 
        bot_token: str,
        eot_token: str,
        latent_token: str,         
        num_latent: int,
        model_name: str,
        mask_inter_res: bool
    ) -> Dict:
        """
        Construct dual-perspective data
        
        dialogue_list: List[List[Dict]], each dialog consits of multi-turns
        format of each turn: {"question": str, "cot": str, "answer": str}
        """        
        print("Tokenizing inputs... This may take some time...")
        # Agent1
        agent1_encoder_ids_list = []
        agent1_decoder_ids_list = []
        agent1_labels_list = []
        agent1_model_answer_pos_list = []
        agent1_ref_answer_pos_list = []
        agent1_full_ref_tokens_list = []
        agent1_full_ref_labels_list = []
        agent1_long_decoder_ids_list = []
        agent1_long_labels_list = []
        
        # Agent2
        agent2_encoder_ids_list = []
        agent2_decoder_ids_list = []
        agent2_labels_list = []
        agent2_model_answer_pos_list = []
        agent2_ref_answer_pos_list = []
        agent2_full_ref_tokens_list = []
        agent2_full_ref_labels_list = []
        agent2_long_decoder_ids_list = []
        agent2_long_labels_list = []
        
        mask_inter_res_list = []
        
        answer_prompts = [
            torch.tensor(tokenizer.encode("The answer is:")),
            torch.tensor(tokenizer.encode("I would say:"))
        ]
        if answer_prompts[0][0] == tokenizer.bos_token_id:
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]        
        
        
        for dialogue_turns in dialogue_list:
            sample_mask_inter_res = dialogue_turns[0].get("mask_inter_res", mask_inter_res)
            # data for a single dialog
            agent1_encoder_ids = []
            agent1_decoder_ids = []
            agent1_labels = []
            agent1_model_answer_pos = []
            agent1_ref_answer_pos = []
            agent1_long_decoder_ids = []
            agent1_long_labels = []
            agent1_full_ref_tokens = []
            agent1_cumulative_length = 0

            agent2_encoder_ids = []
            agent2_decoder_ids = []
            agent2_labels = []
            agent2_model_answer_pos = []
            agent2_ref_answer_pos = []
            agent2_long_decoder_ids = []
            agent2_long_labels = []
            agent2_full_ref_tokens = []
            agent2_cumulative_length = 0
            
            agent1_response_spans, agent2_response_spans = [],[]
            
            # initial_question = dialogue_turns[0].get("question", None) # only for turn 0
            initial_question_agent1 = dialogue_turns[0].get("question1", None) # only for turn 0
            initial_question_agent2 = dialogue_turns[0].get("question2", None) # only for turn 0
            dialogue_turns = dialogue_turns[1:]
            for turn_idx, turn_data in enumerate(dialogue_turns):
                cot = turn_data["cot"]
                answer = turn_data["answer"]
                
                # ====== Agent1 ======
                if turn_idx % 2==0:
                    if turn_idx == 0:
                        question_text = add_chat_template_prefix(
                            initial_question_agent1,
                            add_system=True,
                            model_name=model_name
                        )
                        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                        context_tokens = []
                        other_text_contexts_tokens = []
                    else:
                        context_tokens, other_text_contexts_tokens = build_agent_context(
                            dialogue_turns[turn_idx-1],
                            tokenizer,
                            bot_token,
                            eot_token,
                            latent_token,
                            num_latent,
                            model_name
                        )
                        question_tokens = []
                        
                    # Encoder = context + question + bot
                    encoder_tokens = context_tokens + question_tokens + [bot_id]
                    encoder_ids = torch.tensor(encoder_tokens, dtype=torch.long)
                    agent1_encoder_ids.append(encoder_ids)              
                                                        
                    # decoder: answer
                    answer_text = add_chat_template_suffix(answer, model_name)
                    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
                    decoder_ids = torch.tensor([eot_id] + answer_tokens, dtype=torch.long)
                    agent1_decoder_ids.append(decoder_ids)
                    agent1_labels.append(decoder_ids.clone())                
                    
                    # Model answer position (local position to decoder_ids)
                    model_pos = get_answer_token_position(decoder_ids, answer_prompts, tokenizer)
                    agent1_model_answer_pos.append(model_pos)
                    
                    # === teacher task ===
                    if turn_idx ==0:
                        ref_text = question_text + cot + answer_text
                    else:
                        ref_text = cot + answer_text
                        
                    ref_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
                    ref_tokens = other_text_contexts_tokens + ref_tokens
                    ref_tokens_tensor = torch.tensor(ref_tokens, dtype=torch.long)
                    local_ref_pos = get_answer_token_position(ref_tokens_tensor, answer_prompts, tokenizer)
                    global_ref_pos = agent1_cumulative_length + local_ref_pos
                    agent1_ref_answer_pos.append(global_ref_pos)     
                    
                    answer_start = agent1_cumulative_length + (len(ref_tokens) - len(answer_tokens))
                    answer_end = agent1_cumulative_length + len(ref_tokens)
                    agent1_response_spans.append((answer_start, answer_end))
                    
                    agent1_cumulative_length += len(ref_tokens)
                    agent1_full_ref_tokens.extend(ref_tokens)
                    
                    # === external decoder ===
                    cot_tokens = tokenizer.encode(cot, add_special_tokens=False)
                    long_decoder_ids = torch.tensor(cot_tokens, dtype=torch.long)
                    long_labels = torch.cat([
                        torch.tensor([-100] * (num_latent + 1)),
                        long_decoder_ids
                    ], dim=0)
                    
                    agent1_long_decoder_ids.append(long_decoder_ids)
                    agent1_long_labels.append(long_labels)

                # ====== Agent2 ======
                if turn_idx % 2 == 1:      
                    if turn_idx == 1:
                        question_text = add_chat_template_prefix(
                            initial_question_agent2,
                            add_system=True,
                            model_name=model_name
                        )
                        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                        context_tokens = []
                        other_text_contexts_tokens = []
                    else:
                        context_tokens, other_text_contexts_tokens = build_agent_context(
                            dialogue_turns[turn_idx-1],
                            tokenizer,
                            bot_token,
                            eot_token,
                            latent_token,
                            num_latent,
                            model_name
                        )
                        question_tokens = []    
                                
                    # Encoder = context + question + bot
                    encoder_tokens = context_tokens + question_tokens + [bot_id]
                    encoder_ids = torch.tensor(encoder_tokens, dtype=torch.long)
                    agent2_encoder_ids.append(encoder_ids)                
                
                    # decoder: answer
                    answer_text = add_chat_template_suffix(answer, model_name=model_name)
                    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
                    decoder_ids = torch.tensor([eot_id] + answer_tokens, dtype=torch.long)
                    agent2_decoder_ids.append(decoder_ids)
                    agent2_labels.append(decoder_ids.clone())
                    
                    # Model answer position
                    model_pos = get_answer_token_position(decoder_ids, answer_prompts, tokenizer)
                    agent2_model_answer_pos.append(model_pos)                            
                
                    # === teacher task ===
                    if turn_idx ==1:
                        ref_text = question_text + cot + answer_text
                    else:
                        ref_text = cot + answer_text
                    
                    ref_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
                    ref_tokens = other_text_contexts_tokens + ref_tokens
                    ref_tokens_tensor = torch.tensor(ref_tokens, dtype=torch.long)
                    local_ref_pos = get_answer_token_position(ref_tokens_tensor, answer_prompts, tokenizer)
                    global_ref_pos = agent2_cumulative_length + local_ref_pos
                    agent2_ref_answer_pos.append(global_ref_pos)
                    
                    answer_start = agent2_cumulative_length + (len(ref_tokens) - len(answer_tokens))
                    answer_end = agent2_cumulative_length + len(ref_tokens)
                    agent2_response_spans.append((answer_start, answer_end))
                    
                    agent2_cumulative_length += len(ref_tokens)
                    agent2_full_ref_tokens.extend(ref_tokens)
                    
                    # === external decoder ===
                    cot_tokens = tokenizer.encode(cot, add_special_tokens=False)
                    long_decoder_ids = torch.tensor(cot_tokens, dtype=torch.long)
                    long_labels = torch.cat([
                        torch.tensor([-100] * (num_latent + 1)),
                        long_decoder_ids
                    ], dim=0)
                    agent2_long_decoder_ids.append(long_decoder_ids)
                    agent2_long_labels.append(long_labels)                
                
                
            # --- Agent1's full_ref_labels ---
            agent1_full_ref_ids = torch.tensor(agent1_full_ref_tokens, dtype=torch.long)
            agent1_full_ref_labels = torch.full_like(agent1_full_ref_ids, -100)
            spans = agent1_response_spans[-1:] if sample_mask_inter_res else agent1_response_spans
            for start, end in spans:
                agent1_full_ref_labels[start:end] = agent1_full_ref_ids[start:end]             
                
            # --- Agent2's full_ref_labels ---
            agent2_full_ref_ids = torch.tensor(agent2_full_ref_tokens, dtype=torch.long)
            agent2_full_ref_labels = torch.full_like(agent2_full_ref_ids, -100)
            spans = agent2_response_spans[-1:] if sample_mask_inter_res else agent2_response_spans
            for start, end in spans:
                agent2_full_ref_labels[start:end] = agent2_full_ref_ids[start:end]           
                
            # save data for this dialog
            agent1_encoder_ids_list.append(agent1_encoder_ids)
            agent1_decoder_ids_list.append(agent1_decoder_ids)
            agent1_labels_list.append(agent1_labels)
            agent1_model_answer_pos_list.append(agent1_model_answer_pos)
            agent1_ref_answer_pos_list.append(agent1_ref_answer_pos)
            agent1_full_ref_tokens_list.append(agent1_full_ref_ids)
            agent1_full_ref_labels_list.append(agent1_full_ref_labels)
            agent1_long_decoder_ids_list.append(agent1_long_decoder_ids)
            agent1_long_labels_list.append(agent1_long_labels)
                
            agent2_encoder_ids_list.append(agent2_encoder_ids)
            agent2_decoder_ids_list.append(agent2_decoder_ids)
            agent2_labels_list.append(agent2_labels)
            agent2_model_answer_pos_list.append(agent2_model_answer_pos)
            agent2_ref_answer_pos_list.append(agent2_ref_answer_pos)
            agent2_full_ref_tokens_list.append(agent2_full_ref_ids)
            agent2_full_ref_labels_list.append(agent2_full_ref_labels)
            agent2_long_decoder_ids_list.append(agent2_long_decoder_ids)
            agent2_long_labels_list.append(agent2_long_labels)
           
            mask_inter_res_list.append(torch.tensor(sample_mask_inter_res, dtype=torch.bool))

        return dict(
            # Agent1
            agent1_encoder_input_ids_list=agent1_encoder_ids_list,
            agent1_decoder_input_ids_list=agent1_decoder_ids_list,
            agent1_labels_list=agent1_labels_list,
            agent1_model_answer_position=agent1_model_answer_pos_list,
            agent1_ref_answer_position=agent1_ref_answer_pos_list,
            agent1_full_ref_input_ids=agent1_full_ref_tokens_list,
            agent1_full_ref_labels=agent1_full_ref_labels_list,
            agent1_long_decoder_input_ids_list=agent1_long_decoder_ids_list,
            agent1_long_labels_list=agent1_long_labels_list,
                
            # Agent2
            agent2_encoder_input_ids_list=agent2_encoder_ids_list,
            agent2_decoder_input_ids_list=agent2_decoder_ids_list,
            agent2_labels_list=agent2_labels_list,
            agent2_model_answer_position=agent2_model_answer_pos_list,
            agent2_ref_answer_position=agent2_ref_answer_pos_list,
            agent2_full_ref_input_ids=agent2_full_ref_tokens_list,
            agent2_full_ref_labels=agent2_full_ref_labels_list,
            agent2_long_decoder_input_ids_list=agent2_long_decoder_ids_list,
            agent2_long_labels_list=agent2_long_labels_list,
            
            mask_inter_res = mask_inter_res_list
            
        )            

    class MASSupervisedDataset(Dataset):
        QUESTION_COM_PROMPT = "\nAnswer the above question. Provide any details you want at first and then give your final answer, decision or conclusion."
        def __init__(self, data_name, raw_data, tokenizer, bot_id, eot_id, latent_id, 
                     bot_token, eot_token, latent_token, num_latent, fixed_num_turns=2, 
                     model_name=None, mask_inter_res=False):
            super(MASSupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")

            self.data_name = data_name
            dialogue_list = []
            num_ops_list = []
            operators = ["+", "-", "*", "/"]

            token_nums = []
            for num_iter, example in enumerate(raw_data):
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break
                
                messages = example["messages"][:1+2*fixed_num_turns]
                if len(messages)<2*fixed_num_turns:continue
                turns = []
                
                for i in range(0, len(messages)):
                    if i==0:
                        # question = f"{messages[i]['content']}"
                        question1 = f"{messages[i]['question_agent1']}"
                        question2 = f"{messages[i]['question_agent2']}"
                    else:
                        # question = ""
                        question1 = ""
                        question2 = ""
                        
                    short_answer, long_answer = f"{messages[i]['short_answer']}" , f"{messages[i]['long_answer']}"
                    cot = long_answer.split(". ")
                    if not (training_args.include_last_cot):
                         cot = cot[:-1]
                    if cot:
                        cot = ". ".join(cot)+".\n"
                    else:
                        cot = ""

                    # answer = short_answer.split(' ')[-1]
                    answer = extract_short_answer(short_answer)
                    # if not answer[0].isdigit():
                    #     continue
                    answer = f"The answer is: {answer}"
                    answer = answer.replace("####", "")
                    # if len(answer) > 30 or '\n' in short_answer:
                    #     print(f"[WARNING] short_answer={repr(short_answer)}")
                    #     print(f"          extracted answer={repr(answer)}")
                    if 'true' in short_answer.lower() or 'false' in short_answer.lower():
                        print(f"[WARNING] short_answer={repr(short_answer)}")
                        print(f"          extracted answer={repr(answer)}")
                    sample_mask_inter_res = example.get("mask_inter_res", mask_inter_res)
    
                    turns.append({
                        "question1": question1,
                        "question2": question2,
                        "cot": cot,
                        "answer": answer,
                        "mask_inter_res": sample_mask_inter_res
                    })                    
            
                token_num = len(tokenizer.encode(" ".join([turn["question1"] + turn["cot"] + turn["answer"] for turn in turns])))
                if token_num > training_args.max_token_num:
                    continue
                if turns:
                    dialogue_list.append(turns)
                
            if training_args.exp_mode:
                dialogue_list = dialogue_list[:training_args.exp_data_num]
            
            print(f"{len(dialogue_list)} dialogues with {sum(len(d) for d in dialogue_list)} turns in total...")
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess_mas(dialogue_list, tokenizer, bot_id, eot_id, latent_id, bot_token, eot_token, 
                                            latent_token, num_latent, model_name, mask_inter_res)
            self.keys = list(self.data_dict.keys())


        def __len__(self):
            return len(self.data_dict["agent1_full_ref_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            max_turns = max(len(inst["agent1_encoder_input_ids_list"]) for inst in instances)
            batch_size = len(instances)
            
            agent1_encoder_input_ids_batch = []
            agent1_decoder_input_ids_batch = []
            agent1_labels_batch = []
            agent1_model_answer_position_batch = []
            agent1_ref_answer_position_batch = []
            agent1_long_decoder_input_ids_batch = []
            agent1_long_labels_batch = []
            
            agent2_encoder_input_ids_batch = []
            agent2_decoder_input_ids_batch = []
            agent2_labels_batch = []
            agent2_model_answer_position_batch = []
            agent2_ref_answer_position_batch = []
            agent2_long_decoder_input_ids_batch = []
            agent2_long_labels_batch = []         
            
            for inst in instances:
                num_turns = len(inst["agent1_encoder_input_ids_list"])
                # padding to max_turns
                agent1_encoder_ids_padded = inst["agent1_encoder_input_ids_list"] + \
                    [torch.zeros(1, dtype=torch.long)] * (max_turns - num_turns)
                agent1_decoder_ids_padded = inst["agent1_decoder_input_ids_list"] + \
                    [torch.zeros(1, dtype=torch.long)] * (max_turns - num_turns)
                agent1_labels_padded = inst["agent1_labels_list"] + \
                    [torch.full((1,), IGNORE_INDEX, dtype=torch.long)] * (max_turns - num_turns)
                agent1_model_pos_padded = inst["agent1_model_answer_position"] + \
                    [0] * (max_turns - num_turns)
                agent1_ref_pos_padded = inst["agent1_ref_answer_position"] + \
                    [0] * (max_turns - num_turns)
                agent1_long_decoder_padded = inst["agent1_long_decoder_input_ids_list"] + \
                    [torch.zeros(1, dtype=torch.long)] * (max_turns - num_turns)
                agent1_long_labels_padded = inst["agent1_long_labels_list"] + \
                    [torch.full((1,), IGNORE_INDEX, dtype=torch.long)] * (max_turns - num_turns)
                    
                agent2_encoder_ids_padded = inst["agent2_encoder_input_ids_list"] + \
                    [torch.zeros(1, dtype=torch.long)] * (max_turns - num_turns)
                agent2_decoder_ids_padded = inst["agent2_decoder_input_ids_list"] + \
                    [torch.zeros(1, dtype=torch.long)] * (max_turns - num_turns)
                agent2_labels_padded = inst["agent2_labels_list"] + \
                    [torch.full((1,), IGNORE_INDEX, dtype=torch.long)] * (max_turns - num_turns)
                agent2_model_pos_padded = inst["agent2_model_answer_position"] + \
                    [0] * (max_turns - num_turns)
                agent2_ref_pos_padded = inst["agent2_ref_answer_position"] + \
                    [0] * (max_turns - num_turns)
                agent2_long_decoder_padded = inst["agent2_long_decoder_input_ids_list"] + \
                    [torch.zeros(1, dtype=torch.long)] * (max_turns - num_turns)
                agent2_long_labels_padded = inst["agent2_long_labels_list"] + \
                    [torch.full((1,), IGNORE_INDEX, dtype=torch.long)] * (max_turns - num_turns)
                
                agent1_encoder_input_ids_batch.append(agent1_encoder_ids_padded)
                agent1_decoder_input_ids_batch.append(agent1_decoder_ids_padded)
                agent1_labels_batch.append(agent1_labels_padded)
                agent1_model_answer_position_batch.append(agent1_model_pos_padded)
                agent1_ref_answer_position_batch.append(agent1_ref_pos_padded)
                agent1_long_decoder_input_ids_batch.append(agent1_long_decoder_padded)
                agent1_long_labels_batch.append(agent1_long_labels_padded)

                agent2_encoder_input_ids_batch.append(agent2_encoder_ids_padded)
                agent2_decoder_input_ids_batch.append(agent2_decoder_ids_padded)
                agent2_labels_batch.append(agent2_labels_padded)
                agent2_model_answer_position_batch.append(agent2_model_pos_padded)
                agent2_ref_answer_position_batch.append(agent2_ref_pos_padded)
                agent2_long_decoder_input_ids_batch.append(agent2_long_decoder_padded)
                agent2_long_labels_batch.append(agent2_long_labels_padded)
            
            # padding each turn for agent1
            max_encoder_len = 0
            max_decoder_len = 0
            max_long_decoder_len = 0 
            max_long_labels_len = 0
            
            for turn_idx in range(max_turns):
                turn_encoder = [inst[turn_idx] for inst in agent1_encoder_input_ids_batch]
                turn_decoder = [inst[turn_idx] for inst in agent1_decoder_input_ids_batch]
                turn_labels = [inst[turn_idx] for inst in agent1_labels_batch]
                turn_long_decoder = [inst[turn_idx] for inst in agent1_long_decoder_input_ids_batch]
                turn_long_labels = [inst[turn_idx] for inst in agent1_long_labels_batch]    
                
                max_encoder_len = max(max_encoder_len, max(seq.size(0) for seq in turn_encoder))
                max_decoder_len = max(max_decoder_len, max(seq.size(0) for seq in turn_decoder))
                max_long_decoder_len = max(max_long_decoder_len, max(seq.size(0) for seq in turn_long_decoder))
                max_long_labels_len = max(max_long_labels_len, max(seq.size(0) for seq in turn_long_labels))
         
            agent1_encoder_input_ids_final = []
            agent1_decoder_input_ids_final = []
            agent1_labels_final = []
            agent1_long_decoder_input_ids_final = []
            agent1_long_labels_final = []        
            
            for turn_idx in range(max_turns):
                turn_encoder = [inst[turn_idx] for inst in agent1_encoder_input_ids_batch]
                turn_decoder = [inst[turn_idx] for inst in agent1_decoder_input_ids_batch]
                turn_labels = [inst[turn_idx] for inst in agent1_labels_batch]
                turn_long_decoder = [inst[turn_idx] for inst in agent1_long_decoder_input_ids_batch]
                turn_long_labels = [inst[turn_idx] for inst in agent1_long_labels_batch]  
                
                # padding - encoder (left)
                reversed_encoder = [seq.flip(0) for seq in turn_encoder]
                padded_encoder = torch.nn.utils.rnn.pad_sequence(
                    reversed_encoder, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).flip(1)
                if padded_encoder.size(1) < max_encoder_len:
                    padding_size = max_encoder_len - padded_encoder.size(1)
                    # Left padding: 在左边补0
                    padded_encoder = torch.cat([
                        torch.full((batch_size, padding_size), self.tokenizer.pad_token_id, dtype=torch.long),
                        padded_encoder
                    ], dim=1)
                
                # padding - decoder (right)
                padded_decoder = torch.nn.utils.rnn.pad_sequence(
                    turn_decoder, batch_first=True, padding_value=self.tokenizer.pad_token_id
                )         
                if padded_decoder.size(1) < max_decoder_len:
                    padding_size = max_decoder_len - padded_decoder.size(1)
                    padded_decoder = torch.cat([
                        padded_decoder,
                        torch.full((batch_size, padding_size), self.tokenizer.pad_token_id, dtype=torch.long)
                    ], dim=1)        
                    
                padded_labels = torch.nn.utils.rnn.pad_sequence(
                    turn_labels, batch_first=True, padding_value=IGNORE_INDEX
                )
                if padded_labels.size(1) < max_decoder_len:
                    padding_size = max_decoder_len - padded_labels.size(1)
                    padded_labels = torch.cat([
                        padded_labels,
                        torch.full((batch_size, padding_size), IGNORE_INDEX, dtype=torch.long)
                    ], dim=1)   
                             
                padded_long_decoder = torch.nn.utils.rnn.pad_sequence(
                    turn_long_decoder, batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
                if padded_long_decoder.size(1) < max_long_decoder_len:
                    padding_size = max_long_decoder_len - padded_long_decoder.size(1)
                    padded_long_decoder = torch.cat([
                        padded_long_decoder,
                        torch.full((batch_size, padding_size), self.tokenizer.pad_token_id, dtype=torch.long)
                    ], dim=1)
                
                padded_long_labels = torch.nn.utils.rnn.pad_sequence(
                    turn_long_labels, batch_first=True, padding_value=IGNORE_INDEX
                )
                if padded_long_labels.size(1) < max_long_labels_len:
                    padding_size = max_long_labels_len - padded_long_labels.size(1)
                    padded_long_labels = torch.cat([
                        padded_long_labels,
                        torch.full((batch_size, padding_size), IGNORE_INDEX, dtype=torch.long)
                    ], dim=1)
                agent1_encoder_input_ids_final.append(padded_encoder)
                agent1_decoder_input_ids_final.append(padded_decoder)
                agent1_labels_final.append(padded_labels)
                agent1_long_decoder_input_ids_final.append(padded_long_decoder)
                agent1_long_labels_final.append(padded_long_labels)                       
          
            # Stack: [T, B, L] -> [B, T, L]
            agent1_encoder_input_ids_stacked = torch.stack(agent1_encoder_input_ids_final, dim=0).transpose(0, 1)
            agent1_decoder_input_ids_stacked = torch.stack(agent1_decoder_input_ids_final, dim=0).transpose(0, 1)
            agent1_labels_stacked = torch.stack(agent1_labels_final, dim=0).transpose(0, 1)
            agent1_long_decoder_input_ids_stacked = torch.stack(agent1_long_decoder_input_ids_final, dim=0).transpose(0, 1)
            agent1_long_labels_stacked = torch.stack(agent1_long_labels_final, dim=0).transpose(0, 1)          

            agent1_full_ref_input_ids = [inst["agent1_full_ref_input_ids"] for inst in instances]
            agent1_full_ref_labels = [inst["agent1_full_ref_labels"] for inst in instances]
            
            agent1_full_ref_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                agent1_full_ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            agent1_full_ref_labels_padded = torch.nn.utils.rnn.pad_sequence(
                agent1_full_ref_labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            
            agent1_model_answer_position_tensor = torch.tensor(
                agent1_model_answer_position_batch, dtype=torch.long
            )
            agent1_ref_answer_position_tensor = torch.tensor(
                agent1_ref_answer_position_batch, dtype=torch.long
            )
            
            # padding each turn for agent2
            max_encoder_len = 0
            max_decoder_len = 0
            max_long_decoder_len = 0 
            max_long_labels_len = 0
            
            for turn_idx in range(max_turns):
                turn_encoder = [inst[turn_idx] for inst in agent2_encoder_input_ids_batch]
                turn_decoder = [inst[turn_idx] for inst in agent2_decoder_input_ids_batch]
                turn_labels = [inst[turn_idx] for inst in agent2_labels_batch]
                turn_long_decoder = [inst[turn_idx] for inst in agent2_long_decoder_input_ids_batch]
                turn_long_labels = [inst[turn_idx] for inst in agent2_long_labels_batch]    
                
                max_encoder_len = max(max_encoder_len, max(seq.size(0) for seq in turn_encoder))
                max_decoder_len = max(max_decoder_len, max(seq.size(0) for seq in turn_decoder))
                max_long_decoder_len = max(max_long_decoder_len, max(seq.size(0) for seq in turn_long_decoder))
                max_long_labels_len = max(max_long_labels_len, max(seq.size(0) for seq in turn_long_labels))
         
            agent2_encoder_input_ids_final = []
            agent2_decoder_input_ids_final = []
            agent2_labels_final = []
            agent2_long_decoder_input_ids_final = []
            agent2_long_labels_final = []        
            
            for turn_idx in range(max_turns):
                turn_encoder = [inst[turn_idx] for inst in agent2_encoder_input_ids_batch]
                turn_decoder = [inst[turn_idx] for inst in agent2_decoder_input_ids_batch]
                turn_labels = [inst[turn_idx] for inst in agent2_labels_batch]
                turn_long_decoder = [inst[turn_idx] for inst in agent2_long_decoder_input_ids_batch]
                turn_long_labels = [inst[turn_idx] for inst in agent2_long_labels_batch]  
                
                # padding - encoder (left)
                reversed_encoder = [seq.flip(0) for seq in turn_encoder]
                padded_encoder = torch.nn.utils.rnn.pad_sequence(
                    reversed_encoder, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).flip(1)
                if padded_encoder.size(1) < max_encoder_len:
                    padding_size = max_encoder_len - padded_encoder.size(1)
                    # Left padding: 在左边补0
                    padded_encoder = torch.cat([
                        torch.full((batch_size, padding_size), self.tokenizer.pad_token_id, dtype=torch.long),
                        padded_encoder
                    ], dim=1)
                
                # padding - decoder (right)
                padded_decoder = torch.nn.utils.rnn.pad_sequence(
                    turn_decoder, batch_first=True, padding_value=self.tokenizer.pad_token_id
                )         
                if padded_decoder.size(1) < max_decoder_len:
                    padding_size = max_decoder_len - padded_decoder.size(1)
                    padded_decoder = torch.cat([
                        padded_decoder,
                        torch.full((batch_size, padding_size), self.tokenizer.pad_token_id, dtype=torch.long)
                    ], dim=1)        
                    
                padded_labels = torch.nn.utils.rnn.pad_sequence(
                    turn_labels, batch_first=True, padding_value=IGNORE_INDEX
                )
                if padded_labels.size(1) < max_decoder_len:
                    padding_size = max_decoder_len - padded_labels.size(1)
                    padded_labels = torch.cat([
                        padded_labels,
                        torch.full((batch_size, padding_size), IGNORE_INDEX, dtype=torch.long)
                    ], dim=1)   
                             
                padded_long_decoder = torch.nn.utils.rnn.pad_sequence(
                    turn_long_decoder, batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
                if padded_long_decoder.size(1) < max_long_decoder_len:
                    padding_size = max_long_decoder_len - padded_long_decoder.size(1)
                    padded_long_decoder = torch.cat([
                        padded_long_decoder,
                        torch.full((batch_size, padding_size), self.tokenizer.pad_token_id, dtype=torch.long)
                    ], dim=1)
                
                padded_long_labels = torch.nn.utils.rnn.pad_sequence(
                    turn_long_labels, batch_first=True, padding_value=IGNORE_INDEX
                )
                if padded_long_labels.size(1) < max_long_labels_len:
                    padding_size = max_long_labels_len - padded_long_labels.size(1)
                    padded_long_labels = torch.cat([
                        padded_long_labels,
                        torch.full((batch_size, padding_size), IGNORE_INDEX, dtype=torch.long)
                    ], dim=1)
                agent2_encoder_input_ids_final.append(padded_encoder)
                agent2_decoder_input_ids_final.append(padded_decoder)
                agent2_labels_final.append(padded_labels)
                agent2_long_decoder_input_ids_final.append(padded_long_decoder)
                agent2_long_labels_final.append(padded_long_labels)                       
          
            # Stack: [T, B, L] -> [B, T, L]
            agent2_encoder_input_ids_stacked = torch.stack(agent2_encoder_input_ids_final, dim=0).transpose(0, 1)
            agent2_decoder_input_ids_stacked = torch.stack(agent2_decoder_input_ids_final, dim=0).transpose(0, 1)
            agent2_labels_stacked = torch.stack(agent2_labels_final, dim=0).transpose(0, 1)
            agent2_long_decoder_input_ids_stacked = torch.stack(agent2_long_decoder_input_ids_final, dim=0).transpose(0, 1)
            agent2_long_labels_stacked = torch.stack(agent2_long_labels_final, dim=0).transpose(0, 1)          

            agent2_full_ref_input_ids = [inst["agent2_full_ref_input_ids"] for inst in instances]
            agent2_full_ref_labels = [inst["agent2_full_ref_labels"] for inst in instances]
            
            agent2_full_ref_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                agent2_full_ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            agent2_full_ref_labels_padded = torch.nn.utils.rnn.pad_sequence(
                agent2_full_ref_labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            
            agent2_model_answer_position_tensor = torch.tensor(
                agent2_model_answer_position_batch, dtype=torch.long
            )
            agent2_ref_answer_position_tensor = torch.tensor(
                agent2_ref_answer_position_batch, dtype=torch.long
            )            

            mask_inter_res_batch = torch.stack(
                [inst["mask_inter_res"] for inst in instances]
            )              
          
            return dict(
                agent1_encoder_input_ids_list=agent1_encoder_input_ids_stacked,  # [B, T, L_q]
                agent1_decoder_input_ids_list=agent1_decoder_input_ids_stacked,  # [B, T, L_a]
                agent1_labels_list=agent1_labels_stacked,  # [B, T, L_a]
                agent1_encoder_attention_mask_list=agent1_encoder_input_ids_stacked.ne(self.tokenizer.pad_token_id),  # [B, T, L_q]
                agent1_model_answer_position=agent1_model_answer_position_tensor,  # [B, T]
                agent1_ref_answer_position=agent1_ref_answer_position_tensor,  # [B, T]
                
                agent1_full_ref_input_ids=agent1_full_ref_input_ids_padded,  # [B, L_total]
                agent1_full_ref_labels=agent1_full_ref_labels_padded,  # [B, L_total]
                agent1_full_ref_attention_mask=agent1_full_ref_input_ids_padded.ne(self.tokenizer.pad_token_id),

                agent1_long_decoder_input_ids_list=agent1_long_decoder_input_ids_stacked,  # [B, T, L_c]
                agent1_long_labels_list=agent1_long_labels_stacked,  # [B, T, L_c]
                
                agent2_encoder_input_ids_list=agent2_encoder_input_ids_stacked,  # [B, T, L_q]
                agent2_decoder_input_ids_list=agent2_decoder_input_ids_stacked,  # [B, T, L_a]
                agent2_labels_list=agent2_labels_stacked,  # [B, T, L_a]
                agent2_encoder_attention_mask_list=agent2_encoder_input_ids_stacked.ne(self.tokenizer.pad_token_id),  # [B, T, L_q]
                agent2_model_answer_position=agent2_model_answer_position_tensor,  # [B, T]
                agent2_ref_answer_position=agent2_ref_answer_position_tensor,  # [B, T]
                
                agent2_full_ref_input_ids=agent2_full_ref_input_ids_padded,  # [B, L_total]
                agent2_full_ref_labels=agent2_full_ref_labels_padded,  # [B, L_total]
                agent2_full_ref_attention_mask=agent2_full_ref_input_ids_padded.ne(self.tokenizer.pad_token_id),

                agent2_long_decoder_input_ids_list=agent2_long_decoder_input_ids_stacked,  # [B, T, L_c]
                agent2_long_labels_list=agent2_long_labels_stacked,  # [B, T, L_c]
                
                mask_inter_res = mask_inter_res_batch
            )

    def make_supervised_data_module(tokenizer, data_args, model_args, training_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Downloading Data")

        # dataset = load_dataset(data_args.data_name)["train"]
        dataset = load_dataset("json", data_files=data_args.data_name)["train"]
        train_dataset = MASSupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot_id=model.bot_id, eot_id=model.eot_id, latent_id=model.latent_id,
                        bot_token=model.bot_token, eot_token=model.eot_token, latent_token=model.latent_token, 
                        num_latent=model.num_latent, fixed_num_turns=data_args.fixed_num_turns, 
                        model_name=model_args.model_name_or_path.split('/')[-1].lower(),
                        mask_inter_res = training_args.mask_inter_res)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args, training_args=training_args)
    training_args.learning_rate = float(training_args.learning_rate)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    # train()
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str,
                        default="configs/stage2_llama_mixed.yaml") 
    args = parser.parse_args()
    train(args)
