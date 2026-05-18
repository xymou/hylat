"""
HyLAT: multi-agent multi-turn dialogs with hybrid latent-and-text communication
"""
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from peft import (
    get_peft_model,
    PeftModel,
    PeftConfig
)
from torch.nn.functional import gelu
import math
from safetensors.torch import load_file
from transformers.modeling_outputs import ModelOutput
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(default=True, metadata={"help": "whether use int4 for the base model"})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir for inference."})
    stage1_ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir of stage1."})

@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})
    fixed_num_turns: int = field(default=2, metadata={"help": "fixed number of turns of the dialogs"})
    output_path: str = field(
        default=None, metadata={"help": "Path to predicted output."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    icot_train_path: str = field(default="/users/k24020023/efficient_cot/icae/code/coconut/icot_gsm8k/train.txt", metadata={"help":"The training data path"})
    num_latent: int = field(default=5, metadata={"help": "The number of latent for training or inference."})
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding during inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial number of data. for debugging."})
    exp_data_num: int = field(default=10000, metadata={"help": "The number of data used in exp mode"}) 
    use_prj: bool = field(default=False, metadata={"help": "Use a prj module after the llm for latent generation."}) 
    prj_dim: int = field(default=2048, metadata={"help": "The hidden dim of the projection module."})
    prj_dropout: float = field(default=0.0, metadata={"help": "Dropout ratio of the projection module."})
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove the Layer Norm layer for the projection module."})
    distill_loss_div_std: bool = field(default=False, metadata={"help": "Divide the distillation loss by a std for normallisation."})
    distill_loss_type: str = field(default="smooth_l1", metadata={"help": "Specify the distillation loss. Use smoothL1 by default."})
    distill_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    ref_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(default=5, metadata={"help": "Run multiple times during inference"})
    remove_eos: bool = field(default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."})
    print_ref_model_stats: bool = field(default=False, metadata={"help": "Print some stats for the teacher task."})
    include_last_cot: bool = field(default=False, metadata={"help": "Include the last CoT step in the training data."})
    fix_attn_mask: bool = field(default=False, metadata={"help": "Correct a bug about attention mask."})
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000, metadata={"help": "Limit the longest data to avoid OOM."})
    decode_loss_factor: float = field(default=0.1, metadata={"help": "A multiplier of the decode loss."})
    use_decode: bool = field(default=False)
    mask_inter_res: bool = field(default=False)

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

class HyLAT(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        model_wrapper_class = AutoModelForCausalLM 
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                    quantization_config=transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type='nf4',
                    )
                )


        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2
        self.ae_id = ori_vocab_size + 3
        self.latent_id = ori_vocab_size + 4
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        dummy_tokens = [f"<dummy_{i}>" for i in range(ori_vocab_size - len(self.tokenizer))]
        if dummy_tokens:
            self.tokenizer.add_tokens(dummy_tokens, special_tokens=False)
        self.tokenizer.add_special_tokens({
            "pad_token": "<pad>",
            "additional_special_tokens": ["<bot>", "<eot>", "<ae>", "<latent>"]
        })        
        self.bot_token = "<bot>"
        self.eot_token = "<eot>"
        self.latent_token = "<latent>"

        self.codi.resize_token_embeddings(
            ori_vocab_size + 5
        )  # dummy values for mem tokens
        
        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent

        # LoRA
        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
            self.prj.to(self.codi.dtype)
            
        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
        
        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError
        
        # Decode Loss
        self.use_decode = training_args.use_decode
        self.decode_loss_factor = training_args.decode_loss_factor

        # general 
        self.fix_attn_mask = training_args.fix_attn_mask
        self.mask_inter_res = training_args.mask_inter_res

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training:
            self.init()

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception: # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception: # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def init(self):
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

        

    def forward(
        self,
        # Agent1
        agent1_encoder_input_ids_list: torch.LongTensor = None, # [B/2, T, L_q]
        agent1_decoder_input_ids_list: torch.LongTensor = None,
        agent1_labels_list: torch.LongTensor = None,
        agent1_encoder_attention_mask_list: torch.LongTensor = None,
        agent1_model_answer_position: torch.LongTensor = None,
        agent1_full_ref_input_ids: torch.LongTensor = None,
        agent1_full_ref_labels: torch.LongTensor = None,
        agent1_full_ref_attention_mask: torch.LongTensor = None,
        agent1_ref_answer_position: torch.LongTensor = None,
        agent1_long_decoder_input_ids_list: torch.LongTensor = None,
        agent1_long_labels_list: torch.LongTensor = None,
        
        # Agent2
        agent2_encoder_input_ids_list: torch.LongTensor = None,
        agent2_decoder_input_ids_list: torch.LongTensor = None,
        agent2_labels_list: torch.LongTensor = None,
        agent2_encoder_attention_mask_list: torch.LongTensor = None,
        agent2_model_answer_position: torch.LongTensor = None,
        agent2_full_ref_input_ids: torch.LongTensor = None,
        agent2_full_ref_labels: torch.LongTensor = None,
        agent2_full_ref_attention_mask: torch.LongTensor = None,
        agent2_ref_answer_position: torch.LongTensor = None,
        agent2_long_decoder_input_ids_list: torch.LongTensor = None,
        agent2_long_labels_list: torch.LongTensor = None,
        
        step: int = None,
        step_ratio: float = None,
        mask_inter_res: torch.BoolTensor = None,
    ):
        device = agent1_full_ref_input_ids.device
        
        batch_size = agent1_encoder_input_ids_list.size(0)
        num_turns = agent1_encoder_input_ids_list.size(1)
        loss_detail = {
            "agent1_teacher_ce": [],
            "agent1_student_ce": [],
            "agent1_distill": [],
            "agent1_decode": [],
            "agent1_total": [],
            "agent2_teacher_ce": [],
            "agent2_student_ce": [],
            "agent2_distill": [],
            "agent2_decode": [],
            "agent2_total": []
        }
        
        # Transpose: [B, T, L] -> [T, B, L]
        agent1_encoder_input_ids_list = agent1_encoder_input_ids_list.transpose(0, 1)
        agent1_decoder_input_ids_list = agent1_decoder_input_ids_list.transpose(0, 1)
        agent1_labels_list = agent1_labels_list.transpose(0, 1)
        agent1_encoder_attention_mask_list = agent1_encoder_attention_mask_list.transpose(0, 1)
        agent1_long_decoder_input_ids_list = agent1_long_decoder_input_ids_list.transpose(0, 1)
        agent1_long_labels_list = agent1_long_labels_list.transpose(0, 1)
        
        agent2_encoder_input_ids_list = agent2_encoder_input_ids_list.transpose(0, 1)
        agent2_decoder_input_ids_list = agent2_decoder_input_ids_list.transpose(0, 1)
        agent2_labels_list = agent2_labels_list.transpose(0, 1)
        agent2_encoder_attention_mask_list = agent2_encoder_attention_mask_list.transpose(0, 1)
        agent2_long_decoder_input_ids_list = agent2_long_decoder_input_ids_list.transpose(0, 1)
        agent2_long_labels_list = agent2_long_labels_list.transpose(0, 1)
        
        # Position normalization
        # if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower():
        #     agent1_model_answer_position = agent1_model_answer_position + 1
        #     agent1_ref_answer_position = agent1_ref_answer_position + 1
        #     agent2_model_answer_position = agent2_model_answer_position + 1
        #     agent2_ref_answer_position = agent2_ref_answer_position + 1
        
        # ===== Teacher Task =====
        # Agent1 teacher
        with torch.no_grad():
            agent1_ref_outputs = self.codi(
                input_ids=agent1_full_ref_input_ids,
                output_hidden_states=True,
                attention_mask=agent1_full_ref_attention_mask if self.fix_attn_mask else None
            )
        agent1_ref_outputs_with_grad = self.codi(
            input_ids=agent1_full_ref_input_ids,
            output_hidden_states=True,
            attention_mask=agent1_full_ref_attention_mask if self.fix_attn_mask else None
        )
        
        agent1_ref_logits = agent1_ref_outputs_with_grad.logits
        agent1_ref_loss = self.loss_fct(
            agent1_ref_logits[:, :-1, :].reshape(-1, agent1_ref_logits.size(-1)),
            agent1_full_ref_labels[:, 1:].reshape(-1)
        )
        agent1_ref_loss *= self.ref_loss_factor
        
        # per-turn teacher loss
        shift_logits = agent1_ref_logits[:, :-1, :]
        shift_labels = agent1_full_ref_labels[:, 1:]        
        token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none"
        ).view(shift_labels.size()) 
        for t in range(num_turns):
            start = agent1_ref_answer_position[:, t]
            if t < num_turns - 1:
                end = agent1_ref_answer_position[:, t + 1] - 5
            else:
                end = [token_ce.size(1)]*token_ce.size(0)

            # mask each sample separately
            turn_losses = []
            for b in range(token_ce.size(0)):
                turn_losses.append(token_ce[b, start[b]:end[b]].mean())
                pos = agent1_ref_answer_position[b, t].item()
                if pos < agent1_full_ref_input_ids.size(1):
                    token_at_pos = agent1_full_ref_input_ids[b, pos].item()
                    decoded = self.tokenizer.decode([token_at_pos])
                else:
                    decoded = "OUT_OF_RANGE"
                model_pos_debug = agent1_model_answer_position[b, t]
                model_token_at_pos = agent1_decoder_input_ids_list[t, b, model_pos_debug]
                assert model_token_at_pos==token_at_pos
                
                segment = token_ce[b, start[b]:end[b]]
                label_segment = shift_labels[b, start[b]:end[b]]
                valid_mask = label_segment != IGNORE_INDEX
                loss_val = segment.mean().item()
                
                # if loss_val > 5.0:
                #     print(f"[Agent1 Teacher] b={b}, turn={t}, loss={loss_val:.4f}")
                #     print(f"  ref_pos={pos}, token_at_pos='{decoded}'")  # 应该是"The"
                #     print(f"  loc [{start[b]}, {end[b]}], valid token num={valid_mask.sum().item()}")
                #     print(f"  first 20 token: '{self.tokenizer.decode(agent1_full_ref_input_ids[b, start[b]:start[b]+20].tolist())}'")

            loss_detail["agent1_teacher_ce"].append(torch.stack(turn_losses).mean().item())           
        
        
        # Agent2 teacher
        with torch.no_grad():
            agent2_ref_outputs = self.codi(
                input_ids=agent2_full_ref_input_ids,
                output_hidden_states=True,
                attention_mask=agent2_full_ref_attention_mask if self.fix_attn_mask else None
            )
        agent2_ref_outputs_with_grad = self.codi(
            input_ids=agent2_full_ref_input_ids,
            output_hidden_states=True,
            attention_mask=agent2_full_ref_attention_mask if self.fix_attn_mask else None
        )
        
        agent2_ref_logits = agent2_ref_outputs_with_grad.logits
        agent2_ref_loss = self.loss_fct(
            agent2_ref_logits[:, :-1, :].reshape(-1, agent2_ref_logits.size(-1)),
            agent2_full_ref_labels[:, 1:].reshape(-1)
        )
        agent2_ref_loss *= self.ref_loss_factor
        
        # per-turn teacher loss
        shift_logits = agent2_ref_logits[:, :-1, :]
        shift_labels = agent2_full_ref_labels[:, 1:]        
        token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none"
        ).view(shift_labels.size()) 
        for t in range(num_turns):
            start = agent2_ref_answer_position[:, t]
            if t < num_turns - 1:
                end = agent2_ref_answer_position[:, t + 1] - 1
            else:
                end = [token_ce.size(1)]*token_ce.size(0)

            # mask each sample separately
            turn_losses = []
            for b in range(token_ce.size(0)):
                turn_losses.append(token_ce[b, start[b]:end[b]].mean())

            loss_detail["agent2_teacher_ce"].append(torch.stack(turn_losses).mean().item())           
        
        
        total_ref_loss = agent1_ref_loss + agent2_ref_loss
        
        # ===== Student Task: Alternately process two agents =====
        agent1_past_kv = None
        agent2_past_kv = None
        agent1_gen_latents = []
        agent2_gen_latents = []
        
        total_ce_loss = 0
        total_distill_loss = 0
        total_decode_loss = 0
        
        for turn_idx in range(num_turns):
            is_last_turn = (turn_idx == num_turns - 1)
            if mask_inter_res is not None:
                sample_should_compute = (~mask_inter_res) | is_last_turn  # [B], bool
            else:
                sample_should_compute = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            # ========== Agent1's turn ==========
            agent1_encoder_ids = agent1_encoder_input_ids_list[turn_idx]
            agent1_decoder_ids = agent1_decoder_input_ids_list[turn_idx]
            agent1_labels = agent1_labels_list[turn_idx]
            agent1_encoder_mask = agent1_encoder_attention_mask_list[turn_idx]
            agent1_long_decoder_ids = agent1_long_decoder_input_ids_list[turn_idx]
            agent1_long_labels = agent1_long_labels_list[turn_idx]
            
            agent1_cur_model_pos = agent1_model_answer_position[:, turn_idx] - 1
            agent1_cur_ref_pos = agent1_ref_answer_position[:, turn_idx] - 1
            
            # 1. construct encoder embeds: fill the historical latent
            agent1_encoder_embeds = self.get_embd(self.codi, self.model_name)(agent1_encoder_ids)
            
            if turn_idx > 0 and len(agent2_gen_latents) > 0:
                # inject latent of agent 2
                agent1_encoder_embeds = self._inject_other_agent_latent(
                    encoder_input_ids=agent1_encoder_ids,
                    encoder_embeds=agent1_encoder_embeds,
                    other_latents=agent2_gen_latents[turn_idx-1],  # latent of agent2
                )
            
            # 2. Encode question
            # process the attention mask
            if agent1_past_kv is not None:
                past_length = agent1_past_kv[0][0].size(2)  # [B, num_heads, past_len, head_dim]
            else:
                past_length = 0
            if past_length:
                past_mask = torch.ones(
                    agent1_encoder_embeds.size(0), past_length,
                    dtype=agent1_encoder_mask.dtype,
                    device=agent1_encoder_mask.device
                )
                agent1_encoder_mask = torch.cat([past_mask, agent1_encoder_mask], dim=1)                    
            
            outputs = self.codi(
                inputs_embeds=agent1_encoder_embeds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=agent1_past_kv,
                attention_mask=agent1_encoder_mask
            )
            agent1_past_kv = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            
            if self.use_prj:
                latent_embd = self.prj(latent_embd)
            
            # 3. Generate latent
            latent_list = []
            for i in range(self.num_latent):
                outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=agent1_past_kv
                )
                agent1_past_kv = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)
                
                latent_list.append(latent_embd)
            
            agent1_current_latents = torch.cat(latent_list, dim=1)  # [B, num_latent, H]
            agent1_gen_latents.append(agent1_current_latents)
            
            # 4. Generate answer
            agent1_decoder_embeds = self.get_embd(self.codi, self.model_name)(agent1_decoder_ids)
            
            dynamic_mask = None
            if self.fix_attn_mask:
                dynamic_mask = torch.cat([
                    agent1_encoder_mask,
                    torch.ones((batch_size, self.num_latent), device=device),
                    torch.ones((batch_size, agent1_decoder_embeds.size(1)), device=device)
                ], dim=1).bool()
            
            outputs = self.codi(
                inputs_embeds=agent1_decoder_embeds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=agent1_past_kv,
                attention_mask=dynamic_mask
            )
            agent1_past_kv = outputs.past_key_values
            
            # 5. Compute agent1 losses
            # CE loss
            logits = outputs.logits
            labels = agent1_labels
            masked_labels = labels.clone()
            masked_labels[~sample_should_compute] = IGNORE_INDEX

            valid_token_count = (masked_labels[:, 1:] != IGNORE_INDEX).sum()
            if valid_token_count > 0:
                ce_loss = self.loss_fct(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    masked_labels[:, 1:].reshape(-1)
                )
            else:
                ce_loss = torch.tensor(0.0, device=device).to(logits)
            total_ce_loss += ce_loss
                
            # Distill loss
            # distill_loss = 0
            distill_loss = torch.tensor(0.0, device=device).to(ce_loss)
            for j, (out, ref_out) in enumerate(zip(outputs.hidden_states, agent1_ref_outputs.hidden_states)):
                # print(agent1_decoder_input_ids_list[0,0,agent1_cur_model_pos[0]], agent1_full_ref_input_ids[0,agent1_cur_ref_pos[0]])
                ref_selected = ref_out.gather(
                    1, agent1_cur_ref_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1))
                )
                out_selected = out.gather(
                    1, agent1_cur_model_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1))
                )
                valid = sample_should_compute
                if valid.any():
                    distill_loss_tmp = self.distill_loss_fct(
                        out_selected[valid], ref_selected[valid].detach()
                    )

                    if self.distill_loss_div_std:
                        std = ref_selected[valid].std()
                        if std>0:
                            if self.distill_loss_type == 'l2':
                                distill_loss_tmp /= std
                            distill_loss_tmp /= std
                    distill_loss += distill_loss_tmp
                
            distill_loss /= len(outputs.hidden_states)
            distill_loss *= self.distill_loss_factor        
            total_distill_loss += distill_loss
                
            # Decode loss
            if self.use_decode:
                decode_loss = self.decoder_forward(
                    agent1_current_latents, agent1_long_decoder_ids, agent1_long_labels
                )
            else:
                decode_loss = torch.tensor(0).to(ce_loss)
            decode_loss *= self.decode_loss_factor
            total_decode_loss += decode_loss
                
            # record per-turn loss
            turn_ce_loss = ce_loss
            turn_distill_loss = distill_loss
            turn_decode_loss = decode_loss
            turn_total_loss = (
                turn_ce_loss
                + turn_distill_loss
                + turn_decode_loss
            ) 
            loss_detail["agent1_student_ce"].append(turn_ce_loss.detach().item())
            loss_detail["agent1_distill"].append(turn_distill_loss.detach().item())
            loss_detail["agent1_decode"].append(turn_decode_loss.detach().item())
            loss_detail["agent1_total"].append(turn_total_loss.detach().item())            
         
            
            # ========== Agent2's turn ==========
            agent2_encoder_ids = agent2_encoder_input_ids_list[turn_idx]
            agent2_decoder_ids = agent2_decoder_input_ids_list[turn_idx]
            agent2_labels = agent2_labels_list[turn_idx]
            agent2_encoder_mask = agent2_encoder_attention_mask_list[turn_idx]
            agent2_long_decoder_ids = agent2_long_decoder_input_ids_list[turn_idx]
            agent2_long_labels = agent2_long_labels_list[turn_idx]
            
            agent2_cur_model_pos = agent2_model_answer_position[:, turn_idx] - 1
            agent2_cur_ref_pos = agent2_ref_answer_position[:, turn_idx] - 1
            
            # 1. construct encoder embeds: fill historical latent (agent1's latent in current turn)
            agent2_encoder_embeds = self.get_embd(self.codi, self.model_name)(agent2_encoder_ids)
            
            if turn_idx > 0 and len(agent1_gen_latents) > 0:
                agent2_encoder_embeds = self._inject_other_agent_latent(
                    encoder_input_ids=agent2_encoder_ids,
                    encoder_embeds=agent2_encoder_embeds,
                    other_latents=agent1_gen_latents[turn_idx-1]
                )
            
            # 2. Encode question
            # process the attention mask
            if agent2_past_kv is not None:
                past_length = agent2_past_kv[0][0].size(2)  # [B, num_heads, past_len, head_dim]
            else:
                past_length = 0
            if past_length:
                past_mask = torch.ones(
                    agent2_encoder_embeds.size(0), past_length,
                    dtype=agent2_encoder_mask.dtype,
                    device=agent2_encoder_mask.device
                )
                agent2_encoder_mask = torch.cat([past_mask, agent2_encoder_mask], dim=1)

            outputs = self.codi(
                inputs_embeds=agent2_encoder_embeds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=agent2_past_kv,
                attention_mask=agent2_encoder_mask
            )
            agent2_past_kv = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            
            if self.use_prj:
                latent_embd = self.prj(latent_embd)
            
            # 3. Generate latent
            latent_list = []
            for i in range(self.num_latent):
                outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=agent2_past_kv
                )
                agent2_past_kv = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)
                
                latent_list.append(latent_embd)
            
            agent2_current_latents = torch.cat(latent_list, dim=1)  # [B, num_latent, H]
            agent2_gen_latents.append(agent2_current_latents)
            
            # 4. Generate answer
            agent2_decoder_embeds = self.get_embd(self.codi, self.model_name)(agent2_decoder_ids)
            
            dynamic_mask = None
            if self.fix_attn_mask:
                dynamic_mask = torch.cat([
                    agent2_encoder_mask,
                    torch.ones((batch_size, self.num_latent), device=device),
                    torch.ones((batch_size, agent2_decoder_embeds.size(1)), device=device)
                ], dim=1).bool()
            
            outputs = self.codi(
                inputs_embeds=agent2_decoder_embeds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=agent2_past_kv,
                attention_mask=dynamic_mask
            )
            agent2_past_kv = outputs.past_key_values
            
            # 5. Compute agent2 losses
            # CE loss
            logits = outputs.logits
            labels = agent2_labels
            masked_labels = labels.clone()
            masked_labels[~sample_should_compute] = IGNORE_INDEX

            valid_token_count = (masked_labels[:, 1:] != IGNORE_INDEX).sum()
            if valid_token_count > 0:
                ce_loss = self.loss_fct(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    masked_labels[:, 1:].reshape(-1)
                )
            else:
                ce_loss = torch.tensor(0.0, device=device).to(logits)
            total_ce_loss += ce_loss
                
            # Distill loss
            # distill_loss = 0
            distill_loss = torch.tensor(0.0, device=device).to(ce_loss)
            for j, (out, ref_out) in enumerate(zip(outputs.hidden_states, agent2_ref_outputs.hidden_states)):
                ref_selected = ref_out.gather(
                    1, agent2_cur_ref_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1))
                )
                out_selected = out.gather(
                    1, agent2_cur_model_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1))
                )
                valid = sample_should_compute
                if valid.any():
                    distill_loss_tmp = self.distill_loss_fct(
                        out_selected[valid], ref_selected[valid].detach()
                    )

                    if self.distill_loss_div_std:
                        std = ref_selected[valid].std()
                        if std>0:
                            if self.distill_loss_type == 'l2':
                                distill_loss_tmp /= std
                            distill_loss_tmp /= std
                    distill_loss += distill_loss_tmp
                
            distill_loss /= len(outputs.hidden_states)
            distill_loss *=self.distill_loss_factor
            total_distill_loss += distill_loss
                
            # Decode loss
            if self.use_decode:
                decode_loss = self.decoder_forward(
                agent2_current_latents, agent2_long_decoder_ids, agent2_long_labels
                )
            else:
                decode_loss = torch.tensor(0).to(ce_loss)
            decode_loss *= self.decode_loss_factor
            total_decode_loss += decode_loss
                
            # record per-turn loss
            turn_ce_loss = ce_loss
            turn_distill_loss = distill_loss
            turn_decode_loss = decode_loss
            turn_total_loss = (
                turn_ce_loss
                + turn_distill_loss
                + turn_decode_loss
            ) 
            loss_detail["agent2_student_ce"].append(turn_ce_loss.detach().item())
            loss_detail["agent2_distill"].append(turn_distill_loss.detach().item())
            loss_detail["agent2_decode"].append(turn_decode_loss.detach().item())
            loss_detail["agent2_total"].append(turn_total_loss.detach().item())            

        
        # Total loss
        loss = total_ce_loss + total_distill_loss + total_ref_loss + total_decode_loss
        
        if self.print_loss:
            print(f'total_ce={total_ce_loss:.4f}, distill={total_distill_loss:.4f}, '
                f'ref={total_ref_loss:.4f}, decode={total_decode_loss:.4f}')

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": total_ce_loss.detach().item(),
            "distill_loss": total_distill_loss.detach().item(),
            "ref_ce_loss": total_ref_loss.detach().item(),
            "decode_loss": total_decode_loss.detach().item(),
            "loss_detail": loss_detail
        }
        
    def _inject_other_agent_latent(
        self,
        encoder_input_ids: torch.Tensor,  # [B, L]
        encoder_embeds: torch.Tensor,  # [B, L, H]
        other_latents: torch.Tensor,  # [B, num_latent, H]
    ):
        """
        fill the latent into input_embeds
        """
        batch_size = other_latents.size(0)
        for i in range(batch_size):
            latent_flag = encoder_input_ids[i]==self.latent_id
            encoder_embeds[i][latent_flag] = other_latents[i].to(encoder_embeds).detach()
        return encoder_embeds